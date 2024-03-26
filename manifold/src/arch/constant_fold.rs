use std::error::Error;
use std::fs;
use std::ops::Deref;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::thread::{self, available_parallelism};
use std::{collections::VecDeque, thread::JoinHandle};

use rand::{thread_rng, Rng};

use super::{Manifold, Neuron, Signal};

#[derive(Clone)]
pub struct Basis {
    pub neuros: Arc<Vec<Neuron>>,
    pub x: Vec<Vec<f64>>,
    pub y: Vec<Vec<f64>>,
}

pub struct Hyper {
    pub population_size: usize,
    pub carryover_rate: f64,
    pub elitism_carryover: usize,
    pub sample_size: usize,
}

pub struct ConstantFold {
    d_in: usize,
    d_out: usize,
    reach: Vec<usize>,
}

impl ConstantFold {
    pub fn new(d_in: usize, d_out: usize, reach: Vec<usize>) -> ConstantFold {
        ConstantFold { d_in, d_out, reach }
    }

    pub fn vectorize(signals: &[Signal]) -> Vec<f64> {
        signals.iter().map(|s| s.x.clone()).collect::<Vec<f64>>()
    }

    pub fn signalize(x: &[f64]) -> Vec<Signal> {
        x.iter()
            .map(|x| Signal { x: x.clone() })
            .collect::<Vec<Signal>>()
    }

    pub fn distributed<T: Send + 'static>(tasks: Vec<Box<dyn (FnOnce() -> T) + Send>>) -> Vec<T> {
        let cores: usize = available_parallelism().unwrap().into();
        let mut batches: Vec<Vec<Box<dyn (FnOnce() -> T) + Send>>> = Vec::with_capacity(cores);

        for _ in 0..cores {
            batches.push(vec![]);
        }

        for (i, task) in tasks.into_iter().enumerate() {
            let batch = i % cores;
            batches[batch].push(task);
        }

        let handles: Vec<JoinHandle<Vec<T>>> = batches
            .into_iter()
            .map(|mut batch| {
                thread::spawn(move || {
                    let mut results: Vec<T> = vec![];
                    for task in batch.drain(..) {
                        let r: T = task();
                        results.push(r)
                    }
                    results
                })
            })
            .collect();

        let mut results: Vec<T> = vec![];
        for handle in handles.into_iter() {
            if let Some(mut v) = handle.join().ok() {
                results.append(&mut v);
            }
        }

        results
    }

    pub fn mse(signals: &[Signal], target: &[f64]) -> f64 {
        let svec = ConstantFold::vectorize(signals);

        let diff = target
            .iter()
            .enumerate()
            .map(|(i, e)| (e - svec[i]).powi(2))
            .collect::<Vec<f64>>();

        diff.into_iter().fold(0. as f64, |a, e| a + e) / signals.len() as f64
    }

    pub fn weave_population(
        reach: Vec<usize>,
        d_in: usize,
        d_out: usize,
        neurons: usize,
        count: usize,
    ) -> VecDeque<Arc<Mutex<Manifold>>> {
        let mut p: VecDeque<Arc<Mutex<Manifold>>> = VecDeque::new();

        let mut weave_tasks: Vec<Box<dyn (FnOnce() -> Arc<Mutex<Manifold>>) + Send>> = vec![];

        for _ in 0..count {
            let d_inc = d_in.clone();
            let d_outc = d_out.clone();
            let nc = neurons.clone();
            let reachc = reach.clone();

            let task = Box::new(move || {
                let mut organic = Manifold::new(d_inc, d_outc, reachc, nc);
                organic.weave();
                Arc::new(Mutex::new(organic))
            }) as Box<dyn (FnOnce() -> Arc<Mutex<Manifold>>) + Send>;

            weave_tasks.push(task);
        }

        let woven_manifolds = ConstantFold::distributed::<Arc<Mutex<Manifold>>>(weave_tasks);

        for m in woven_manifolds.into_iter() {
            p.push_back(m)
        }

        p
    }

    pub fn reweave_population(
        from: Arc<Mutex<Manifold>>,
        backtrack: usize,
        count: usize,
    ) -> VecDeque<Arc<Mutex<Manifold>>> {
        let mut p: VecDeque<Arc<Mutex<Manifold>>> = VecDeque::new();

        for _ in 0..count {
            let mut parent = from.lock().unwrap();
            let child = parent.reweave_backtrack(backtrack.clone());
            p.push_back(Arc::new(Mutex::new(child)));
        }

        p
    }

    pub fn evaluate_one(
        manifold_am: Arc<Mutex<Manifold>>,
        sample_x: Vec<Vec<f64>>,
        sample_y: Vec<Vec<f64>>,
        neuros: Arc<Vec<Neuron>>,
    ) {
        let mut manifold = manifold_am.lock().unwrap();

        sample_x
            .into_iter()
            .map(|x| ConstantFold::signalize(&x))
            .enumerate()
            .for_each(|(i, mut x)| {
                manifold.forward(&mut x, &neuros);
                if x.len() != sample_y[i].len() {
                    panic!("Output malformed")
                }
                let loss = ConstantFold::mse(&mut x, &sample_y[i]);
                manifold.accumulate_loss(loss)
            });
    }

    pub fn evaluate(population: &mut VecDeque<Arc<Mutex<Manifold>>>, basis: &Basis, hyper: &Hyper) {
        let mut rng = thread_rng();
        let set_length = basis.x.len();
        let mut sample_x: Vec<Vec<f64>> = vec![];
        let mut sample_y: Vec<Vec<f64>> = vec![];

        for _ in 0..hyper.sample_size {
            let xy_ix = rng.gen_range(0..set_length);
            sample_x.push(basis.x[xy_ix].clone());
            sample_y.push(basis.y[xy_ix].clone());
        }

        let tasks = population
            .iter()
            .map(|m| {
                let sx = sample_x.clone();
                let sy = sample_y.clone();
                let neuros = Arc::clone(&basis.neuros);
                let manifold = Arc::clone(&m);
                Box::new(move || ConstantFold::evaluate_one(manifold, sx, sy, neuros))
                    as Box<dyn FnOnce() + Send>
            })
            .collect::<Vec<Box<dyn FnOnce() + Send>>>();

        let _ = ConstantFold::distributed::<()>(tasks);

        population.make_contiguous().sort_unstable_by(|m, n| {
            let m = m.lock().unwrap();
            let n = n.lock().unwrap();
            m.loss.partial_cmp(&n.loss).unwrap()
        });
    }

    /// Brute evolution of a pathway over a neuron mesh, not continuous. One pass.
    pub fn optimize_traversal(
        &self,
        basis: Basis,
        hyper: Hyper,
    ) -> (VecDeque<Arc<Mutex<Manifold>>>, Basis, Hyper) {
        let Basis { neuros, .. } = basis.clone();
        let Hyper {
            population_size,
            carryover_rate,
            elitism_carryover,
            ..
        } = hyper;

        let mut population: VecDeque<Arc<Mutex<Manifold>>> = ConstantFold::weave_population(
            self.reach.clone(),
            self.d_in,
            self.d_out,
            neuros.len(),
            population_size,
        );

        let get_max_layers = |population: &VecDeque<Arc<Mutex<Manifold>>>| {
            population.iter().fold(0, |a, m| {
                let l = m.lock().unwrap().get_num_layers();
                if l > a {
                    return l;
                }
                a
            })
        };

        let mut max_layers = get_max_layers(&population);
        let mut backtrack = 0;

        loop {
            ConstantFold::evaluate(&mut population, &basis, &hyper);

            println!(
                "({}/{}) Min loss (pathway optim): {}",
                backtrack,
                max_layers - 1,
                population[0].lock().unwrap().loss
            );

            // Carry over the elite
            let mut elite: VecDeque<Arc<Mutex<Manifold>>> = VecDeque::new();
            for _ in 0..elitism_carryover {
                let m = population.pop_front();
                if let Some(m) = m {
                    elite.push_back(m)
                }
            }

            let mut carryover: VecDeque<Arc<Mutex<Manifold>>> = VecDeque::new();
            let num_carryover = (carryover_rate * population_size as f64).floor() as usize;
            for _ in 0..num_carryover {
                let m = population.pop_front();
                if let Some(m) = m {
                    carryover.push_back(m)
                }
            }

            // Append the carryover manifolds that can no longer evolve to elite
            // If inviable, they will wash out.
            // Evolve the rest
            let mut evolvable: VecDeque<Arc<Mutex<Manifold>>> = VecDeque::new();
            for m in carryover.into_iter() {
                let manifold = m.lock().unwrap();
                if manifold.get_num_layers() < backtrack + 1 {
                    drop(manifold);
                    elite.push_back(m);
                    continue;
                }
                drop(manifold);
                evolvable.push_back(m);
            }

            let num_evolvable = evolvable.len();

            if num_evolvable == 0 {
                break;
            }

            let mut children_per: Vec<usize> = evolvable.iter().map(|_| 0).collect();

            let mut parent_selector = 0;
            for i in 0..=(population_size - elite.len()) {
                children_per[parent_selector] += 1;
                parent_selector = i % num_evolvable;
            }

            let mut next_population: VecDeque<Arc<Mutex<Manifold>>> = VecDeque::new();
            next_population.append(&mut elite);

            type PopulationSlice = VecDeque<Arc<Mutex<Manifold>>>;

            let create_population_tasks = evolvable
                .into_iter()
                .enumerate()
                .map(|(i, manifold)| {
                    let m = Arc::clone(&manifold);
                    let count = children_per[i];
                    let bt = backtrack.clone();

                    Box::new(move || ConstantFold::reweave_population(m, bt, count))
                        as Box<dyn (FnOnce() -> PopulationSlice) + Send>
                })
                .collect::<Vec<_>>();

            let population_slices =
                ConstantFold::distributed::<PopulationSlice>(create_population_tasks);

            for mut slice in population_slices.into_iter() {
                next_population.append(&mut slice)
            }

            population = next_population;
            backtrack += 1;
            max_layers = get_max_layers(&population);

            if backtrack >= max_layers {
                break;
            }
        }

        return (population, basis, hyper);
    }

    pub fn out(
        tag: &str,
        population: &mut VecDeque<Arc<Mutex<Manifold>>>,
        neuros: Arc<Vec<Neuron>>,
    ) -> Result<(), Box<dyn Error>> {
        population.make_contiguous().sort_unstable_by(|m, n| {
            let m = m.lock().unwrap();
            let n = n.lock().unwrap();
            m.loss.partial_cmp(&n.loss).unwrap()
        });

        let manifold_fname = format!("{}.manifold.json", tag);

        match population.pop_front() {
            Some(manifold) => {
                let m = manifold.lock().unwrap();
                let serial = m.dump()?;
                fs::write(PathBuf::from_str(&manifold_fname)?, serial)?;
            }
            None => (),
        };

        let substrate_fname = format!("{}.substrate.json", tag);

        let substrate_serial = Neuron::dump_substrate(neuros)?;
        fs::write(PathBuf::from_str(&substrate_fname)?, substrate_serial)?;

        Ok(())
    }
}
