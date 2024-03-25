use std::ops::Deref;
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

    pub fn distributed(tasks: Vec<Box<dyn FnOnce() + Send>>) -> Vec<JoinHandle<()>> {
        let cores: usize = available_parallelism().unwrap().into();
        let mut batches: Vec<Vec<Box<dyn FnOnce() + Send>>> = Vec::with_capacity(cores);

        for _ in 0..cores {
            batches.push(vec![]);
        }

        for (i, task) in tasks.into_iter().enumerate() {
            let batch = i % cores;
            batches[batch].push(task);
        }

        batches
            .into_iter()
            .map(|mut batch| {
                thread::spawn(move || {
                    for task in batch.drain(..) {
                        task()
                    }
                })
            })
            .collect()
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
        &self,
        mut from_backtrack: Option<(Arc<Mutex<Manifold>>, &usize)>,
        neuros: &Vec<Neuron>,
        count: usize,
    ) -> VecDeque<Arc<Mutex<Manifold>>> {
        let mut p: VecDeque<Arc<Mutex<Manifold>>> = VecDeque::new();

        for _ in 0..count {
            match &mut from_backtrack {
                Some((manifold, backtrack)) => {
                    let mut parent = manifold.lock().unwrap();
                    let child = parent.reweave_backtrack(backtrack.clone());
                    p.push_back(Arc::new(Mutex::new(child)))
                }
                None => {
                    let mut organic =
                        Manifold::new(self.d_in, self.d_out, self.reach.clone(), neuros.len());
                    organic.weave();
                    p.push_back(Arc::new(Mutex::new(organic)))
                }
            }
        }

        p
    }

    pub fn mutate_population(
        &self,
        manifold: Arc<Mutex<Manifold>>,
        layer: Option<usize>,
        count: usize,
    ) -> VecDeque<Arc<Mutex<Manifold>>> {
        let mut p: VecDeque<Arc<Mutex<Manifold>>> = VecDeque::new();
        let mut parent = manifold.lock().unwrap();
        let mut rng = thread_rng();

        let layer = match layer {
            Some(x) => x,
            None => rng.gen_range(0..parent.get_num_layers()),
        };

        if layer >= parent.get_num_layers() {
            p.push_back(manifold.clone());
            return p;
        }

        for _ in 0..count {
            p.push_back(Arc::new(Mutex::new(parent.hotswap_single(layer))))
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
                manifold.forward(&mut x, neuros.deref());
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

        let handles = ConstantFold::distributed(tasks);
        for h in handles.into_iter() {
            h.join().ok();
        }

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

        let mut population: VecDeque<Arc<Mutex<Manifold>>> =
            self.weave_population(None, &neuros, population_size);
        let max_layers = population.iter().fold(0, |a, m| {
            let l = m.lock().unwrap().get_num_layers();
            if l > a {
                return l;
            }
            a
        });

        for backtrack in 0..max_layers - 1 {
            ConstantFold::evaluate(&mut population, &basis, &hyper);

            println!("Min loss: {}", population[0].lock().unwrap().loss);

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

            for (i, b) in evolvable.into_iter().enumerate() {
                next_population.append(&mut self.weave_population(
                    Some((b, &backtrack)),
                    &neuros,
                    children_per[i],
                ))
            }

            population = next_population;
        }

        return (population, basis, hyper);
    }

    /// Low touch, continuous single-neuron swap step for an already evolved pathway.
    pub fn constant_mutate(
        &self,
        mut population: VecDeque<Arc<Mutex<Manifold>>>,
        basis: Basis,
        hyper: Hyper,
        term_epochs: u64,
    ) {
        let Hyper {
            population_size,
            elitism_carryover,
            carryover_rate,
            ..
        } = hyper;

        let mut epochs = 0;
        let proceed = |epochs: &u64, term_epochs: &u64| {
            if *term_epochs == 0 {
                return true;
            }
            epochs < term_epochs
        };

        while proceed(&epochs, &term_epochs) {
            ConstantFold::evaluate(&mut population, &basis, &hyper);

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

            let mut children_per: Vec<usize> = carryover.iter().map(|_| 0).collect();
            let mut parent_selector = 0;
            for i in 0..=(population_size - elite.len()) {
                children_per[parent_selector] += 1;
                parent_selector = i % num_carryover;
            }

            let mut next_population: VecDeque<Arc<Mutex<Manifold>>> = VecDeque::new();
            next_population.append(&mut elite);

            for (i, b) in carryover.into_iter().enumerate() {
                // Mutate population
                next_population.append(&mut self.mutate_population(b, None, children_per[i]));
            }

            population = next_population;

            epochs += 1
        }
    }
}
