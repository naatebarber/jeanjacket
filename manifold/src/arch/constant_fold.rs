use std::ops::Deref;
use std::sync::{Arc, Mutex};
use std::thread::{self, available_parallelism};
use std::{collections::VecDeque, thread::JoinHandle};

use rand::{thread_rng, Rng};

use super::{Manifold, Neuron, Signal};

pub struct ConstantFold {
    d_in: usize,
    d_out: usize,
    reach: Vec<usize>,
    cores: usize,
}

impl ConstantFold {
    pub fn new(d_in: usize, d_out: usize, reach: Vec<usize>) -> ConstantFold {
        let cores: usize = available_parallelism().unwrap().into();
        ConstantFold {
            d_in,
            d_out,
            reach,
            cores,
        }
    }

    pub fn vectorize(signals: &[Signal]) -> Vec<f64> {
        signals.iter().map(|s| s.x.clone()).collect::<Vec<f64>>()
    }

    pub fn signalize(x: &[f64]) -> Vec<Signal> {
        x.iter()
            .map(|x| Signal { x: x.clone() })
            .collect::<Vec<Signal>>()
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

    pub fn population(
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

    pub fn evaluate_one(
        manifold_am: Arc<Mutex<Manifold>>,
        sample_x: Vec<Vec<f64>>,
        sample_y: Vec<Vec<f64>>,
        neuros: Arc<Vec<Neuron>>,
    ) -> JoinHandle<()> {
        let hand = thread::spawn(move || {
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
        });

        hand
    }

    pub fn optimize_traversal(&self, neuros: Arc<Vec<Neuron>>, x: Vec<Vec<f64>>, y: Vec<Vec<f64>>) {
        let mut rng = thread_rng();
        let set_length = x.len();

        let population_size = 100;
        let carryover_rate = 0.1;
        let elitism_carryover = 3;
        let samples_per_epoch = 40;

        // Also, maybe a hyperparam for splitting new populations based on prev gens
        // successful members. Do we just do an even split or is it based on performance?
        // For now an even split.

        let mut population: VecDeque<Arc<Mutex<Manifold>>> =
            self.population(None, &neuros, population_size);
        let max_layers = population.iter().fold(0, |a, m| {
            let l = m.lock().unwrap().get_num_layers();
            if l > a {
                return l;
            }
            a
        });

        // First training pass. bring everything down to a certain threshold of loss.

        for backtrack in 0..max_layers - 1 {
            let mut sample_x: Vec<Vec<f64>> = vec![];
            let mut sample_y: Vec<Vec<f64>> = vec![];

            for _ in 0..samples_per_epoch {
                let xy_ix = rng.gen_range(0..set_length);
                sample_x.push(x[xy_ix].clone());
                sample_y.push(y[xy_ix].clone());
            }

            let handles = population
                .iter()
                .map(|m| {
                    ConstantFold::evaluate_one(
                        Arc::clone(m),
                        sample_x.clone(),
                        sample_y.clone(),
                        Arc::clone(&neuros),
                    )
                })
                .collect::<Vec<JoinHandle<()>>>();

            for h in handles {
                h.join();
            }

            population.make_contiguous().sort_unstable_by(|m, n| {
                let m = m.lock().unwrap();
                let n = n.lock().unwrap();
                m.loss.partial_cmp(&n.loss).unwrap()
            });

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
            for i in 0..=population_size {
                children_per[parent_selector] += 1;
                parent_selector = i % num_evolvable;
            }

            let mut next_population: VecDeque<Arc<Mutex<Manifold>>> = VecDeque::new();
            for (i, b) in evolvable.into_iter().enumerate() {
                next_population.append(&mut self.population(
                    Some((b, &backtrack)),
                    &neuros,
                    children_per[i],
                ))
            }

            population = next_population;
        }
    }
}
