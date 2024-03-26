use crate::{Manifold, Neuron, Signal};
use std::collections::VecDeque;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::thread::{self, available_parallelism};

#[derive(Clone)]
pub struct Basis {
    pub neuros: Arc<Vec<Neuron>>,
    pub x: Vec<Vec<f64>>,
    pub y: Vec<Vec<f64>>,
}

pub struct EvolutionHyper {
    pub population_size: usize,
    pub carryover_rate: f64,
    pub elitism_carryover: usize,
    pub sample_size: usize,
}

pub type Population = VecDeque<Arc<Mutex<Manifold>>>;

pub trait Optimizer {
    fn vectorize(signals: &[Signal]) -> Vec<f64> {
        signals.iter().map(|s| s.x.clone()).collect::<Vec<f64>>()
    }

    fn signalize(x: &[f64]) -> Vec<Signal> {
        x.iter()
            .map(|x| Signal { x: x.clone() })
            .collect::<Vec<Signal>>()
    }

    fn distributed<T: Send + 'static>(tasks: Vec<Box<dyn (FnOnce() -> T) + Send>>) -> Vec<T> {
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

    fn mse(signals: &[f64], target: &[f64]) -> f64 {
        let diff = target
            .iter()
            .enumerate()
            .map(|(i, e)| (e - signals[i]).powi(2))
            .collect::<Vec<f64>>();

        diff.into_iter().fold(0. as f64, |a, e| a + e) / signals.len() as f64
    }

    fn out(
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

    fn train(&self, basis: Basis, hyper: EvolutionHyper) -> (Population, Basis, EvolutionHyper);
}
