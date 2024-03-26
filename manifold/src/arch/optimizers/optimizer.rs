use std::thread::JoinHandle;
use std::thread::{self, available_parallelism};

#[derive(Clone)]
pub struct Basis<Substrate> {
    pub neuros: Substrate,
    pub x: Vec<Vec<f64>>,
    pub y: Vec<Vec<f64>>,
}

pub struct EvolutionHyper {
    pub population_size: usize,
    pub carryover_rate: f64,
    pub elitism_carryover: usize,
    pub sample_size: usize,
}

pub trait Optimizer<Substrate, Population> {
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

    fn train(
        &self,
        basis: Basis<Substrate>,
        hyper: EvolutionHyper,
    ) -> (Population, Basis<Substrate>, EvolutionHyper);
}
