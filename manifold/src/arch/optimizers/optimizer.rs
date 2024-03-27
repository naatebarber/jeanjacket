use crate::f;

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
        f::distributed::<T>(tasks)
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
