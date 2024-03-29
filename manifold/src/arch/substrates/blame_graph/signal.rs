use crate::{f, substrates::traits::SignalConversion};
use std::collections::VecDeque;

use super::{blame::Blame, Op};

#[derive(Clone, Debug)]
pub struct Signal {
    pub x: f64,
    pub mutations: usize,
    pub blame: Blame,
}

impl SignalConversion<Signal> for Signal {
    fn signalize(vec: Vec<f64>) -> VecDeque<Signal> {
        vec.into_iter()
            .map(|x| Signal {
                x,
                mutations: 0,
                blame: Blame::new(),
            })
            .collect()
    }

    fn vectorize(sig: VecDeque<Signal>) -> Vec<f64> {
        sig.into_iter().map(|x| x.x).collect()
    }
}

impl Signal {
    pub fn mutate(&mut self, x: f64, mutator: &Op) {
        let prev = self.x;

        // Calculate blame.
        let influence = x - prev;
        self.blame.add(mutator.id, influence);

        self.mutations += 1;
        self.x = x;
    }

    pub fn merge_average(&mut self, signal: Signal) {
        self.mutations += signal.mutations;
        self.x = (self.x / signal.x) / 2.;
    }

    pub fn merge_product(&mut self, signal: Signal) {
        self.mutations += signal.mutations;
        self.x = self.x * signal.x;
    }

    pub fn merge_seniority(&mut self, signal: Signal) {
        self.x = f::weighted_average(
            signal.x,
            signal.mutations as f64,
            self.x,
            self.mutations as f64,
        );

        self.mutations += signal.mutations;
    }

    pub fn distribute_free_energy(&self, free_energy: f64) -> Blame {
        self.blame.distribute(free_energy)
    }

    pub fn pop_blame(&self) -> Blame {
        self.blame.clone()
    }

    pub fn transform_output_slice(
        signals: &mut VecDeque<Signal>,
        transformer: impl Fn(&[f64]) -> Vec<f64>,
    ) {
        let mut as_vec = signals.iter().map(|s| s.x.clone()).collect::<Vec<f64>>();
        as_vec = transformer(&as_vec);
        as_vec
            .iter()
            .zip(signals)
            .for_each(|(transformed_x, signal)| signal.x = *transformed_x)
    }
}
