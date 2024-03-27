use std::collections::VecDeque;

use rand::{self, Rng};

use crate::substrates::traits::SignalConversion;

#[derive(Debug)]
pub struct Signal {
    pub x: f64,
}

impl SignalConversion<Signal> for Signal {
    fn signalize(vec: Vec<f64>) -> VecDeque<Signal> {
        vec.into_iter().map(|x| Signal { x }).collect()
    }

    fn vectorize(sig: VecDeque<Signal>) -> Vec<f64> {
        sig.into_iter().map(|x| x.x).collect()
    }
}

impl Signal {
    pub fn _random_normal(size: usize) -> Vec<Signal> {
        let mut signals = vec![];
        let mut rng = rand::thread_rng();

        for _ in 0..size {
            signals.push(Signal { x: rng.gen() })
        }

        signals
    }

    pub fn argmax(signals: &[Signal]) -> usize {
        if signals.len() < 1 {
            return 0;
        }

        let mut max_ix = 0;
        let mut max = signals[0].x;

        for (i, s) in signals.iter().enumerate() {
            if s.x > max {
                max = s.x;
                max_ix = i;
            }
        }

        max_ix
    }
}
