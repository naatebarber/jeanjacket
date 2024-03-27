use crate::substrates::traits::SignalConversion;
use std::collections::VecDeque;

use rand::{thread_rng, Rng};

#[derive(Clone, Debug)]
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
    pub fn _random_normal(size: usize) -> VecDeque<Signal> {
        let mut signals = VecDeque::new();
        let mut rng = thread_rng();

        for _ in 0..size {
            signals.push_back(Signal { x: rng.gen() })
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
