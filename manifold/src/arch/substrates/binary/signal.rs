use rand::{self, Rng};

#[derive(Debug)]
pub struct Signal {
    pub x: f64,
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
