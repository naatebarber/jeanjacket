use rand::{self, Rng};

#[derive(Debug)]
pub struct Signal {
    pub x: f64,
}

impl Signal {
    pub fn random_normal(size: usize) -> Vec<Signal> {
        let mut signals = vec![];
        let mut rng = rand::thread_rng();

        for _ in 0..size {
            signals.push(Signal { x: rng.gen() })
        }

        signals
    }
}
