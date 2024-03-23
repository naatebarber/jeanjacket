use super::Signal;

pub struct ConstantFold {
    d_in: usize,
    d_out: usize,
    heuristic: Box<dyn Fn(Vec<Signal>) -> Vec<Signal>>,
}

impl ConstantFold {
    pub fn new(
        d_in: usize,
        d_out: usize,
        heuristic: Box<dyn Fn(Vec<Signal>) -> Vec<Signal>>,
    ) -> ConstantFold {
        ConstantFold {
            d_in,
            d_out,
            heuristic,
        }
    }

    pub fn vectorize(signals: &Vec<Signal>) -> Vec<f64> {
        signals.iter().map(|s| s.x.clone()).collect::<Vec<f64>>()
    }

    pub fn mse(signals: &Vec<Signal>, target: &Vec<Signal>) -> f64 {
        let svec = ConstantFold::vectorize(signals);
        let tvec = ConstantFold::vectorize(target);

        let diff = tvec
            .clone()
            .iter()
            .enumerate()
            .map(|(i, e)| (e - svec[i]).powi(2))
            .collect::<Vec<f64>>();

        diff.into_iter().fold(0. as f64, |a, e| a + e) / signals.len() as f64
    }
}
