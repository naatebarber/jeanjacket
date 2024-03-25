use rand::{prelude::*, thread_rng};
use std::sync::Arc;

use manifold::{
    constant_fold::{Basis, Hyper},
    ConstantFold, Neuron,
};

fn main() {
    let neuros = Neuron::substrate(100000);

    let heuristic = |x: &f64| (3. * x.powi(3) + x.powi(2)) / 0.14678;
    let mut x: Vec<Vec<f64>> = vec![];
    let mut y: Vec<Vec<f64>> = vec![];
    let dssize = 10000;
    let mut rng = thread_rng();

    for _ in 0..dssize {
        let i: f64 = rng.gen();
        let o = heuristic(&i);
        x.push(vec![i]);
        y.push(vec![o]);
    }

    let cf = ConstantFold::new(1, 1, vec![10, 5, 10, 50, 10, 10, 5]);

    let _ = cf.optimize_traversal(
        Basis {
            neuros: Arc::new(neuros),
            x,
            y,
        },
        Hyper {
            population_size: 100,
            carryover_rate: 0.2,
            elitism_carryover: 5,
            sample_size: 40,
        },
    );
}
