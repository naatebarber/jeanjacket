use rand::{prelude::*, thread_rng};
use std::{fs, sync::Arc};

use manifold::{
    constant_fold::{Basis, Hyper},
    ConstantFold, Manifold, Neuron,
};

fn main() {
    let neuros = Neuron::substrate(100000, -2.0..2.0);

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

    let cf = ConstantFold::new(1, 1, vec![10, 5, 10, 100, 10, 10, 5]);

    let (mut population, basis, ..) = cf.optimize_traversal(
        Basis {
            neuros: Arc::new(neuros),
            x,
            y,
        },
        Hyper {
            population_size: 1000,
            carryover_rate: 0.2,
            elitism_carryover: 50,
            sample_size: 40,
        },
    );

    ConstantFold::out("./.models/approx_eq", &mut population, basis.neuros).unwrap();

    let serial_manifold = fs::read_to_string("./.models/approx_eq.manifold.json").unwrap();
    let serial_substrate = fs::read_to_string("./.models/approx_eq.substrate.json").unwrap();
    let manifold = Manifold::load(serial_manifold).unwrap();
    let neuros = Neuron::load_substrate(serial_substrate).unwrap();

    let x = 0.6;
    println!("From heuristic with x of {}: {}", x, heuristic(&x));

    let vex = vec![x];
    let mut signals = ConstantFold::signalize(&vex);
    manifold.forward(&mut signals, &neuros);
    println!("From manifold with x of {}: {}", x, signals[0].x);
}
