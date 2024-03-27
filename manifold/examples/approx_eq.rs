use rand::{prelude::*, thread_rng};
use std::{fs, sync::Arc};

use manifold::optimizers::{Basis, EvolutionHyper, FixedReweave, Optimizer};
use manifold::substrates::binary::{Manifold, Neuron, Signal};
use manifold::substrates::traits::SignalConversion;

fn main() {
    let neuros = Neuron::substrate(10000, -2.0..2.0);

    let heuristic = |x: &f64| x * 2.;
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

    let optim = FixedReweave::new(1, 1, vec![1, 2, 3]);

    let (mut population, basis, ..) = optim.train(
        Basis {
            neuros: Arc::new(neuros),
            x,
            y,
        },
        EvolutionHyper {
            population_size: 100,
            carryover_rate: 0.02,
            elitism_carryover: 20,
            sample_size: 60,
        },
    );

    FixedReweave::out("./.models/approx_eq", &mut population, basis.neuros).unwrap();

    let serial_manifold = fs::read_to_string("./.models/approx_eq.manifold.json").unwrap();
    let serial_substrate = fs::read_to_string("./.models/approx_eq.substrate.json").unwrap();
    let manifold = Manifold::load(serial_manifold).unwrap();
    let neuros = Neuron::load_substrate(serial_substrate).unwrap();

    let x = 0.6;
    println!("From heuristic with x of {}: {}", x, heuristic(&x));

    let vex = vec![x];
    let mut signals = Vec::from(Signal::signalize(vex));
    manifold.forward(&mut signals, &neuros);
    println!("From manifold with x of {}: {}", x, signals[0].x);
}
