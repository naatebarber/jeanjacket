use std::sync::Arc;

use manifold::constant_fold::{Basis, Hyper};
use manifold::{ConstantFold, Manifold, Neuron, Signal};
use rand::{thread_rng, Rng};

fn main() {
    let neuros = Arc::new(Neuron::substrate(100, 0.0..1.0));

    let mut x: Vec<Vec<f64>> = vec![];
    let mut y: Vec<Vec<f64>> = vec![];
    let mut rng = thread_rng();

    for _ in 0..100000 {
        let decide: f64 = rng.gen();
        let mut ma: f64 = 1.;
        if decide < 0.5 {
            ma = 0.
        }

        x.push(vec![ma]);
        y.push(vec![ma]);
    }

    let tx = x[0..10].to_vec();
    let ty = y[0..10].to_vec();

    let cf = ConstantFold::new(1, 1, vec![2, 8, 3]);
    let (mut population, basis, hyper) = cf.optimize_traversal(
        Basis {
            neuros: Arc::clone(&neuros),
            x,
            y,
        },
        Hyper {
            population_size: 1,
            carryover_rate: 1.,
            elitism_carryover: 0,
            sample_size: 40,
        },
    );

    let manifold = population.pop_front().unwrap();
    let manifold = manifold.lock().unwrap();

    for (i, x) in tx.iter().enumerate() {
        let mut signals = ConstantFold::signalize(x);
        manifold.forward(&mut signals, &neuros);

        let target = ConstantFold::signalize(&ty[i]);

        if target.len() != signals.len() {
            println!(
                "Output size mismatch. TARGET {} SIGNALS {}",
                target.len(),
                signals.len()
            );
            continue;
        }

        if target[0].x == signals[0].x {
            println!("PASS")
        } else {
            println!("FAIL")
        }
    }
}
