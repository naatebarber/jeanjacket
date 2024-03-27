use mnist::*;

use manifold::f;
use manifold::optimizers::{Basis, EvolutionHyper, Optimizer, Turnstile};
use manifold::substrates::fully_connected::{Neuron, Signal};
use manifold::substrates::traits::SignalConversion;

fn normalize_mnist_xy(x: Vec<u8>, y: Vec<u8>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let x = x.chunks_exact(784).collect::<Vec<&[u8]>>();
    let x = x
        .into_iter()
        .map(|image| {
            image
                .into_iter()
                .map(|pix| (*pix as f64) / 256.)
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();

    let y = y
        .into_iter()
        .map(|l| f::onehot(l, 9))
        .collect::<Vec<Vec<f64>>>();

    (x, y)
}

fn main() {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(5_000)
        .validation_set_length(1_000)
        .test_set_length(1_000)
        .finalize();

    let (x, y) = normalize_mnist_xy(trn_img, trn_lbl);

    let neuros = Neuron::substrate(100000, -10.0..10.0);

    let cf = Turnstile::new(784, 9, 5..20, 5..20, -2..2, 100);

    let (mut population, basis, ..) = cf.train(
        Basis { neuros, x, y },
        EvolutionHyper {
            population_size: 400,
            carryover_rate: 0.2,
            elitism_carryover: 40,
            sample_size: 20,
        },
    );

    let (tx, ty) = normalize_mnist_xy(tst_img, tst_lbl);

    let manifold_am = population.pop_front().unwrap();
    let manifold = manifold_am.lock().unwrap();
    let neuros = basis.neuros;

    let mut predictions: Vec<usize> = vec![];
    let mut actuals: Vec<usize> = vec![];

    for (i, x) in tx.into_iter().enumerate() {
        let mut signals = Signal::signalize(x);
        manifold.forward(&mut signals, &neuros);

        println!("{}", signals.len());

        let svecs = Signal::vectorize(signals);

        predictions.push(f::argmax(&svecs));
        actuals.push(f::argmax(&ty[i]));
    }

    println!("Accuracy: {}%", f::accuracy(&predictions, &actuals))
}
