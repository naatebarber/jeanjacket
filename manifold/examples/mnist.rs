use mnist::*;
use std::sync::Arc;

use manifold::{
    constant_fold::{Basis, Hyper},
    ConstantFold, Neuron,
};

fn onehot(i: u8, m: u8) -> Vec<f64> {
    let mut oh = vec![0.; m.into()];
    if i < m {
        oh[i as usize] = 1.;
    }
    oh
}

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
        .map(|l| onehot(l, 9))
        .collect::<Vec<Vec<f64>>>();

    (x, y)
}

fn main() {
    let Mnist {
        trn_img,
        trn_lbl,
        // tst_img,
        // tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(5_000)
        .validation_set_length(1_000)
        .test_set_length(1_000)
        .finalize();

    let (x, y) = normalize_mnist_xy(trn_img, trn_lbl);

    let neuros = Neuron::substrate(1000000);

    let cf = ConstantFold::new(784, 9, vec![50]);

    let _ = cf.optimize_traversal(
        Basis {
            neuros: Arc::new(neuros),
            x,
            y,
        },
        Hyper {
            population_size: 10,
            carryover_rate: 0.2,
            elitism_carryover: 3,
            sample_size: 40,
        },
    );
}
