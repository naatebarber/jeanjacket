use manifold::activation::ActivationType;
use mnist::*;

use manifold::f;
use manifold::substrates::fully_connected::{Manifold, Neuron};

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
    let (neuros, mesh_len) =
        Neuron::load_substrate_or_create("basis", 1000000, -1.0..1.0, ActivationType::Relu);
}
