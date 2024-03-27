use std::sync::Arc;

use manifold::activation::ActivationType;
use mnist::*;

use manifold::f;
use manifold::substrates::fully_connected::{Manifold, Neuron, Signal, Trainer};
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

    let (train_x, train_y) = normalize_mnist_xy(trn_img, trn_lbl);

    let neuros = Neuron::substrate(1000000, -10.0..10.0, ActivationType::Relu);
    let mut manifold = Manifold::new(100000 - 1, 784, 9, vec![100, 30, 10, 40, 10]);
    manifold.weave();

    let mut trainer = Trainer::new(&train_x, &train_y);
    let post_processor = Arc::new(|sig| {
        let vecs = Signal::vectorize(sig);
        let softmax = f::softmax(&vecs);
        softmax
    });

    let loss_fn = Arc::new(f::binary_cross_entropy);

    trainer
        .set_sample_size(40)
        .set_epochs(300)
        .set_amplitude(2)
        .train(
            &mut manifold,
            &neuros,
            post_processor.clone(),
            loss_fn.clone(),
        );

    let mut predictions: Vec<usize> = vec![];
    let mut actuals: Vec<usize> = vec![];

    let (test_x, test_y) = normalize_mnist_xy(tst_img, tst_lbl);

    for (i, x) in test_x.into_iter().enumerate() {
        let mut signals = Signal::signalize(x);
        manifold.forward(&mut signals, neuros.clone());

        let svecs = Signal::vectorize(signals);

        predictions.push(f::argmax(&svecs));
        actuals.push(f::argmax(&test_y[i]));
    }

    println!("Accuracy: {}%", f::accuracy(&predictions, &actuals))
}
