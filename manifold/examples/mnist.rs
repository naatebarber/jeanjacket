use std::sync::Arc;

use manifold::activation::ActivationType;
use mnist::*;

use manifold::f;
use manifold::substrates::fully_connected::{Manifold, Neuron, Signal, Trainer};
use manifold::substrates::traits::SignalConversion;

fn _print_x_y(x: &Vec<f64>, y: &Vec<f64>) {
    x.chunks_exact(28).for_each(|c| {
        c.iter().for_each(|v| print!("{}", v.round()));
        println!("");
    });

    println!("{}", f::argmax(&y));
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
        .map(|l| f::onehot(l, 10))
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
    let (test_x, test_y) = normalize_mnist_xy(tst_img, tst_lbl);

    let neuros = Neuron::substrate(100000, -2.0..2.0, ActivationType::Relu);
    let mut manifold = Manifold::new(100000 - 1, 784, 10, vec![1000, 1200, 64]);
    manifold.weave();

    let mut trainer = Trainer::new(&train_x, &train_y);
    let post_processor = Arc::new(|sig| {
        let vecs = Signal::vectorize(sig);
        let softmax = f::softmax(&vecs);
        softmax
    });

    let loss_fn = Arc::new(f::binary_cross_entropy);

    trainer
        .set_sample_size(80)
        .set_epochs(300)
        .set_amplitude(20)
        .train(
            &mut manifold,
            &neuros,
            post_processor.clone(),
            loss_fn.clone(),
        );

    let mut winrate: Vec<u32> = vec![];

    for (i, x) in test_x.into_iter().enumerate() {
        let mut signals = Signal::signalize(x);
        manifold.forward(&mut signals, neuros.clone());

        let svecs = Signal::vectorize(signals);

        let win = f::argmax(&svecs) == f::argmax(&test_y[i]);

        if win {
            winrate.push(1);
        } else {
            winrate.push(0);
        }
    }

    println!(
        "Accuracy: {}",
        winrate.iter().fold(0, |a, x| a + x) as f64 / winrate.len() as f64
    );
}
