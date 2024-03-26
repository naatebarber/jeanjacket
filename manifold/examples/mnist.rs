use mnist::*;
use std::{
    fs,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use manifold::{
    constant_fold::{Basis, Hyper},
    ConstantFold, Manifold, Neuron, Signal,
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
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(5_000)
        .validation_set_length(1_000)
        .test_set_length(1_000)
        .finalize();

    if !Path::exists(
        PathBuf::from_str("./.models/mnist.manifold.json")
            .unwrap()
            .as_path(),
    ) {
        // if true {
        let (x, y) = normalize_mnist_xy(trn_img, trn_lbl);

        let neuros = Neuron::substrate(100000, 0.0..1.0);

        let cf = ConstantFold::new(784, 9, vec![20, 50, 20]);

        let (mut population, basis, ..) = cf.optimize_traversal(
            Basis {
                neuros: Arc::new(neuros),
                x,
                y,
            },
            Hyper {
                population_size: 200,
                carryover_rate: 0.2,
                elitism_carryover: 3,
                sample_size: 20,
            },
        );

        ConstantFold::out("./.models/mnist", &mut population, basis.neuros).unwrap();
    }

    let (tx, ty) = normalize_mnist_xy(tst_img, tst_lbl);

    let serial_manifold = fs::read_to_string("./.models/mnist.manifold.json").unwrap();
    let serial_substrate = fs::read_to_string("./.models/mnist.substrate.json").unwrap();
    let manifold = Manifold::load(serial_manifold).unwrap();
    let neuros = Neuron::load_substrate(serial_substrate).unwrap();

    println!("OS {}", manifold.web[manifold.web.len() - 1].len());

    let mut winrate: Vec<f64> = vec![];

    for (i, mut x) in tx.into_iter().enumerate() {
        let mut signals = ConstantFold::signalize(&mut x);
        manifold.forward(&mut signals, &neuros);

        let targ = ConstantFold::signalize(&ty[i]);

        println!("{}", signals.len());

        let prediction = Signal::argmax(&signals);
        let actual = Signal::argmax(&targ);

        if prediction == actual {
            winrate.push(1.);
            println!("{} === {}", prediction, actual);
        } else {
            winrate.push(0.);
            println!("{} =/= {}", prediction, actual);
        }

        break;
    }

    let acc = winrate.iter().fold(0., |a, v| a + v) / winrate.len() as f64;
    println!("Final accuracy on MNIST dataset: {}%", acc * 100.);
}
