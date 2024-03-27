use manifold::substrates::fully_connected::{Manifold, Neuron, Signal, Trainer};
use manifold::substrates::traits::SignalConversion;
use manifold::{activation::ActivationType, f};
use rand::{thread_rng, Rng};
use std::sync::Arc;

fn zero_two() {
    fn gen_binary_training_data(size: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut x: Vec<Vec<f64>> = vec![];
        let mut y: Vec<Vec<f64>> = vec![];
        let mut rng = thread_rng();

        for _ in 0..size {
            let dec: bool = rng.gen_bool(0.5);

            let (xv, yv) = match dec {
                true => (vec![1.], vec![1., 0.]),
                false => (vec![0.], vec![0., 1.]),
            };

            x.push(xv);
            y.push(yv);
        }

        (x, y)
    }

    let (sx, sy) = gen_binary_training_data(5);
    println!("{:?} {:?}", sx, sy);

    let as_argmax = sy
        .into_iter()
        .map(|x| f::argmax(&x))
        .collect::<Vec<usize>>();

    println!("{}", f::accuracy::<usize>(&as_argmax, &as_argmax));

    let (train_x, train_y) = gen_binary_training_data(5000);
    let (test_x, test_y) = gen_binary_training_data(100);

    let neuros = Neuron::substrate(100000, -0.0..2.0, ActivationType::Relu);
    let mut manifold = Manifold::new(100000 - 1, 1, 2, vec![10, 20, 10]);
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
    let actual = test_y
        .into_iter()
        .map(|x| f::argmax(&x))
        .collect::<Vec<usize>>();

    for test_xv in test_x.into_iter() {
        let mut signals = Signal::signalize(test_xv);
        manifold.forward(&mut signals, Arc::clone(&neuros));

        let prediction = post_processor(signals);

        predictions.push(f::argmax(&prediction));
    }

    println!("{:?}", predictions);
    println!("{:?}", actual);

    let accuracy = f::accuracy::<usize>(&predictions, &actual);

    println!("{}% Accuracy", accuracy);
}

fn main() {
    zero_two()
}
