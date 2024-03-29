use manifold::activation::ActivationType;
use manifold::f;
use manifold::substrates::blame_graph::{Manifold, Neuron, Signal, Trainer};
use manifold::substrates::traits::SignalConversion;
use rand::{prelude::*, thread_rng};

fn gen_training_data(size: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut x: Vec<Vec<f64>> = vec![];
    let mut y: Vec<Vec<f64>> = vec![];
    let mut rng = thread_rng();

    let classes: Vec<(Vec<f64>, Vec<f64>)> = vec![
        (vec![0., 1.], vec![1.]),
        (vec![1., 1.], vec![0.]),
        (vec![1., 0.], vec![0.]),
        (vec![0., 0.], vec![1.]),
    ];

    for _ in 0..size {
        let data = classes.choose(&mut rng).unwrap();
        x.push(data.0.clone());
        y.push(data.1.clone());
    }

    (x, y)
}

fn train() {
    let (train_x, train_y) = gen_training_data(5000);
    let (test_x, test_y) = gen_training_data(100);

    let (neuros, mesh_len) =
        Neuron::load_substrate_or_create("xor", 1000000, 0.0..1.0, ActivationType::Relu);

    let mut manifold = Manifold::new(mesh_len, 2, 1, vec![10, 20, 10]);
    manifold.weave();

    let mut trainer = Trainer::new(&train_x, &train_y);

    trainer
        .set_sample_size(8)
        .set_epochs(300)
        .set_rate(0.005)
        // .set_post_processor(f::softmax)
        .set_loss_fn(f::mean_squared_error)
        .train(&mut manifold, &neuros)
        .loss_graph();

    let mut predictions: Vec<usize> = vec![];
    let actual = test_y
        .into_iter()
        .map(|x| f::argmax(&x))
        .collect::<Vec<usize>>();

    for test_xv in test_x.into_iter() {
        let mut signals = Signal::signalize(test_xv);
        manifold.forward(&mut signals, &neuros);
        Signal::transform_output_slice(&mut signals, f::softmax);
        let prediction = f::argmax(&Signal::vectorize(signals));
        predictions.push(prediction);
    }

    let accuracy = f::accuracy::<usize>(&predictions, &actual);

    println!("{}% Accuracy", accuracy);
}

fn main() {
    train()
}
