use manifold::activation::ActivationType;
use manifold::f;
use manifold::substrates::fully_connected::{Manifold, Neuron, Signal, Trainer};
use manifold::substrates::traits::SignalConversion;
use rand::{thread_rng, Rng};

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

fn zero_two() {
    let (train_x, train_y) = gen_binary_training_data(5000);
    let (test_x, test_y) = gen_binary_training_data(100);

    let (neuros, mesh_len) =
        Neuron::load_substrate_or_create("basis", 1000000, -1.0..1.0, ActivationType::Elu);

    let mut manifold = Manifold::new(mesh_len, 1, 2, vec![10, 20, 10]);
    manifold.weave();

    let mut trainer = Trainer::new(&train_x, &train_y);
    let post_processor = |sig| {
        let vecs = Signal::vectorize(sig);
        let softmax = f::softmax(&vecs);
        softmax
    };

    trainer
        .set_sample_size(8)
        .set_epochs(3000)
        .set_amplitude(1000)
        .set_post_processor(post_processor)
        .set_loss_fn(f::mean_squared_error)
        .train(&mut manifold, &neuros);

    let mut predictions: Vec<usize> = vec![];
    let actual = test_y
        .into_iter()
        .map(|x| f::argmax(&x))
        .collect::<Vec<usize>>();

    for test_xv in test_x.into_iter() {
        let mut signals = Signal::signalize(test_xv);
        manifold.forward(&mut signals, &neuros);

        let prediction = post_processor(signals);

        predictions.push(f::argmax(&prediction));
    }

    let accuracy = f::accuracy::<usize>(&predictions, &actual);

    println!("{}% Accuracy", accuracy);
}

fn main() {
    zero_two()
}
