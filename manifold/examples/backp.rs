use manifold::f;
use manifold::substrates::fully_connected::{Manifold, Neuron, Signal, Trainer};
use manifold::substrates::traits::SignalConversion;
use rand::{thread_rng, Rng};

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

    let (train_x, train_y) = gen_binary_training_data(5000);
    let (test_x, test_y) = gen_binary_training_data(1000);

    let neuros = Neuron::substrate(10000, -10.0..10.0);
    let mut manifold = Manifold::new(10000, 1, 2, vec![10]);
    let mut trainer = Trainer::new(&train_x, &train_y);

    trainer
        .set_sample_size(20)
        .set_epochs(100)
        .set_loss_fn(f::binary_cross_entropy)
        .set_post_processor(|sig| {
            let vecs = Signal::vectorize(sig);
            let softmax = f::softmax(&vecs);
            softmax
        })
        .train(&mut manifold, &neuros);

    let mut predictions: Vec<usize> = vec![];
    let actual = test_y
        .into_iter()
        .map(|x| f::argmax(&x))
        .collect::<Vec<usize>>();

    for test_xv in test_x.into_iter() {
        let mut signals = Signal::signalize(test_xv);
        manifold.forward(&mut signals, &neuros);

        let prediction = Signal::vectorize(signals);

        predictions.push(f::argmax(&prediction));
    }

    let accuracy = f::accuracy(&predictions, &actual);

    println!("{}% Accuracy", accuracy);
}

fn main() {
    zero_two()
}
