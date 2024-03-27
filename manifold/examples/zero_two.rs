use manifold::optimizers::{Basis, EvolutionHyper, Turnstile};
use manifold::substrates::fully_connected::{Neuron, Signal};
use manifold::substrates::traits::SignalConversion;
use manifold::{f, Optimizer};
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

    let neuros = Neuron::substrate(1000000, -10.0..10.0);

    let hyper = EvolutionHyper {
        population_size: 200,
        elitism_carryover: 20,
        carryover_rate: 0.3,
        sample_size: 60,
    };

    let basis = Basis {
        x: train_x,
        y: train_y,
        neuros,
    };

    let optim = Turnstile::new(1, 2, 2..3, 4..5, -2..2, 100);

    let (mut population, basis, ..) = optim.train(basis, hyper);

    let manifold_am = population.pop_front().unwrap();
    let manifold = manifold_am.lock().unwrap();

    let mut predictions: Vec<usize> = vec![];
    let actual = test_y
        .into_iter()
        .map(|x| f::argmax(&x))
        .collect::<Vec<usize>>();

    for test_xv in test_x.into_iter() {
        let mut signals = Signal::signalize(test_xv);
        manifold.forward(&mut signals, &basis.neuros);

        let prediction = Signal::vectorize(signals);

        predictions.push(f::argmax(&prediction));
    }

    let accuracy = f::accuracy(&predictions, &actual);

    println!("{}% Accuracy", accuracy);
}

fn main() {
    zero_two()
}
