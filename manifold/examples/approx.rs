use manifold::{
    activation::ActivationType,
    f,
    substrates::{
        fully_connected::{Dataset, Manifold, Neuron, Signal, Trainer},
        traits::SignalConversion,
    },
};
use rand::{thread_rng, Rng};

fn gen_xy(size: usize) -> Dataset {
    let mut rng = thread_rng();

    let mutator = |x: f64| x;

    let (mut x, mut y): Dataset = (vec![], vec![]);

    for _ in 0..size {
        let xv: f64 = rng.gen_range(0.0..5.0);
        x.push(vec![xv.clone()]);
        y.push(vec![mutator(xv)]);
    }

    (x, y)
}

fn main() {
    let dataset = gen_xy(10000);

    let (neuros, mesh_len) =
        Neuron::load_substrate_or_create("sec", 100000, -1.0..1.5, ActivationType::Relu);

    let mut manifold = Manifold::new(mesh_len, 1, 1, vec![5, 100, 12, 13]);
    manifold.weave();

    let mut trainer = Trainer::new(&dataset.0, &dataset.1);

    trainer
        .set_epochs(10000)
        .set_sample_size(30)
        .set_loss_fn(f::mean_squared_error)
        .set_post_processor(Signal::vectorize)
        .set_amplitude(100)
        .train(&mut manifold, &neuros);
}
