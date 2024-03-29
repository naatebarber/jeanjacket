use manifold::{
    activation::ActivationType,
    f,
    substrates::blame_graph::{Dataset, Manifold, Neuron, Trainer},
};
use rand::{thread_rng, Rng};

fn gen_xy(size: usize) -> Dataset {
    let mut rng = thread_rng();

    let mutator = |x: f64| x.powi(2);

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
        Neuron::load_substrate_or_create("approx", 100000, -1.0..1.0, ActivationType::Relu);

    let mut manifold = Manifold::new(mesh_len, 1, 1, vec![5, 8, 3]);
    manifold.weave();

    let mut trainer = Trainer::new(&dataset.0, &dataset.1);

    trainer
        .set_epochs(1000)
        .set_sample_size(30)
        .set_loss_fn(|signal, expected| {
            let svs = signal.iter().map(|v| v.x).collect::<Vec<f64>>();
            f::componentized_mean_squared_error(&svs, expected)
        })
        .set_rate(0.2)
        .set_decay(0.99)
        .train(&mut manifold, &neuros)
        .loss_graph();
}
