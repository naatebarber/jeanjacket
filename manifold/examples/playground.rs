use manifold::substrates::fully_connected::{Manifold, Neuron, Signal};

fn main() {
    let d_in = 784;
    let d_out = 10;
    let layers = vec![64, 64];

    let mut sigs = Signal::_random_normal(d_in);
    println!("{:?}", sigs);

    let substrate = Neuron::substrate(100000, -1.0..1.0);

    let mut manifold = Manifold::new(substrate.len(), d_in, d_out, layers);
    manifold.weave();

    manifold.forward(&mut sigs, &substrate);
    println!("{:?}", sigs);
}
