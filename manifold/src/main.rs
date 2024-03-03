mod arch;

use arch::Manifold;
use arch::Neuron;
use arch::Signal;

fn make_neurons(size: usize) -> Vec<Neuron> {
    let mut neurons: Vec<Neuron> = vec![];
    for _ in 0..=size {
        neurons.push(Neuron::random_normal())
    }
    neurons
}

fn main() {
    // Use a-star search algorithm or something to weave a manifold / pathway for signals to travel
    // Manifold must meet certain criterium, such as:
    // Min hidden signals - at some point in the manifold, there must be N signals
    // Output signals = output size
    // Do this by stretching and shrinking num signals through neurons

    // Stretch to max, and shrink to min in least number of steps possible
    // Swap neurons in path

    let neuros = make_neurons(100);
    let mut manifold = Manifold::new(4, 2, 30, &neuros);
    manifold.weave();

    let mut test_signals = Signal::random_normal(4);
    println!("{:?}", test_signals);

    manifold.forward(&mut test_signals);
    println!("{:?}", test_signals);
}
