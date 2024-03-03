mod arch;
mod util;

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

    static mut NEURONS: Vec<Neuron> = vec![];

    unsafe {
        let mut neurons = make_neurons(100);
        NEURONS.append(&mut neurons);
        let manifold = Manifold::new(4, 2, 30, &NEURONS);

        manifold.weave();
    }
}
