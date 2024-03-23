mod activation;
pub mod constant_fold;
pub mod manifold;
pub mod neuron;
pub mod signal;

pub use manifold::Manifold;
pub use neuron::{Neuron, NeuronOperation};
pub use signal::Signal;
