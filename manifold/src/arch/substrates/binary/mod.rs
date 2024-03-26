pub mod manifold;
pub mod neuron;
pub mod signal;

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

pub use manifold::Manifold;
pub use neuron::{Neuron, NeuronOperation};
pub use signal::Signal;

pub type Population = VecDeque<Arc<Mutex<Manifold>>>;
pub type Substrate = Arc<Vec<Neuron>>;
