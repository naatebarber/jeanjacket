pub mod manifold;
pub mod neuron;
pub mod signal;
pub mod trainer;

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

pub use manifold::Manifold;
pub use neuron::Neuron;
pub use signal::Signal;
pub use trainer::Trainer;

pub type Substrate = Arc<VecDeque<Neuron>>;
pub type Population = VecDeque<Arc<Mutex<Manifold>>>;
