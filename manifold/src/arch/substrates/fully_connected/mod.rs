pub mod manifold;
pub mod neuron;
pub mod signal;

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

pub use manifold::*;
pub use neuron::Neuron;
pub use signal::Signal;

pub type Substrate = Arc<VecDeque<Neuron>>;
pub type Population = VecDeque<Arc<Mutex<Manifold>>>;
pub type Dataset = (Vec<Vec<f64>>, Vec<Vec<f64>>);
pub type DatasetReference<'a> = (Vec<&'a Vec<f64>>, Vec<&'a Vec<f64>>);
