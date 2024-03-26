use std::error::Error;
use std::ops::Range;
use std::sync::Arc;

use super::{Signal, Substrate};
use crate::activation::{Activation, ActivationType};
use rand::prelude::*;
use rand::thread_rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NeuronOperation {
    Forward,
    Split,
    Merge,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Neuron {
    w: f64,
    b: f64,
    a: ActivationType,
}

impl Neuron {
    pub fn random_normal(range: &Range<f64>) -> Neuron {
        let mut rng = thread_rng();
        let mut ats = vec![|| ActivationType::Relu, || ActivationType::Elu, || {
            ActivationType::LeakyRelu
        }];
        let activation = ats.choose_mut(&mut rng).unwrap();

        Neuron {
            w: rng.gen_range(range.clone()),
            b: rng.gen_range(range.clone()),
            a: activation(),
        }
    }

    pub fn substrate(size: usize, range: Range<f64>) -> Vec<Neuron> {
        let mut neurons: Vec<Neuron> = vec![];
        for _ in 0..=size {
            neurons.push(Neuron::random_normal(&range))
        }
        neurons
    }

    pub fn activation(&self, x: f64) -> f64 {
        match self.a {
            ActivationType::Relu => Activation::relu(x),
            ActivationType::LeakyRelu => Activation::leaky_relu(x),
            ActivationType::Elu => Activation::elu(x),
        }
    }

    pub fn forward(&self, signals: &mut Vec<Signal>, target: usize, discount: f64) {
        let signal = match signals.get_mut(target) {
            Some(x) => x,
            None => return,
        };

        let mut after = signal.x.clone();
        after *= self.w;
        after += self.b;
        let mut diff = after - signal.x;
        diff *= discount;

        signal.x += diff;
        signal.x = self.activation(signal.x);
    }

    pub fn merge(&self, signals: &mut Vec<Signal>, target: usize) {
        let neighbor = signals.len() % (target + 1);
        let signal_a = signals.get(target).unwrap();
        let signal_b = signals.get(neighbor).unwrap();

        let merge = Signal {
            x: signal_a.x + signal_b.x,
        };

        if target < neighbor {
            signals.splice(target..neighbor + 1, vec![merge]);
        } else {
            signals.splice(target..target + 1, vec![]);
            signals.splice(neighbor..neighbor + 1, vec![]);
            signals.push(merge);
        }
    }

    pub fn split(&self, signals: &mut Vec<Signal>, target: usize) {
        let signal = signals.get(target).unwrap();

        let md = signal.x / 2.;
        let splits = vec![Signal { x: md }, Signal { x: md }];

        signals.splice(target..target + 1, splits);
    }

    pub fn dump_substrate(neuros: Substrate) -> Result<String, Box<dyn Error>> {
        Ok(serde_json::to_string(&neuros)?)
    }

    pub fn load_substrate(serial: String) -> Result<Substrate, Box<dyn Error>> {
        Ok(Arc::new(serde_json::from_str(&serial)?))
    }
}
