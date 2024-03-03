use super::activation::{Activation, ActivationType};
use super::Signal;
use rand::prelude::*;
use rand::thread_rng;

#[derive(Debug)]
pub enum NeuronOperation {
    Split,
    Merge,
    Forward,
}

#[derive(Debug)]
pub struct Neuron {
    w: f64,
    b: f64,
    a: ActivationType,
}

impl Neuron {
    pub fn random_normal() -> Neuron {
        let mut rng = thread_rng();
        let mut ats = vec![|| ActivationType::Relu, || ActivationType::Elu, || {
            ActivationType::LeakyRelu
        }];
        let activation = ats.choose_mut(&mut rng).unwrap();

        Neuron {
            w: rng.gen(),
            b: rng.gen(),
            a: activation(),
        }
    }

    pub fn activation(&self, x: f64) -> f64 {
        match self.a {
            ActivationType::Relu => Activation::relu(x),
            ActivationType::LeakyRelu => Activation::leaky_relu(x),
            ActivationType::Elu => Activation::elu(x),
        }
    }

    pub fn forward(&self, signals: &mut Vec<Signal>, target: usize) {
        let signal = signals.get_mut(target).unwrap();
        signal.x *= self.w;
        signal.x += self.b;
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
}
