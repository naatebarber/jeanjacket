use std::collections::VecDeque;
use std::error::Error;
use std::ops::Range;
use std::sync::Arc;

use super::{Signal, Substrate};
use crate::activation::{Activation, ActivationType};

use rand::prelude::*;
use rand::thread_rng;
use serde::{Deserialize, Serialize};

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

    pub fn substrate(size: usize, range: Range<f64>) -> Substrate {
        let mut neurons: VecDeque<Neuron> = VecDeque::new();
        for _ in 0..=size {
            neurons.push_back(Neuron::random_normal(&range))
        }

        neurons.make_contiguous().sort_unstable_by(|a, b| {
            let v = a.w * 2. + a.b;
            let w = b.w * 2. + b.b;
            match v > w {
                true => std::cmp::Ordering::Greater,
                false => std::cmp::Ordering::Less,
            }
        });

        Arc::new(neurons)
    }

    pub fn activation(&self, x: f64) -> f64 {
        match self.a {
            ActivationType::Relu => Activation::relu(x),
            ActivationType::LeakyRelu => Activation::leaky_relu(x),
            ActivationType::Elu => Activation::elu(x),
        }
    }

    pub fn forward(&self, signal: &mut Signal, discount: f64) {
        let mut after = signal.x.clone();
        after *= self.w;
        after += self.b;
        let mut diff = after - signal.x;
        diff *= discount;

        signal.x += diff;
        signal.x = self.activation(signal.x);
    }

    pub fn dump_substrate(neuros: Substrate) -> Result<String, Box<dyn Error>> {
        Ok(serde_json::to_string(&neuros)?)
    }

    pub fn load_substrate(serial: String) -> Result<Substrate, Box<dyn Error>> {
        Ok(Arc::new(serde_json::from_str(&serial)?))
    }
}
