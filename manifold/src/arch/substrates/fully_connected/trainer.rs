use std::collections::VecDeque;

use rand::{seq::SliceRandom, thread_rng};

use crate::substrates::traits::SignalConversion;

use super::{Manifold, Signal, Substrate};

pub struct Trainer<'a> {
    x: &'a Vec<Vec<f64>>,
    y: &'a Vec<Vec<f64>>,
    sample_size: usize,
    epochs: usize,
    post_processor: Option<Box<dyn Fn(VecDeque<Signal>) -> Vec<f64>>>,
    loss_fn: Option<Box<dyn Fn(&[f64], &[f64]) -> f64>>,
}

impl Trainer<'_> {
    pub fn new<'a>(x: &'a Vec<Vec<f64>>, y: &'a Vec<Vec<f64>>) -> Trainer<'a> {
        Trainer {
            x,
            y,
            sample_size: 1,
            epochs: 1,
            post_processor: None,
            loss_fn: None,
        }
    }

    pub fn set_sample_size(&mut self, x: usize) -> &mut Self {
        self.sample_size = x;
        self
    }

    pub fn set_epochs(&mut self, x: usize) -> &mut Self {
        self.epochs = x;
        self
    }

    pub fn set_post_processor(
        &mut self,
        processor: impl Fn(VecDeque<Signal>) -> Vec<f64> + 'static,
    ) -> &mut Self {
        self.post_processor =
            Some(Box::new(processor) as Box<dyn Fn(VecDeque<Signal>) -> Vec<f64>>);
        self
    }

    pub fn set_loss_fn(
        &mut self,
        processor: impl Fn(&[f64], &[f64]) -> f64 + 'static,
    ) -> &mut Self {
        self.loss_fn = Some(Box::new(processor) as Box<dyn Fn(&[f64], &[f64]) -> f64>);
        self
    }

    pub fn sample(&self) -> Vec<(&Vec<f64>, &Vec<f64>)> {
        let mut rng = thread_rng();

        let mut ixlist = (0..self.x.len()).collect::<Vec<usize>>();
        ixlist.shuffle(&mut rng);
        let sample_ixlist = (0..self.sample_size)
            .filter_map(|_| ixlist.pop())
            .collect::<Vec<usize>>();
        let to_xy = sample_ixlist
            .iter()
            .map(|ix| (&self.x[ix.clone()], &self.y[*ix]))
            .collect();

        to_xy
    }

    pub fn train<'a>(
        &'a mut self,
        manifold: &'a mut Manifold,
        neuros: &'a Substrate,
    ) -> &'a mut Manifold {
        let loss_fn = match &self.loss_fn {
            Some(x) => x,
            None => {
                println!("Trainer 'train' called without first setting a loss metric.");
                return manifold;
            }
        };

        let default_post_processor =
            Box::new(Signal::vectorize) as Box<dyn Fn(VecDeque<Signal>) -> Vec<f64>>;
        let post_processor = match &self.post_processor {
            Some(x) => x,
            None => &default_post_processor,
        };

        for epoch in 0..self.epochs {
            let samples = self.sample();
            let mut losses: Vec<f64> = vec![];

            for data in samples.into_iter() {
                let (x, y) = data;
                let mut signals = Signal::signalize(x.clone());
                manifold.forward(&mut signals, neuros);

                let signal_vector = post_processor(signals);
                let target_vector = y.clone();

                let loss = loss_fn(&signal_vector, &target_vector);

                manifold.backward(loss);

                losses.push(loss);
            }

            let avg_loss = losses.iter().fold(0., |a, v| a + v) / losses.len() as f64;
            println!("({} / {}) Avg loss: {}", epoch, self.epochs, avg_loss);
        }

        manifold
    }
}
