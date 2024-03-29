use std::collections::VecDeque;

use super::{Manifold, Signal, Substrate};
use crate::f;
use crate::substrates::traits::SignalConversion;
use plotly::{Bar, Plot};
use rand::{prelude::*, thread_rng};

pub type LossFn = Box<dyn Fn(&VecDeque<Signal>, &[f64]) -> Vec<f64>>;
pub type PostProcessor = Box<dyn Fn(&[f64]) -> Vec<f64>>;

pub struct Trainer<'a> {
    x: &'a Vec<Vec<f64>>,
    y: &'a Vec<Vec<f64>>,
    losses: Vec<f64>,
    sample_size: usize,
    epochs: usize,
    rate: f64,
    post_processor: PostProcessor,
    loss_fn: LossFn,
}

impl Trainer<'_> {
    pub fn new<'a>(x: &'a Vec<Vec<f64>>, y: &'a Vec<Vec<f64>>) -> Trainer<'a> {
        Trainer {
            x,
            y,
            losses: vec![],
            sample_size: 1,
            epochs: 1,
            rate: 0.1,
            post_processor: Box::new(|x| x.to_vec()),
            loss_fn: Box::new(|signal, expected| {
                signal
                    .iter()
                    .zip(expected.iter())
                    .map(|(signal, expected)| {
                        f::mean_squared_error(&vec![signal.x], &vec![*expected])
                    })
                    .collect::<Vec<f64>>()
            }),
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

    pub fn set_rate(&mut self, x: f64) -> &mut Self {
        self.rate = x;
        self
    }

    pub fn set_post_processor(
        &mut self,
        processor: impl Fn(&[f64]) -> Vec<f64> + 'static,
    ) -> &mut Self {
        self.post_processor = Box::new(processor) as PostProcessor;
        self
    }

    pub fn set_loss_fn(
        &mut self,
        loss_fn: impl Fn(&VecDeque<Signal>, &[f64]) -> Vec<f64> + 'static,
    ) -> &mut Self {
        self.loss_fn = Box::new(loss_fn) as Box<dyn Fn(&VecDeque<Signal>, &[f64]) -> Vec<f64>>;
        self
    }

    pub fn sample(&self) -> (Vec<&Vec<f64>>, Vec<&Vec<f64>>) {
        let mut rng = thread_rng();

        let mut ixlist = (0..self.x.len()).collect::<Vec<usize>>();
        ixlist.shuffle(&mut rng);
        let sample_ixlist = (0..self.sample_size)
            .filter_map(|_| ixlist.pop())
            .collect::<Vec<usize>>();

        let x = sample_ixlist
            .iter()
            .map(|ix| &self.x[ix.clone()])
            .collect::<Vec<&Vec<f64>>>();
        let y = sample_ixlist
            .iter()
            .map(|ix| &self.y[*ix])
            .collect::<Vec<&Vec<f64>>>();

        (x, y)
    }

    pub fn train<'a>(&'a mut self, manifold: &'a mut Manifold, neuros: &'a Substrate) -> &mut Self {
        for epoch in 0..self.epochs {
            let (x, y) = self.sample();
            let mut losses: Vec<f64> = vec![];

            let zip_x_y = x.iter().zip(y.iter());

            for (&x, &y) in zip_x_y {
                let mut signals = Signal::signalize(x.clone());
                manifold.forward(&mut signals, neuros);
                println!(
                    "Unfiltered Prediction: {:?}",
                    signals.iter().map(|s| s.x).collect::<Vec<f64>>()
                );
                Signal::transform_output_slice(&mut signals, &self.post_processor);
                let system_free_energy =
                    manifold.apply_blame(&mut signals, y, &self.loss_fn, self.rate);
                losses.push(system_free_energy);
                println!();
            }

            let avg_loss = losses.iter().fold(0., |a, v| a + v) / losses.len() as f64;
            self.losses.push(avg_loss);
            println!("({} / {}) Avg loss: {}", epoch, self.epochs, avg_loss);
        }

        self
    }

    pub fn loss_graph(&mut self) -> &mut Self {
        let mut plot = Plot::new();

        let x = (0..self.losses.len()).collect();

        let trace = Bar::new(x, self.losses.clone());
        plot.add_trace(trace);
        plot.write_html("loss.html");
        plot.show();

        self
    }
}
