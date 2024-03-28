use std::{
    collections::{HashMap, VecDeque},
    ops::Range,
    sync::Arc,
};

use rand::{seq::SliceRandom, thread_rng, Rng};

use super::{Signal, Substrate};
use crate::substrates::traits::SignalConversion;

#[derive(Clone)]
pub struct Op {
    neuron_ix: usize,
    pub prior: f64,
    pub action_potential: f64,
    pub influence: f64,
}

impl Op {
    pub fn forward(
        &mut self,
        target_signal: &mut Signal,
        signals: &mut VecDeque<Signal>,
        neuros: &Substrate,
        _discount: f64,
    ) -> Signal {
        let neuron = match neuros.get(self.neuron_ix) {
            Some(x) => x,
            None => panic!("Tried to forward neuron {}", self.neuron_ix),
        };

        let xout = signals.iter().fold(0., |a, s| a + s.x);
        target_signal.x += xout;
        self.prior = target_signal.x.clone();
        neuron.forward(target_signal, 1.);

        self.action_potential = target_signal.x;
        self.influence = (self.action_potential - self.prior).powi(2);
        target_signal.clone()
    }
}

#[derive(Clone)]
pub struct Manifold {
    mesh_len: usize,
    d_in: usize,
    d_out: usize,
    layers: Vec<usize>,
    web: Vec<HashMap<usize, Vec<Op>>>,
    pub loss: f64,
}

impl Manifold {
    pub fn new(mesh_len: usize, d_in: usize, d_out: usize, layers: Vec<usize>) -> Manifold {
        Manifold {
            mesh_len,
            d_in,
            d_out,
            layers,
            web: vec![],
            loss: 0.,
        }
    }

    pub fn dynamic(
        mesh_len: usize,
        d_in: usize,
        d_out: usize,
        breadth: Range<usize>,
        depth: Range<usize>,
    ) -> Manifold {
        let mut rng = thread_rng();
        let depth = rng.gen_range(depth);
        let layers = (0..depth)
            .map(|_| rng.gen_range(breadth.clone()))
            .collect::<Vec<usize>>();

        Manifold {
            mesh_len,
            d_in,
            d_out,
            web: vec![],
            layers,
            loss: 0.,
        }
    }

    pub fn weave(&mut self) {
        let mut rng = thread_rng();
        let mut neuron_rng = thread_rng();
        let mut pick_neuron = || neuron_rng.gen_range(0..self.mesh_len);

        let steps = vec![self.layers.clone(), vec![self.d_out]].concat();
        let mut prev_signals = self.d_in;

        for signals in steps.iter() {
            let mut ixlookup = HashMap::<usize, Vec<Op>>::new();

            if prev_signals > *signals {
                // Shrink previous into current

                let mut prev_six = (0..prev_signals).collect::<Vec<usize>>();
                prev_six.shuffle(&mut rng);

                for _ in 0..*signals {
                    let six = prev_six.pop().unwrap();
                    ixlookup.insert(
                        six,
                        vec![Op {
                            neuron_ix: pick_neuron(),
                            prior: 0.,
                            action_potential: 0.,
                            influence: 0.,
                        }],
                    );
                }
            } else {
                // Extend previous into current.

                for six in 0..prev_signals {
                    ixlookup.insert(
                        six,
                        vec![Op {
                            neuron_ix: pick_neuron(),
                            prior: 0.,
                            action_potential: 0.,
                            influence: 0.,
                        }],
                    );
                }

                let remaining_expand = signals - prev_signals;
                let new_six = (0..remaining_expand)
                    .map(|_| rng.gen_range(0..prev_signals))
                    .collect::<Vec<usize>>();

                for six in new_six.iter() {
                    let opvec = ixlookup.get_mut(six).unwrap();
                    opvec.push(Op {
                        neuron_ix: pick_neuron(),
                        prior: 0.,
                        action_potential: 0.,
                        influence: 0.,
                    });
                }
            }

            self.web.push(ixlookup);

            prev_signals = *signals;
        }
    }

    pub fn forward(&mut self, signals: &mut VecDeque<Signal>, neuros: &Substrate) {
        for layer in self.web.iter_mut() {
            let total_turns = signals.len();
            let mut turns = 0;

            while turns < total_turns {
                let mut target_signal = match signals.pop_front() {
                    Some(s) => s,
                    None => return,
                };

                // Need multiple ops per layer for growing network size.
                let ops_for_turn = match layer.get_mut(&turns) {
                    Some(x) => {
                        turns += 1;
                        x
                    }
                    None => {
                        turns += 1;
                        continue;
                    }
                };

                let mut next_signals = ops_for_turn
                    .iter_mut()
                    .map(|op| op.forward(&mut target_signal, signals, &neuros, 1.))
                    .collect::<VecDeque<Signal>>();

                signals.append(&mut next_signals);
            }
        }
    }

    pub fn turn_one(total: usize, neuron_ix: usize, amplitude: i32) -> usize {
        let mut turned = (neuron_ix as i32) + amplitude;
        let mesh_len = total as i32;

        while turned.abs() > mesh_len {
            turned -= match turned < 0 {
                true => -mesh_len,
                false => mesh_len,
            };
        }

        if turned < 0 {
            turned = mesh_len + turned;
        }

        if turned as usize > total {
            panic!("Tf you tryna pass off {} as less than {}", turned, total);
        }

        turned as usize
    }

    pub fn turn(&mut self, amplitude: i32) {
        for layer in self.web.iter_mut() {
            for (_, opvec) in layer.iter_mut() {
                for op in opvec.iter_mut() {
                    op.neuron_ix = Manifold::turn_one(self.mesh_len, op.neuron_ix, amplitude);
                }
            }
        }
    }

    pub fn find_max_op_with(
        &mut self,
        layer: usize,
        heuristic: Arc<dyn Fn(&Op) -> f64>,
    ) -> Option<(usize, usize, &Op)> {
        match self.web.get(layer) {
            Some(layer) => {
                let mut six = 0;
                let mut opix = 0;
                let mut op: Option<&Op> = None;
                let mut max = f64::MIN;

                for (_six, opvec) in layer.iter() {
                    for (_opix, _op) in opvec.iter().enumerate() {
                        // TODO if two ops' heuristics are equal this will always pick either the last or the first one.
                        // [0.5, 0.5]
                        if heuristic(_op) >= max {
                            six = *_six;
                            opix = _opix;
                            op = Some(_op);
                            max = heuristic(_op);
                        }
                    }
                }

                match op {
                    Some(op) => Some((six, opix, op)),
                    None => None,
                }
            }
            None => None,
        }
    }

    pub fn select_op(&mut self, layer: usize, six: usize, ix: usize) -> Option<&mut Op> {
        match self.web.get_mut(layer) {
            Some(layer) => match layer.get_mut(&six) {
                Some(opvec) => match opvec.get_mut(ix) {
                    Some(op) => Some(op),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        }
    }

    fn explode_inward_with(
        &self,
        layer: usize,
        heuristic: Arc<dyn Fn(&Op) -> f64>,
        amplitude: i32,
    ) -> Vec<Manifold> {
        let mut left_self = self.clone();
        let mut right_self = self.clone();

        let left_self_max_at_layer = left_self.find_max_op_with(layer, Arc::clone(&heuristic));
        let right_self_max_at_layer = right_self.find_max_op_with(layer, Arc::clone(&heuristic));

        let mut next_selves: Vec<Manifold> = vec![];

        if let Some((six, ix, ..)) = left_self_max_at_layer {
            if let Some(op) = left_self.select_op(layer, six, ix) {
                op.neuron_ix = Manifold::turn_one(self.mesh_len, op.neuron_ix, -amplitude);
                next_selves.push(left_self);
            }
        }

        if let Some((six, ix, ..)) = right_self_max_at_layer {
            if let Some(op) = right_self.select_op(layer, six, ix) {
                op.neuron_ix = Manifold::turn_one(self.mesh_len, op.neuron_ix, amplitude);
                next_selves.push(right_self);
            }
        }

        if layer <= 0 || next_selves.len() < 1 {
            return next_selves;
        }

        next_selves
            .into_iter()
            .map(|myself| myself.explode_inward_with(layer - 1, Arc::clone(&heuristic), amplitude))
            .collect::<Vec<Vec<Manifold>>>()
            .concat()
    }

    fn cannibalize(&mut self, other: &mut Manifold) {
        self.layers = other.layers.to_owned();
        self.web = other.web.to_owned();
        self.loss = 0.;
    }

    pub fn backwards(
        &mut self,
        loss: f64,
        mut amplitude: i32,
        x: &Vec<f64>,
        y: &Vec<f64>,
        loss_fn: &LossFn,
        post_processor: &PostProcessor,
        neuros: &Substrate,
    ) {
        // Manifold splits at every point of greatest influence along the pathway from end to start recursively.
        // Incrementing the Op of greatest influence both +1/-1 neuron on the graph.
        // These pathways are then tested, and the best one lives. A mixture of backprop and genetic.
        amplitude = (amplitude as f64 * loss.abs().powi(2)).floor() as i32;

        // Max amplitude
        if amplitude as f32 > self.mesh_len as f32 / 3. {
            amplitude = (self.mesh_len as f32 / 3.).floor() as i32;
        }

        let heuristic_action_potential = Arc::new(|op: &Op| op.action_potential);

        let mut all_selves =
            self.explode_inward_with(self.layers.len(), heuristic_action_potential, amplitude);

        all_selves.iter_mut().for_each(|m| {
            let mut signals = Signal::signalize(x.clone());
            m.forward(&mut signals, neuros);
            let signal_vector = post_processor(signals);

            let loss = loss_fn(&signal_vector, y);
            m.accumulate_loss(loss);
        });

        let mut lossy_selves = VecDeque::from(all_selves);
        lossy_selves.make_contiguous().sort_unstable_by(|m, n| {
            m.loss
                .partial_cmp(&n.loss)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut optimized = match lossy_selves.pop_front() {
            Some(x) => x,
            None => return,
        };

        self.cannibalize(&mut optimized);

        drop(lossy_selves);
    }

    pub fn accumulate_loss(&mut self, a: f64) {
        self.loss += a;
    }

    pub fn reset_loss(&mut self) {
        self.loss = 0.;
    }
}

pub type LossFn = Box<dyn Fn(&[f64], &[f64]) -> f64>;
pub type PostProcessor = Box<dyn Fn(VecDeque<Signal>) -> Vec<f64>>;

pub struct Trainer<'a> {
    x: &'a Vec<Vec<f64>>,
    y: &'a Vec<Vec<f64>>,
    sample_size: usize,
    epochs: usize,
    amplitude: i32,
    post_processor: Option<PostProcessor>,
    loss_fn: Option<LossFn>,
}

impl Trainer<'_> {
    pub fn new<'a>(x: &'a Vec<Vec<f64>>, y: &'a Vec<Vec<f64>>) -> Trainer<'a> {
        Trainer {
            x,
            y,
            sample_size: 1,
            epochs: 1,
            amplitude: 1,
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

    pub fn set_amplitude(&mut self, x: i32) -> &mut Self {
        self.amplitude = x;
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

                manifold.backwards(
                    loss,
                    self.amplitude,
                    &x,
                    &y,
                    &loss_fn,
                    &post_processor,
                    &neuros,
                );

                losses.push(loss);
            }

            let avg_loss = losses.iter().fold(0., |a, v| a + v) / losses.len() as f64;
            println!("({} / {}) Avg loss: {}", epoch, self.epochs, avg_loss);
        }

        manifold
    }
}
