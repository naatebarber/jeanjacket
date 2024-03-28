use std::{
    collections::{HashMap, VecDeque},
    ops::Range,
    sync::Arc,
};

use rand::{seq::SliceRandom, thread_rng, Rng};

use super::{Signal, Substrate};
use crate::substrates::traits::SignalConversion;

#[derive(Clone, Debug)]
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
        discount: f64,
    ) -> Signal {
        let neuron = match neuros.get(self.neuron_ix) {
            Some(x) => x,
            None => panic!("Tried to forward neuron {}", self.neuron_ix),
        };

        let xout = signals.iter().fold(0., |a, s| a + s.x);
        target_signal.x += xout;
        self.prior = target_signal.x.clone();

        target_signal.x = neuron.forward(Signal::vectorize(signals.clone()), discount);

        self.action_potential = target_signal.x;
        self.influence = (self.action_potential - self.prior).powi(2);
        target_signal.clone()
    }

    pub fn swap_focus(&mut self, neuron_ix: usize) {
        self.neuron_ix = neuron_ix;
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

    pub fn discount_factor(&self) -> f64 {
        let layers = self.web.len();
        -(0.5 / (layers + 1) as f64) + 1.
    }

    pub fn discount(web_layer: usize, factor: f64) -> f64 {
        factor.powi(web_layer as i32)
    }

    pub fn forward(&mut self, signals: &mut VecDeque<Signal>, neuros: &Substrate) {
        for (_, layer) in self.web.iter_mut().enumerate() {
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

    pub fn turn_one(total: usize, neuron_ix: usize, amplitude: i64) -> usize {
        let mut turned = (neuron_ix as i64) + amplitude;
        let mesh_len = total as i64;

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

    pub fn turn(&mut self, amplitude: i64) {
        for layer in self.web.iter_mut() {
            for (_, opvec) in layer.iter_mut() {
                for op in opvec.iter_mut() {
                    op.neuron_ix = Manifold::turn_one(self.mesh_len, op.neuron_ix, amplitude);
                }
            }
        }
    }

    pub fn find_max_op_with(
        &self,
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

    fn turn_greatest_influence_at_layer(
        &mut self,
        layer: usize,
        heuristic: Arc<dyn Fn(&Op) -> f64>,
        amplitude: i64,
    ) -> bool {
        let max_at_layer = self.find_max_op_with(layer, Arc::clone(&heuristic));

        let mesh_len = self.mesh_len;

        if let Some((six, ix, ..)) = max_at_layer {
            if let Some(op) = self.select_op(layer, six, ix) {
                let next_neuron = Manifold::turn_one(mesh_len, op.neuron_ix, amplitude);
                op.swap_focus(next_neuron);
                return true;
            }
        }

        false
    }

    fn explode_inward_with(
        &self,
        layer: usize,
        heuristic: Arc<dyn Fn(&Op) -> f64>,
        amplitude: i64,
    ) -> Vec<Manifold> {
        let mut left_self = self.clone();
        let mut right_self = self.clone();

        let left_turned =
            left_self.turn_greatest_influence_at_layer(layer, Arc::clone(&heuristic), -amplitude);
        let right_turned =
            right_self.turn_greatest_influence_at_layer(layer, Arc::clone(&heuristic), amplitude);

        let mut next_selves: Vec<Manifold> = vec![];

        if left_turned {
            next_selves.push(left_self);
        }

        if right_turned {
            next_selves.push(right_self);
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

    fn cannibalize_layer(&mut self, other: &mut Manifold, layer: usize) {
        let mine: Vec<HashMap<usize, Vec<Op>>> =
            other.web.splice(layer..layer + 1, vec![]).collect();
        self.web.splice(layer..layer + 1, mine);
    }

    pub fn backward_explode(
        &mut self,
        mut amplitude: i64,
        x: &Vec<&Vec<f64>>,
        y: &Vec<&Vec<f64>>,
        loss_fn: &LossFn,
        post_processor: &PostProcessor,
        neuros: &Substrate,
    ) -> (bool, f64) {
        // Manifold splits at every point of greatest influence along the pathway from end to start recursively. O(2^n)
        // Incrementing the Op of greatest influence both +1/-1 neuron on the graph.
        // These pathways are then tested, and the best one lives. A mixture of backprop and genetic.
        self.reset_loss();

        for (x, y) in x.iter().zip(y.iter()) {
            let mut signals = Signal::signalize(x.to_vec());
            self.forward(&mut signals, neuros);
            let signal_vector = post_processor(signals);

            let loss = loss_fn(&signal_vector, y);
            self.accumulate_loss(loss);
        }

        let loss = self.loss.clone();

        amplitude = (amplitude as f64 * loss.powi(2)) as i64;

        if amplitude < 1 {
            amplitude = 1
        }

        if amplitude as f32 > self.mesh_len as f32 / 2. {
            amplitude = (self.mesh_len as f32 / 2.).floor() as i64;
        }

        let heuristic_action_potential = Arc::new(|op: &Op| op.action_potential * op.influence);

        let all_selves =
            self.explode_inward_with(self.layers.len(), heuristic_action_potential, amplitude);

        for mut alternative_self in all_selves.into_iter() {
            alternative_self.reset_loss();
            for (x, y) in x.iter().zip(y.iter()) {
                let mut signals = Signal::signalize(x.to_vec());
                alternative_self.forward(&mut signals, neuros);
                let signal_vector = post_processor(signals);
                let loss = loss_fn(&signal_vector, y);

                alternative_self.accumulate_loss(loss);
            }

            if alternative_self.loss < loss {
                self.cannibalize(&mut alternative_self);
            }
        }

        (true, self.loss)
    }

    pub fn backward_methodic(
        &mut self,
        mut amplitude: i64,
        x: &Vec<&Vec<f64>>,
        y: &Vec<&Vec<f64>>,
        loss_fn: &LossFn,
        post_processor: &PostProcessor,
        neuros: &Substrate,
    ) -> (bool, f64) {
        // Manifold clones once for each point of greatest influence. There is no split. O(n)
        // Incrementing the Op of greatest influence both +1/0/-1 neuron on the graph.
        // These pathways are then tested, and the best one lives. A mixture of backprop and genetic.

        // Gather my own loss first before proceeding.
        self.reset_loss();

        for (x, y) in x.iter().zip(y.iter()) {
            let mut signals = Signal::signalize(x.to_vec());
            self.forward(&mut signals, neuros);
            let signal_vector = post_processor(signals);

            let loss = loss_fn(&signal_vector, y);
            self.accumulate_loss(loss);
        }

        let loss = self.loss.clone();

        amplitude = (amplitude as f64 * loss.abs().powf(2.)) as i64;

        if amplitude < 1 {
            amplitude = 1
        }

        if amplitude as f32 > self.mesh_len as f32 / 2. {
            amplitude = (self.mesh_len as f32 / 2.).floor() as i64;
        }

        let heuristic_action_potential =
            Arc::new(|op: &Op| op.action_potential.powf(op.influence)) as Arc<dyn Fn(&Op) -> f64>;
        let _heuristic_neuron_ix =
            Arc::new(|op: &Op| op.neuron_ix as f64) as Arc<dyn Fn(&Op) -> f64>;

        let heuristic = heuristic_action_potential;

        let hyperview = (0..self.layers.len())
            .into_iter()
            .map(|layer| {
                let mut left_self = self.clone();
                let mut right_self = self.clone();

                let left_turned = left_self.turn_greatest_influence_at_layer(
                    layer,
                    Arc::clone(&heuristic),
                    -amplitude,
                );

                let right_turned = right_self.turn_greatest_influence_at_layer(
                    layer,
                    Arc::clone(&heuristic),
                    amplitude,
                );

                let mut next_selves: Vec<Manifold> = vec![];

                if left_turned {
                    next_selves.push(left_self);
                }

                if right_turned {
                    next_selves.push(right_self);
                }

                next_selves
            })
            .collect::<Vec<Vec<Manifold>>>();

        let mut adapted_count = 0;
        let mut alternative_count = 0;

        for (layer_ix, layer) in hyperview.into_iter().enumerate() {
            for mut alternative_self in layer.into_iter() {
                alternative_self.reset_loss();
                for (x, y) in x.iter().zip(y.iter()) {
                    let mut signals = Signal::signalize(x.to_vec());
                    alternative_self.forward(&mut signals, neuros);
                    let signal_vector = post_processor(signals);
                    let loss = loss_fn(&signal_vector, y);

                    alternative_self.accumulate_loss(loss);
                }

                if alternative_self.loss < loss {
                    self.cannibalize_layer(&mut alternative_self, layer_ix);
                    adapted_count += 1;
                    break;
                }

                alternative_count += 1;

                print!("{}, ", alternative_self.loss);
            }
        }

        println!("Alternatives: {}", alternative_count);
        println!("Adaptations: {}", adapted_count);

        if adapted_count < 1 {
            // Check for early stop if loss is tolerable
            // self.weave();
            // Otherwise:
            // - Mutate structure
            // - Randomly change some neurons
            // - Increase amplitude (not here)
        }

        (true, loss)
    }

    pub fn accumulate_loss(&mut self, a: f64) {
        self.loss += a;
    }

    pub fn reset_loss(&mut self) {
        self.loss = 0.;
    }

    pub fn current_neurons(&self) -> String {
        let mut nmatch = String::default();

        for h in self.web.iter() {
            for op in h.values() {
                for o in op.iter() {
                    nmatch.push_str(format!("{},", o.neuron_ix).as_str());
                }
            }
            nmatch.push_str("\n");
        }

        nmatch
    }

    pub fn neuron_ix_average(&self) -> f64 {
        let mut ixlist: Vec<usize> = vec![];

        for h in self.web.iter() {
            for op in h.values() {
                for o in op.iter() {
                    ixlist.push(o.neuron_ix.clone());
                }
            }
        }

        ixlist.iter().fold(0., |a, v| a + *v as f64) / ixlist.len() as f64
    }
}

pub type LossFn = Box<dyn Fn(&[f64], &[f64]) -> f64>;
pub type PostProcessor = Box<dyn Fn(VecDeque<Signal>) -> Vec<f64>>;

pub struct Trainer<'a> {
    x: &'a Vec<Vec<f64>>,
    y: &'a Vec<Vec<f64>>,
    sample_size: usize,
    epochs: usize,
    amplitude: i64,
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

    pub fn set_amplitude(&mut self, x: i64) -> &mut Self {
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
            let (x, y) = self.sample();
            let mut losses: Vec<f64> = vec![];

            let (proceed, loss) = manifold.backward_explode(
                self.amplitude,
                &x,
                &y,
                &loss_fn,
                &post_processor,
                &neuros,
            );

            losses.push(loss);

            let avg_loss = losses.iter().fold(0., |a, v| a + v) / losses.len() as f64;
            println!("({} / {}) Avg loss: {}", epoch, self.epochs, avg_loss);

            if !proceed {
                break;
            }
        }

        manifold
    }
}
