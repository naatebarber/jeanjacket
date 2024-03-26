use super::{Neuron, NeuronOperation, Signal};
use rand;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::error::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Op {
    operation: NeuronOperation,
    neuron_ix: usize,
    signal_ix: usize,
}

impl Op {
    pub fn execute(&self, signals: &mut Vec<Signal>, mesh: &Vec<Neuron>, discount: f64) {
        let neuro = mesh.get(self.neuron_ix).unwrap();

        match self.operation {
            NeuronOperation::Forward => neuro.forward(signals, self.signal_ix, discount),
            NeuronOperation::Merge => {
                neuro.merge(signals, self.signal_ix);
                neuro.forward(signals, self.signal_ix, discount);
            }
            NeuronOperation::Split => {
                neuro.split(signals, self.signal_ix);
                neuro.forward(signals, self.signal_ix, discount);
            }
        }
    }

    pub fn swap_focus(&mut self, neuron_ix: usize) {
        self.neuron_ix = neuron_ix;
    }

    pub fn swap_op(&mut self, neuron_op: NeuronOperation) {
        self.operation = neuron_op;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifold {
    input: usize,
    reaches: Vec<usize>,
    output: usize,
    mesh_len: usize,
    pub web: Vec<Vec<Op>>,
    pub loss: f64,
}

impl Manifold {
    pub fn new<'a>(input: usize, output: usize, reaches: Vec<usize>, mesh_len: usize) -> Manifold {
        Manifold {
            input,
            reaches,
            output,
            mesh_len,
            web: vec![],
            loss: 0.,
        }
    }

    pub fn weave_between<'a>(&'a mut self, s: usize, e: usize) -> Vec<Vec<Op>> {
        let descend = s > e;
        let ascend = s < e;
        let passthrough = s == e;

        let mut c = s;

        let mut rng = rand::thread_rng();

        let mut weave: Vec<Vec<Op>> = Vec::new();

        // While the number of signals is not the desired size
        loop {
            let mut ops: VecDeque<Op> = VecDeque::new();

            let cross = rng.gen_range(0..c);

            for signal_ix in 0..c {
                let neuron_ix = rng.gen_range(0..self.mesh_len);
                let op: Op = match signal_ix == cross {
                    true => match (ascend, descend) {
                        (true, false) => {
                            c += 1;
                            Op {
                                operation: NeuronOperation::Split,
                                neuron_ix,
                                signal_ix,
                            }
                        }
                        (false, true) => {
                            c -= 1;
                            Op {
                                operation: NeuronOperation::Merge,
                                neuron_ix,
                                signal_ix,
                            }
                        }
                        _ => Op {
                            operation: NeuronOperation::Forward,
                            neuron_ix,
                            signal_ix,
                        },
                    },
                    false => Op {
                        operation: NeuronOperation::Forward,
                        neuron_ix,
                        signal_ix,
                    },
                };

                ops.push_back(op);
            }

            weave.push(Vec::from(ops));

            if passthrough {
                break;
            }

            if descend && c <= e {
                break;
            }

            if ascend && c >= e {
                break;
            }
        }

        return weave;
    }

    pub fn weave(&mut self) {
        let steps = vec![vec![self.input], self.reaches.clone(), vec![self.output]].concat();

        for i in 0..(steps.len() - 1) {
            let mut weave = self.weave_between(steps[i], steps[i + 1]);
            self.web.append(&mut weave);
        }
    }

    /// Reweaves the Manifold starting from a point backtrack
    /// Returns a new Manifold instance
    pub fn reweave_backtrack(&mut self, backtrack: usize) -> Manifold {
        let op_ix = backtrack;
        let mut new_web = self.web[0..op_ix].to_vec();
        let backtrack_op = self.web.get(op_ix).unwrap();
        let weave_start_size = backtrack_op.len();

        let mut remaining_reach = VecDeque::from(self.reaches.clone());
        for layer in 0..backtrack {
            if remaining_reach.len() == 0 {
                break;
            }
            let current_size = new_web[layer].len();
            if current_size == remaining_reach[0] {
                remaining_reach.pop_front().unwrap();
            }
        }

        let steps = vec![
            vec![weave_start_size],
            Vec::from(remaining_reach),
            vec![self.output],
        ]
        .concat();

        for i in 0..(steps.len() - 1) {
            let mut weave = self.weave_between(steps[i], steps[i + 1]);
            new_web.append(&mut weave);
        }

        // println!("OS {}", split_web[split_web.len() - 1].len());

        Manifold {
            input: self.input.clone(),
            output: self.output.clone(),
            mesh_len: self.mesh_len.clone(),
            reaches: self.reaches.clone(),
            web: new_web,
            loss: 0.,
        }
    }

    pub fn reweave_layer(&mut self, layer_ix: usize) -> Manifold {
        let mut web = self.web.clone();
        let layer = web.get_mut(layer_ix).unwrap();
        let mut neur_rng = thread_rng();
        let mut cross_rng = thread_rng();

        let mut pick_neuron = || neur_rng.gen_range(0..self.mesh_len) as usize;
        let mut pick_cross = || cross_rng.gen_range(0..layer.len()) as usize;

        let cross = pick_cross();

        let mover_op = layer
            .iter()
            .fold(NeuronOperation::Forward, |nop, op| match op.operation {
                NeuronOperation::Forward => return nop,
                NeuronOperation::Merge | NeuronOperation::Split => return op.operation.clone(),
            });

        for (i, op) in layer.iter_mut().enumerate() {
            if i == cross {
                op.swap_focus(pick_neuron());
                op.swap_op(mover_op.clone())
            } else {
                op.swap_focus(pick_neuron());
                op.swap_op(NeuronOperation::Forward)
            }
        }

        Manifold {
            reaches: self.reaches.clone(),
            web,
            mesh_len: self.mesh_len.clone(),
            input: self.input.clone(),
            output: self.output.clone(),
            loss: 0.,
        }
    }

    pub fn discount_factor(&self) -> f64 {
        let layers = self.web.len();
        -(0.5 / (layers + 1) as f64) + 1.
    }

    pub fn discount(&self, web_layer: usize, factor: f64) -> f64 {
        factor.powi(web_layer as i32)
    }

    pub fn forward(&self, signals: &mut Vec<Signal>, neuros: &Vec<Neuron>) {
        if signals.len() != self.input {
            println!("Length mismatch");
            return;
        }

        let discount_factor = self.discount_factor();

        for (layer, stage) in self.web.iter().enumerate() {
            for op in stage.iter() {
                let discount = self.discount(layer, discount_factor);
                op.execute(signals, neuros, discount)
            }
        }
    }

    pub fn get_num_layers(&self) -> usize {
        self.web.len()
    }

    pub fn accumulate_loss(&mut self, a: f64) {
        self.loss += a;
    }

    pub fn _sequence(&self) -> String {
        let mut seq_slices: Vec<Vec<String>> = Vec::new();
        let mut seq = String::default();

        let mut max_x = 0;
        let max_y = self.web.len();

        for opl in self.web.iter() {
            if opl.len() > max_x {
                max_x = opl.len()
            }

            let op_sequence = opl
                .iter()
                .map(|op| match op.operation {
                    NeuronOperation::Forward => format!("F{}", op.signal_ix),
                    NeuronOperation::Merge => format!("M{}", op.signal_ix),
                    NeuronOperation::Split => format!("S{}", op.signal_ix),
                })
                .collect::<Vec<String>>();

            seq_slices.push(op_sequence)
        }

        for x in 0..max_x {
            for y in 0..max_y {
                match seq_slices.get(y) {
                    Some(yv) => match yv.get(x) {
                        Some(v) => seq.push_str(v),
                        None => seq.push_str("  "),
                    },
                    None => seq.push_str("  "),
                }
            }
            seq.push('\n');
        }

        seq
    }

    pub fn dump(&self) -> Result<String, Box<dyn Error>> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn load(serial: String) -> Result<Manifold, Box<dyn Error>> {
        let manifold: Manifold = serde_json::from_str(&serial)?;
        Ok(manifold)
    }
}
