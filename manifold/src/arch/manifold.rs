use super::{Neuron, NeuronOperation, Signal};
use rand;
use rand::prelude::*;

#[derive(Debug, Clone)]
pub struct Op {
    operation: NeuronOperation,
    neuron_ix: usize,
    signal_ix: usize,
}

impl Op {
    pub fn execute(&self, signals: &mut Vec<Signal>, mesh: &Vec<Neuron>) {
        let neuro = mesh.get(self.neuron_ix).unwrap();

        match self.operation {
            NeuronOperation::Forward => neuro.forward(signals, self.signal_ix),
            NeuronOperation::Merge => {
                neuro.merge(signals, self.signal_ix);
                neuro.forward(signals, self.signal_ix);
            }
            NeuronOperation::Split => {
                neuro.split(signals, self.signal_ix);
                neuro.forward(signals, self.signal_ix);
            }
        }
    }
}

pub struct Manifold {
    input: usize,
    reaches: Vec<usize>,
    reach_points: Vec<usize>,
    output: usize,
    mesh_len: usize,
    web: Vec<Vec<Op>>,
    _noise: f32,
}

impl Manifold {
    pub fn new<'a>(input: usize, output: usize, reaches: Vec<usize>, mesh_len: usize) -> Manifold {
        Manifold {
            input,
            reaches,
            reach_points: vec![],
            output,
            mesh_len,
            web: vec![],
            _noise: 0.,
        }
    }

    pub fn _with_noise(&mut self, n: f32) {
        self._noise = n;
    }

    pub fn weave_between<'a>(&'a mut self, s: usize, e: usize) -> Vec<Vec<Op>> {
        let descend = s > e;
        let mut c = s;

        let mut rng = rand::thread_rng();

        let mut weave: Vec<Vec<Op>> = Vec::new();

        // While the number of signals is not the desired size
        while c != e {
            let mut ops: Vec<Op> = Vec::new();
            let mut next_c = c;

            let cross = rng.gen_range(0..c);
            let cutoff = match descend {
                true => c - 1,
                false => c,
            };

            for signal_ix in 0..cutoff {
                let neuron_ix = rng.gen_range(0..self.mesh_len);

                let op: Op = match (descend, signal_ix == cross) {
                    (true, true) => {
                        next_c -= 1;
                        Op {
                            operation: NeuronOperation::Merge,
                            neuron_ix,
                            signal_ix,
                        }
                    }
                    (false, true) => {
                        next_c += 1;
                        Op {
                            operation: NeuronOperation::Split,
                            neuron_ix,
                            signal_ix,
                        }
                    }
                    (_, false) => Op {
                        operation: NeuronOperation::Forward,
                        neuron_ix,
                        signal_ix,
                    },
                };

                ops.push(op);
                c = next_c;
            }

            weave.push(ops)
        }

        return weave;
    }

    pub fn weave(&mut self) {
        let steps = vec![vec![self.input], self.reaches.clone(), vec![self.output]].concat();
        for i in 0..(steps.len() - 1) {
            let mut weave = self.weave_between(steps[i], steps[i + 1]);
            self.web.append(&mut weave);
            self.reach_points.push(self.web.len())
        }
    }

    pub fn reweave_backtrack(&mut self, backtrack: usize) -> Manifold {
        let op_ix = self.web.len() - backtrack;
        let backtrack_op = self.web.get(op_ix).unwrap();
        let weave_start_size = backtrack_op.len();

        let mut next_reach_ix = 0;
        for (i, reach_pt) in self.reach_points.iter().enumerate() {
            if *reach_pt >= op_ix {
                next_reach_ix = i;
            }
        }

        let from_backtrack_to_end: Vec<usize> = self.reaches[next_reach_ix..].to_vec();
        let steps = vec![vec![weave_start_size], from_backtrack_to_end].concat();

        let new_web = self.web.clone();
        let mut split_web = new_web[0..op_ix].to_vec();

        let new_reach_points = self.reach_points.clone();
        let mut split_reach_points = new_reach_points[0..next_reach_ix].to_vec();

        for i in 0..(steps.len() - 1) {
            let mut weave = self.weave_between(steps[i], steps[i + 1]);
            split_web.append(&mut weave);
            split_reach_points.push(split_web.len());
        }

        Manifold {
            input: self.input.clone(),
            output: self.output.clone(),
            mesh_len: self.mesh_len.clone(),
            reaches: self.reaches.clone(),
            reach_points: split_reach_points,
            web: split_web,
            _noise: 0.,
        }
    }

    pub fn forward(&self, signals: &mut Vec<Signal>, neuros: &Vec<Neuron>) {
        if signals.len() != self.input {
            println!("Length mismatch");
            return;
        }

        for stage in self.web.iter() {
            for op in stage.iter() {
                op.execute(signals, neuros)
            }
        }
    }

    pub fn sequence(&self) -> String {
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
                    NeuronOperation::Forward => format!("F"),
                    NeuronOperation::Merge => format!("M"),
                    NeuronOperation::Split => format!("S"),
                })
                .collect::<Vec<String>>();

            seq_slices.push(op_sequence)
        }

        for x in 0..max_x {
            for y in 0..max_y {
                match seq_slices.get(y) {
                    Some(yv) => match yv.get(x) {
                        Some(v) => seq.push_str(v),
                        None => seq.push(' '),
                    },
                    None => seq.push(' '),
                }
            }
            seq.push('\n');
        }

        seq
    }
}
