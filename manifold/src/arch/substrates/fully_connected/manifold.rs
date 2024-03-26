use std::{
    collections::{HashMap, VecDeque},
    ops::Range,
};

use rand::{seq::SliceRandom, thread_rng, Rng};

use super::{Signal, Substrate};

#[derive(Clone)]
pub struct Op {
    neuron_ix: usize,
}

impl Op {
    pub fn forward(
        &self,
        target_signal: &mut Signal,
        signals: &mut VecDeque<Signal>,
        neuros: &Substrate,
        _discount: f64,
    ) -> Signal {
        let neuron = neuros.get(self.neuron_ix).unwrap();
        let xout = signals.iter().fold(0., |a, s| a + s.x);
        target_signal.x += xout;
        neuron.forward(target_signal, 1.);
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
                    });
                }
            }

            self.web.push(ixlookup);

            prev_signals = *signals;
        }
    }

    pub fn forward(&self, signals: &mut VecDeque<Signal>, neuros: &Substrate) {
        for layer in self.web.iter() {
            let total_turns = signals.len();
            let mut turns = 0;

            while turns < total_turns {
                let mut target_signal = match signals.pop_front() {
                    Some(s) => s,
                    None => return,
                };

                // Need multiple ops per layer for growing network size.
                let ops_for_turn = match layer.get(&turns) {
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
                    .iter()
                    .map(|op| op.forward(&mut target_signal, signals, &neuros, 1.))
                    .collect::<VecDeque<Signal>>();

                signals.append(&mut next_signals);
            }
        }
    }

    pub fn turn(&mut self, amplitude: i32) {
        let turn_one = |neuron_ix: usize, amplitude: i32| -> usize {
            let mut turned = (neuron_ix as i32) - amplitude;
            let mesh_len = self.mesh_len as i32;
            while turned.abs() > mesh_len {
                turned -= match turned < 0 {
                    true => -mesh_len,
                    false => mesh_len,
                };
            }

            if turned < 0 {
                turned = mesh_len - turned;
            }

            turned as usize
        };

        for layer in self.web.iter_mut() {
            for (_, opvec) in layer.iter_mut() {
                for op in opvec.iter_mut() {
                    op.neuron_ix = turn_one(op.neuron_ix, amplitude);
                }
            }
        }
    }

    pub fn accumulate_loss(&mut self, a: f64) {
        self.loss += a;
    }

    pub fn reset_loss(&mut self) {
        self.loss = 0.;
    }
}
