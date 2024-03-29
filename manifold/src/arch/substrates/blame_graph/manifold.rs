use std::{
    cell::{Ref, RefCell},
    collections::{HashMap, VecDeque},
    ops::Range,
    rc::Rc,
    sync::Arc,
};

use rand::{seq::SliceRandom, thread_rng, Rng};

use super::{blame::Blame, Signal, Substrate};
use crate::substrates::traits::SignalConversion;

#[derive(Clone, Debug)]
pub struct Op {
    pub id: usize,
    pub layer: usize,
    pub neuron_ix: usize,
}

impl Op {
    pub fn forward(
        &mut self,
        target_signal: &mut Signal,
        signals: &mut VecDeque<Signal>,
        neuros: &Substrate,
    ) -> Signal {
        let neuron = match neuros.get(self.neuron_ix) {
            Some(x) => x,
            None => panic!("Tried to forward neuron {}", self.neuron_ix),
        };

        let next_value = neuron.forward(Signal::vectorize(signals.clone()));
        target_signal.mutate(next_value, self);
        target_signal.clone()
    }

    pub fn swap_focus(&mut self, neuron_ix: usize) {
        self.neuron_ix = neuron_ix;
    }
}

pub type LayerSchema = Vec<usize>;
pub type Layer = HashMap<usize, Vec<Rc<RefCell<Op>>>>;
pub type Web = Vec<Layer>;
pub type Flat = HashMap<usize, Rc<RefCell<Op>>>;

#[derive(Clone)]
pub struct Manifold {
    mesh_len: usize,
    d_in: usize,
    d_out: usize,
    layers: LayerSchema,
    web: Web,
    flat: Flat,
    pub loss: f64,
}

impl Manifold {
    pub fn new(mesh_len: usize, d_in: usize, d_out: usize, layers: Vec<usize>) -> Manifold {
        Manifold {
            mesh_len,
            d_in,
            d_out,
            layers,
            web: Web::new(),
            flat: Flat::new(),
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
            web: Web::new(),
            flat: Flat::new(),
            layers,
            loss: 0.,
        }
    }

    pub fn weave(&mut self) {
        let mut rng = thread_rng();
        let mut neuron_rng = thread_rng();
        let mut pick_neuron = || neuron_rng.gen_range(0..self.mesh_len);
        let mut current_op: usize = 0;

        let steps = vec![self.layers.clone(), vec![self.d_out]].concat();
        let mut prev_signals = self.d_in;

        for (layer_ix, signals) in steps.iter().enumerate() {
            let mut layer = Layer::new();

            if prev_signals > *signals {
                // Shrink previous into current

                let mut prev_six = (0..prev_signals).collect::<Vec<usize>>();
                prev_six.shuffle(&mut rng);

                for _ in 0..*signals {
                    let six = prev_six.pop().unwrap();
                    let op = Rc::new(RefCell::new(Op {
                        id: current_op,
                        layer: layer_ix,
                        neuron_ix: pick_neuron(),
                    }));

                    layer.insert(six, vec![Rc::clone(&op)]);
                    self.flat.insert(current_op, Rc::clone(&op));
                    current_op += 1;
                }
            } else {
                // Extend previous into current.
                for six in 0..prev_signals {
                    let op = Rc::new(RefCell::new(Op {
                        id: current_op,
                        layer: layer_ix,
                        neuron_ix: pick_neuron(),
                    }));

                    layer.insert(six, vec![Rc::clone(&op)]);
                    self.flat.insert(current_op, Rc::clone(&op));
                    current_op += 1;
                }

                let remaining_expand = signals - prev_signals;
                let new_six = (0..remaining_expand)
                    .map(|_| rng.gen_range(0..prev_signals))
                    .collect::<Vec<usize>>();

                for six in new_six.iter() {
                    let op = Rc::new(RefCell::new(Op {
                        id: current_op,
                        layer: layer_ix,
                        neuron_ix: pick_neuron(),
                    }));

                    let ops = layer.get_mut(six).unwrap();
                    ops.push(Rc::clone(&op));
                    self.flat.insert(current_op, Rc::clone(&op));
                    current_op += 1;
                }
            }

            self.web.push(layer);

            prev_signals = *signals;
        }

        for layer in self.web.iter() {
            println!("{:?}", layer);
            println!();
        }
    }

    pub fn forward(&mut self, signals: &mut VecDeque<Signal>, neuros: &Substrate) {
        for (_, layer) in self.web.iter_mut().enumerate() {
            let total_turns = signals.len();
            let mut turns = 0;

            let mut revisit_merge: VecDeque<Signal> = VecDeque::new();
            let mut forwarded = 0;

            while turns < total_turns {
                let mut target_signal = match signals.pop_front() {
                    Some(s) => s,
                    None => return,
                };

                // INFO It's not guaranteed the signals will be in order.
                // For example in a shrink operation, weave might have retained signal 7
                // but dropped signal 1. That is why we revisit.
                let ops_for_turn = match layer.get_mut(&turns) {
                    Some(x) => {
                        turns += 1;
                        x
                    }
                    None => {
                        turns += 1;
                        revisit_merge.push_back(target_signal);
                        continue;
                    }
                };

                // INFO Split happens here where we pass one signal to potentially many ops,
                // cloning it.
                let mut next_signals = ops_for_turn
                    .iter_mut()
                    .map(|op| {
                        forwarded += 1;
                        op.borrow_mut()
                            .forward(&mut target_signal, signals, &neuros)
                    })
                    .collect::<VecDeque<Signal>>();

                signals.append(&mut next_signals);
            }

            // MERGING HURTS THE ALGORITHM!
            // for signal in revisit_merge.into_iter() {
            //     let mut least_mutations_ix = 0;
            //     let mut min_mutations = usize::MAX;
            //     for (ix, signal) in signals.iter_mut().enumerate() {
            //         if signal.mutations <= min_mutations {
            //             min_mutations = signal.mutations;
            //             least_mutations_ix = ix;
            //         }
            //     }

            //     signals
            //         .get_mut(least_mutations_ix)
            //         .unwrap()
            //         .merge_seniority(signal);
            // }
        }
    }

    pub fn turn_one(total: usize, neuron_ix: usize, amplitude: i64, sticky: bool) -> usize {
        let mut turned = (neuron_ix as i64) + amplitude;
        let mesh_len = total as i64;

        if sticky {
            if turned > mesh_len {
                return mesh_len as usize;
            }

            if turned < 0 {
                return 0;
            }
        }

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

    pub fn mutate_op(&mut self, id: usize) -> Rc<RefCell<Op>> {
        if id > self.flat.len() {
            panic!(
                "Operation {} out of bounds for {} operation manifold.",
                id,
                self.flat.len()
            );
        }

        Rc::clone(self.flat.get(&id).unwrap())
    }

    pub fn apply_blame(
        &mut self,
        signals: &VecDeque<Signal>,
        expected: &[f64],
        green_fn: impl Fn(&VecDeque<Signal>, &[f64]) -> Vec<f64>,
        max_step: f64,
    ) -> f64 {
        println!(
            "Processed Prediction: {:?}",
            signals.iter().map(|x| x.x).collect::<Vec<f64>>()
        );

        println!("Expected: {:?}", expected);

        let free_energy = green_fn(&signals, &expected);

        println!("Loss: {:?}", free_energy);

        // Combine free energy of the system

        let mut each_free_energy = signals
            .iter()
            .zip(free_energy.iter())
            .map(|(signal, free_energy)| signal.distribute_free_energy(*free_energy))
            .collect::<Vec<Blame>>();

        let mut combined_free_energy = match each_free_energy.pop() {
            Some(x) => x,
            None => return 0.,
        };

        while let Some(merge) = each_free_energy.pop() {
            combined_free_energy.merge_sum(merge);
        }

        println!("Free Energy: {:?}", combined_free_energy);

        // Combine state of the system

        let mut each_blame = signals
            .iter()
            .map(|signal| signal.pop_blame())
            .collect::<Vec<Blame>>();

        let mut combined_blame = match each_blame.pop() {
            Some(x) => x,
            None => return 0.,
        };

        while let Some(merge) = each_blame.pop() {
            combined_blame.merge_sum(merge);
        }

        println!("Blame: {:?}", combined_blame);

        // Calculate max free energy

        let max_free_energy = combined_free_energy.max();

        if max_free_energy == 0. {
            return 0.;
        }

        let corrective_steps = combined_free_energy
            .iter_with_associated(&combined_blame)
            .into_iter()
            .map(|(op_ix, free_energy, influence)| {
                // DID OP CONTRIBUTE OR HARM?
                // if op contributed to a positive move (one in the right direction) increase it's strength
                // if op contributed to a negative move decrease it's strength.
                // Right now I'm just flattening everything to reduce free energy.

                let step_size = max_step * (free_energy / max_free_energy);
                let direction = match *influence > 0. {
                    true => -1,
                    false => 1,
                };

                let mut steps = (step_size * self.mesh_len as f64) as i64;

                steps *= direction;

                (*op_ix, steps)
            })
            .collect::<Vec<(usize, i64)>>();

        println!("Corrective steps: {:?}", corrective_steps);

        for (op_ix, corrective_steps) in corrective_steps.iter() {
            let op_ref = self.mutate_op(*op_ix);
            let mut op = op_ref.borrow_mut();
            let current_neuron_ix = op.neuron_ix;
            let next_neuron_ix =
                Manifold::turn_one(self.mesh_len, current_neuron_ix, *corrective_steps, true);
            op.swap_focus(next_neuron_ix);
        }

        combined_free_energy.sum()
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
                    let _o = o.borrow();
                    nmatch.push_str(format!("{},", _o.neuron_ix).as_str());
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
                    let _o = o.borrow();
                    ixlist.push(_o.neuron_ix.clone());
                }
            }
        }

        ixlist.iter().fold(0., |a, v| a + *v as f64) / ixlist.len() as f64
    }
}

pub type LossFn = Box<dyn Fn(&[f64], &[f64]) -> f64>;
pub type PostProcessor = Box<dyn Fn(VecDeque<Signal>) -> Vec<f64>>;
