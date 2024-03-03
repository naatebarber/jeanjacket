use rand;
use rand::prelude::*;
use super::{Neuron, NeuronOperation, Signal};

pub struct Op {
    neuron: &'static Neuron,
    operation: NeuronOperation,
    target: usize
}

impl Op {
    pub fn execute(&self, signals: &mut Vec<Signal>) {
        match self.operation {
            NeuronOperation::Forward => self.neuron.forward(signals, self.target),
            NeuronOperation::Merge => {
                self.neuron.merge(signals, self.target)
            },
            NeuronOperation::Split => {
                self.neuron.split(signals, self.target)
            }
        }
    }
}

pub struct Manifold {
    input: usize,
    reach: usize,
    output: usize,
    neurons: &'static Vec<Neuron>,
    manifold: Vec<Vec<Op>>
}

impl Manifold {
    pub fn new(input: usize, output: usize, reach: usize, neurons: &'static Vec<Neuron>) -> Manifold {
        Manifold {
            input,
            reach,
            output,
            neurons,
            manifold: vec![]
        }
    }

    pub fn weave_between(&self, s: usize, e: usize) -> Vec<Vec<Op>> {
        let descend = s > e;
        let mut c = s;

        let mut rng = rand::thread_rng();

        let mut weave: Vec<Vec<Op>> = Vec::new();

        // While the number of signals is not the desired size
        while c != s {
            let mut ops: Vec<Op> = Vec::new();
            let mut next_c = c;

            // For every signal in the current number of signals
            for target in 0..=c {
                let neuron = self.neurons.choose(&mut rng).unwrap();
                let op: Op = match descend {
                    true => {
                        next_c -= 1;
                        Op {
                            neuron,
                            operation: NeuronOperation::Merge,
                            target
                        }
                    },
                    false => {
                        next_c += 1;
                        Op {
                            neuron,
                            operation: NeuronOperation::Split,
                            target
                        }
                    }
                };

                ops.push(op);
                c = next_c;
            }

            weave.push(ops)
        }

        weave
    }

    pub fn weave(mut self) {
        let mut l1 = self.weave_between(self.input, self.reach);
        let mut l2 = self.weave_between(self.reach, self.output);
        l1.append(&mut l2);
        self.manifold = l1
    }
}