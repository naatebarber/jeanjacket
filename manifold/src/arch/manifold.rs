use super::{Neuron, NeuronOperation, Signal};
use rand;
use rand::prelude::*;

#[derive(Debug)]
pub struct Op<'a> {
    neuron: &'a Neuron,
    operation: NeuronOperation,
    target: usize,
}

impl Op<'_> {
    pub fn execute(&self, signals: &mut Vec<Signal>) {
        match self.operation {
            NeuronOperation::Forward => self.neuron.forward(signals, self.target),
            NeuronOperation::Merge => {
                self.neuron.merge(signals, self.target);
                self.neuron.forward(signals, self.target);
            }
            NeuronOperation::Split => {
                self.neuron.split(signals, self.target);
                self.neuron.forward(signals, self.target);
            }
        }
    }
}

pub struct Manifold<'a> {
    input: usize,
    reach: usize,
    output: usize,
    neurons: &'a Vec<Neuron>,
    web: Vec<Vec<Op<'a>>>,
}

impl Manifold<'_> {
    pub fn new<'a>(
        input: usize,
        output: usize,
        reach: usize,
        neurons: &'a Vec<Neuron>,
    ) -> Manifold<'a> {
        Manifold {
            input,
            reach,
            output,
            neurons,
            web: vec![],
        }
    }

    pub fn weave_between<'a>(&'a mut self, s: usize, e: usize) {
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

            for target in 0..cutoff {
                let neuron = self.neurons.choose(&mut rng).unwrap();

                let op: Op = match (descend, target == cross) {
                    (true, true) => {
                        next_c -= 1;
                        Op {
                            neuron,
                            operation: NeuronOperation::Merge,
                            target,
                        }
                    }
                    (false, true) => {
                        next_c += 1;
                        Op {
                            neuron,
                            operation: NeuronOperation::Split,
                            target,
                        }
                    }
                    (_, false) => Op {
                        neuron,
                        operation: NeuronOperation::Forward,
                        target,
                    },
                };

                ops.push(op);
                c = next_c;
            }

            weave.push(ops)
        }

        self.web.append(&mut weave);
    }

    pub fn weave<'a>(&'a mut self) {
        let steps = vec![self.input, self.reach, self.output];
        for i in 0..(steps.len() - 1) {
            self.weave_between(steps[i], steps[i + 1]);
        }
    }

    pub fn forward(&self, signals: &mut Vec<Signal>) {
        if signals.len() != self.input {
            println!("Length mismatch");
            return;
        }

        for stage in self.web.iter() {
            for op in stage.iter() {
                op.execute(signals)
            }
        }
    }
}
