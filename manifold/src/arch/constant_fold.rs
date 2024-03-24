use std::collections::VecDeque;

use rand::{thread_rng, Rng};

use super::{Manifold, Neuron, Signal};

pub struct ConstantFold {
    d_in: usize,
    d_out: usize,
    reach: Vec<usize>,
}

impl ConstantFold {
    pub fn new(d_in: usize, d_out: usize, reach: Vec<usize>) -> ConstantFold {
        ConstantFold { d_in, d_out, reach }
    }

    pub fn vectorize(signals: &[Signal]) -> Vec<f64> {
        signals.iter().map(|s| s.x.clone()).collect::<Vec<f64>>()
    }

    pub fn signalize(x: &[f64]) -> Vec<Signal> {
        x.iter()
            .map(|x| Signal { x: x.clone() })
            .collect::<Vec<Signal>>()
    }

    pub fn mse(signals: &[Signal], target: &[f64]) -> f64 {
        let svec = ConstantFold::vectorize(signals);

        let diff = target
            .iter()
            .enumerate()
            .map(|(i, e)| (e - svec[i]).powi(2))
            .collect::<Vec<f64>>();

        diff.into_iter().fold(0. as f64, |a, e| a + e) / signals.len() as f64
    }

    pub fn population(
        &self,
        mut from_backtrack: Option<(Manifold, &usize)>,
        neuros: &Vec<Neuron>,
        count: usize,
    ) -> VecDeque<Manifold> {
        let mut p: VecDeque<Manifold> = VecDeque::new();

        for _ in 0..count {
            match &mut from_backtrack {
                Some((manifold, backtrack)) => {
                    let child = manifold.reweave_backtrack(backtrack.clone());
                    p.push_back(child)
                }
                None => {
                    let mut organic =
                        Manifold::new(self.d_in, self.d_out, self.reach.clone(), neuros.len());
                    organic.weave();
                    p.push_back(organic)
                }
            }
        }

        p
    }

    pub fn optimize_traversal(&self, neuros: Vec<Neuron>, x: Vec<Vec<f64>>, y: Vec<Vec<f64>>) {
        let mut rng = thread_rng();
        let set_length = x.len();

        let population_size = 100;
        let carryover_rate = 0.1;
        let elitism_carryover = 3;
        let samples_per_epoch = 40;

        // Also, maybe a hyperparam for splitting new populations based on prev gens
        // successful members. Do we just do an even split or is it based on performance?
        // For now an even split.

        let mut population: VecDeque<Manifold> = self.population(None, &neuros, population_size);
        let max_layers = population.iter().fold(0, |a, m| {
            let l = m.get_num_layers();
            if l > a {
                return l;
            }
            a
        });

        for backtrack in 0..max_layers - 1 {
            let mut sample_x: Vec<&Vec<f64>> = vec![];
            let mut sample_y: Vec<Vec<f64>> = vec![];

            for _ in 0..samples_per_epoch {
                let xy_ix = rng.gen_range(0..set_length);
                sample_x.push(&x[xy_ix]);
                sample_y.push(y[xy_ix].clone());
            }

            for manifold in population.iter_mut() {
                for (i, x) in sample_x.iter_mut().enumerate() {
                    let mut signals = &mut ConstantFold::signalize(x);
                    manifold.forward(&mut signals, &neuros);
                    let loss = ConstantFold::mse(&signals, &sample_y[i]);
                    manifold.accumulate_loss(loss)
                }
            }

            population
                .make_contiguous()
                .sort_unstable_by(|m, n| m.loss.partial_cmp(&n.loss).unwrap());

            println!("Min loss: {}", population[0].loss);

            // Carry over the elite
            let mut elite: VecDeque<Manifold> = VecDeque::new();
            for _ in 0..elitism_carryover {
                let m = population.pop_front();
                if let Some(m) = m {
                    elite.push_back(m)
                }
            }

            let mut carryover: VecDeque<Manifold> = VecDeque::new();
            let num_carryover = (carryover_rate * population_size as f64).floor() as usize;
            for _ in 0..num_carryover {
                let m = population.pop_front();
                if let Some(m) = m {
                    carryover.push_back(m)
                }
            }

            // Append the carryover manifolds that can no longer evolve to elite
            // If inviable, they will wash out.
            // Evolve the rest
            let mut evolvable: VecDeque<Manifold> = VecDeque::new();
            for m in carryover.into_iter() {
                if m.get_num_layers() < backtrack + 1 {
                    elite.push_back(m);
                    continue;
                }
                evolvable.push_back(m);
            }

            let num_evolvable = evolvable.len();

            if num_evolvable == 0 {
                break;
            }

            let mut children_per: Vec<usize> = evolvable.iter().map(|_| 0).collect();

            let mut parent_selector = 0;
            for i in 0..=population_size {
                children_per[parent_selector] += 1;
                parent_selector = i % num_evolvable;
            }

            let mut next_population: VecDeque<Manifold> = VecDeque::new();
            for (i, b) in evolvable.into_iter().enumerate() {
                next_population.append(&mut self.population(
                    Some((b, &backtrack)),
                    &neuros,
                    children_per[i],
                ))
            }

            population = next_population;
        }
    }
}
