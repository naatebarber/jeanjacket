use std::collections::VecDeque;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use rand::{thread_rng, Rng};

use super::{Basis, EvolutionHyper, Optimizer};
use crate::f;
use crate::substrates::fully_connected::{Manifold, Population, Signal, Substrate};
use crate::substrates::traits::SignalConversion;

pub struct Turnstile {
    d_in: usize,
    d_out: usize,
    breadth: Range<usize>,
    depth: Range<usize>,
    turn_amplitude: Range<i32>,
    epochs: usize,
}

impl Optimizer<Substrate, Population> for Turnstile {
    fn train(
        &self,
        basis: Basis<Substrate>,
        hyper: EvolutionHyper,
    ) -> (Population, Basis<Substrate>, EvolutionHyper) {
        let EvolutionHyper {
            population_size,
            carryover_rate,
            elitism_carryover,
            ..
        } = hyper;

        let mut population = self.weave_population(basis.neuros.len() - 1, population_size.clone());

        for i in 0..self.epochs {
            Turnstile::evaluate(&mut population, &basis, &hyper);

            println!(
                "({}/{}) Min loss (turnstile optim): {}",
                i,
                self.epochs - 1,
                population[0].lock().unwrap().loss
            );

            let mut elite: Population = VecDeque::new();
            for _ in 0..elitism_carryover {
                match population.pop_front() {
                    Some(m) => {
                        let mut manifold = m.lock().unwrap();
                        manifold.reset_loss();
                        drop(manifold);
                        elite.push_back(m);
                    }
                    None => (),
                };
            }

            // TODO Grab carryover percentage
            // Create a subpopulation percentage from each carryover manifold
            // Arbitrarily turn them based on turn_amplitude

            let mut carryover: Population = VecDeque::new();
            let num_carryover = (carryover_rate * population_size as f64).floor() as usize;
            for _ in 0..num_carryover {
                let m = population.pop_front();
                if let Some(m) = m {
                    carryover.push_back(m)
                }
            }

            let num_carryover = carryover.len();

            let mut children_per: Vec<usize> = (0..num_carryover).map(|_| 0).collect();

            let mut parent_selector = 0;
            for i in 0..=(population_size - elite.len()) {
                children_per[parent_selector] += 1;
                parent_selector = i % num_carryover;
            }

            let mut next_population: Population = VecDeque::new();
            next_population.append(&mut elite);

            let create_population_tasks = carryover
                .into_iter()
                .enumerate()
                .map(|(i, manifold)| {
                    let m = Arc::clone(&manifold);
                    let count = children_per[i];
                    let turn_amplitude = self.turn_amplitude.clone();

                    Box::new(move || Turnstile::turn_population(m, turn_amplitude, count))
                        as Box<dyn (FnOnce() -> Population) + Send>
                })
                .collect::<Vec<_>>();

            let population_slices = Turnstile::distributed::<Population>(create_population_tasks);

            for mut slice in population_slices.into_iter() {
                next_population.append(&mut slice)
            }

            population = next_population;
        }

        (population, basis, hyper)
    }
}

impl Turnstile {
    pub fn new(
        d_in: usize,
        d_out: usize,
        breadth: Range<usize>,
        depth: Range<usize>,
        turn_amplitude: Range<i32>,
        epochs: usize,
    ) -> Turnstile {
        Turnstile {
            d_in,
            d_out,
            breadth,
            depth,
            turn_amplitude,
            epochs,
        }
    }

    pub fn weave_population(&self, num_neurons: usize, count: usize) -> Population {
        let mut p: VecDeque<Arc<Mutex<Manifold>>> = VecDeque::new();

        let mut weave_tasks: Vec<Box<dyn (FnOnce() -> Arc<Mutex<Manifold>>) + Send>> = vec![];

        for _ in 0..count {
            let d_in = self.d_in.clone();
            let d_out = self.d_out.clone();
            let breadth = self.breadth.clone();
            let depth = self.depth.clone();
            let mesh_len = num_neurons.clone();

            let task = Box::new(move || {
                let mut organic = Manifold::dynamic(mesh_len, d_in, d_out, breadth, depth);
                organic.weave();
                Arc::new(Mutex::new(organic))
            }) as Box<dyn (FnOnce() -> Arc<Mutex<Manifold>>) + Send>;

            weave_tasks.push(task);
        }

        let woven_manifolds = Turnstile::distributed::<Arc<Mutex<Manifold>>>(weave_tasks);

        for m in woven_manifolds.into_iter() {
            p.push_back(m)
        }

        p
    }

    pub fn turn_population(
        manifold: Arc<Mutex<Manifold>>,
        turn_amplitude: Range<i32>,
        count: usize,
    ) -> Population {
        let mut p: Population = VecDeque::new();
        let mut rng = thread_rng();

        for _ in 0..count {
            let parent = manifold.lock().unwrap();
            let mut child = parent.clone();
            drop(parent);

            let amplitude = rng.gen_range(turn_amplitude.clone());
            child.reset_loss();
            child.turn(amplitude);

            p.push_back(Arc::new(Mutex::new(child)));
        }

        p
    }

    pub fn evaluate_one(
        manifold_am: Arc<Mutex<Manifold>>,
        sample_x: Vec<Vec<f64>>,
        sample_y: Vec<Vec<f64>>,
        neuros: Substrate,
    ) {
        let mut manifold = manifold_am.lock().unwrap();

        sample_x
            .into_iter()
            .map(|x| Signal::signalize(x))
            .enumerate()
            .for_each(|(i, mut x)| {
                manifold.forward(&mut x, &neuros);
                if x.len() != sample_y[i].len() {
                    panic!("Output malformed")
                }
                let loss = f::binary_cross_entropy(&Signal::vectorize(x), &sample_y[i]);
                manifold.accumulate_loss(loss)
            });
    }

    pub fn evaluate(population: &mut Population, basis: &Basis<Substrate>, hyper: &EvolutionHyper) {
        let mut rng = thread_rng();
        let set_length = basis.x.len();
        let mut sample_x: Vec<Vec<f64>> = vec![];
        let mut sample_y: Vec<Vec<f64>> = vec![];

        for _ in 0..hyper.sample_size {
            let xy_ix = rng.gen_range(0..set_length);
            sample_x.push(basis.x[xy_ix].clone());
            sample_y.push(basis.y[xy_ix].clone());
        }

        let tasks = population
            .iter()
            .map(|m| {
                let sx = sample_x.clone();
                let sy = sample_y.clone();
                let neuros = Arc::clone(&basis.neuros);
                let manifold = Arc::clone(&m);
                Box::new(move || Turnstile::evaluate_one(manifold, sx, sy, neuros))
                    as Box<dyn FnOnce() + Send>
            })
            .collect::<Vec<Box<dyn FnOnce() + Send>>>();

        let _ = Turnstile::distributed::<()>(tasks);

        population.make_contiguous().sort_unstable_by(|m, n| {
            let m = m.lock().unwrap();
            let n = n.lock().unwrap();
            m.loss.partial_cmp(&n.loss).unwrap()
        });
    }
}
