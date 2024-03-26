use std::collections::VecDeque;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use rand::{thread_rng, Rng};

use super::super::{Manifold, Neuron};
use super::{Basis, EvolutionHyper, Optimizer, Population};

pub struct Dynamic {
    d_in: usize,
    d_out: usize,
    breadth: Range<usize>,
    depth: Range<usize>,
    epochs: usize,
}

impl Optimizer for Dynamic {
    fn train(&self, basis: Basis, hyper: EvolutionHyper) -> (Population, Basis, EvolutionHyper) {
        let mut population =
            self.weave_population(basis.neuros.len(), hyper.population_size.clone(), None);

        for i in 0..self.epochs {
            Dynamic::evaluate(&mut population, &basis, &hyper);

            println!(
                "({}/{}) Min loss (dynamic optim): {}",
                i,
                self.epochs - 1,
                population[0].lock().unwrap().loss
            );

            let mut elite: Population = VecDeque::new();
            for _ in 0..hyper.elitism_carryover {
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

            let mut new_population = self.weave_population(
                basis.neuros.len(),
                hyper.population_size - hyper.elitism_carryover,
                None,
            );

            new_population.append(&mut elite);

            population = new_population;
        }

        (population, basis, hyper)
    }
}

impl Dynamic {
    pub fn new(
        d_in: usize,
        d_out: usize,
        breadth: Range<usize>,
        depth: Range<usize>,
        epochs: usize,
    ) -> Dynamic {
        Dynamic {
            d_in,
            d_out,
            breadth,
            depth,
            epochs,
        }
    }

    pub fn weave_population(
        &self,
        num_neurons: usize,
        count: usize,
        reach: Option<Vec<usize>>,
    ) -> Population {
        let mut p: VecDeque<Arc<Mutex<Manifold>>> = VecDeque::new();

        let mut weave_tasks: Vec<Box<dyn (FnOnce() -> Arc<Mutex<Manifold>>) + Send>> = vec![];

        for _ in 0..count {
            let d_in = self.d_in.clone();
            let d_out = self.d_out.clone();
            let breadth = self.breadth.clone();
            let depth = self.depth.clone();
            let neuron_ct = num_neurons.clone();
            let reach = reach.clone();

            let task = Box::new(move || {
                let mut organic = match reach {
                    Some(x) => Manifold::new(d_in, d_out, x, neuron_ct),
                    None => Manifold::dynamic(d_in, d_out, breadth, depth, neuron_ct),
                };

                organic.weave();
                Arc::new(Mutex::new(organic))
            }) as Box<dyn (FnOnce() -> Arc<Mutex<Manifold>>) + Send>;

            weave_tasks.push(task);
        }

        let woven_manifolds = Dynamic::distributed::<Arc<Mutex<Manifold>>>(weave_tasks);

        for m in woven_manifolds.into_iter() {
            p.push_back(m)
        }

        p
    }

    pub fn evaluate_one(
        manifold_am: Arc<Mutex<Manifold>>,
        sample_x: Vec<Vec<f64>>,
        sample_y: Vec<Vec<f64>>,
        neuros: Arc<Vec<Neuron>>,
    ) {
        let mut manifold = manifold_am.lock().unwrap();

        sample_x
            .into_iter()
            .map(|x| Dynamic::signalize(&x))
            .enumerate()
            .for_each(|(i, mut x)| {
                manifold.forward(&mut x, &neuros);
                if x.len() != sample_y[i].len() {
                    panic!("Output malformed")
                }
                let loss = Dynamic::mse(&Dynamic::vectorize(&x), &sample_y[i]);
                manifold.accumulate_loss(loss)
            });
    }

    pub fn evaluate(
        population: &mut VecDeque<Arc<Mutex<Manifold>>>,
        basis: &Basis,
        hyper: &EvolutionHyper,
    ) {
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
                Box::new(move || Dynamic::evaluate_one(manifold, sx, sy, neuros))
                    as Box<dyn FnOnce() + Send>
            })
            .collect::<Vec<Box<dyn FnOnce() + Send>>>();

        let _ = Dynamic::distributed::<()>(tasks);

        population.make_contiguous().sort_unstable_by(|m, n| {
            let m = m.lock().unwrap();
            let n = n.lock().unwrap();
            m.loss.partial_cmp(&n.loss).unwrap()
        });
    }
}
