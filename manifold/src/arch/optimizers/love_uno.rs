use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use rand::{thread_rng, Rng};

use super::{Basis, EvolutionHyper, Optimizer};
use crate::substrates::binary::{Manifold, Population, Substrate};

pub struct LoveUno {
    d_in: usize,
    d_out: usize,
    reach: Vec<usize>,
}

impl Optimizer for LoveUno {
    fn train(&self, basis: Basis, hyper: EvolutionHyper) -> (Population, Basis, EvolutionHyper) {
        let Basis { neuros, .. } = basis.clone();
        let EvolutionHyper {
            population_size, ..
        } = hyper;

        let mut population: Population = LoveUno::weave_population(
            self.reach.clone(),
            self.d_in,
            self.d_out,
            neuros.len(),
            population_size,
        );

        let get_max_layers = |population: &Population| {
            population.iter().fold(0, |a, m| {
                let l = m.lock().unwrap().get_num_layers();
                if l > a {
                    return l;
                }
                a
            })
        };

        let mut max_layers = get_max_layers(&population);
        let mut backtrack = 0;

        loop {
            LoveUno::evaluate(&mut population, &basis, &hyper);

            println!(
                "({}/{}) Min loss (love uno optim): {} {}",
                backtrack,
                max_layers - 1,
                population[0].lock().unwrap().loss,
                population[population.len() - 1].lock().unwrap().loss
            );

            // Grab the best
            let best = population.pop_front().unwrap();

            // Mutate the next layer of best * pop size
            let mut next_population =
                LoveUno::reweave_layer_population(best.clone(), backtrack, population_size - 1);
            next_population.push_front(best);

            population = next_population;
            backtrack += 1;
            max_layers = get_max_layers(&population);

            if backtrack >= max_layers {
                break;
            }
        }

        return (population, basis, hyper);
    }
}

impl LoveUno {
    pub fn new(d_in: usize, d_out: usize, reach: Vec<usize>) -> LoveUno {
        LoveUno { d_in, d_out, reach }
    }

    pub fn weave_population(
        reach: Vec<usize>,
        d_in: usize,
        d_out: usize,
        neurons: usize,
        count: usize,
    ) -> Population {
        let mut p: Population = VecDeque::new();

        let mut weave_tasks: Vec<Box<dyn (FnOnce() -> Arc<Mutex<Manifold>>) + Send>> = vec![];

        for _ in 0..count {
            let d_inc = d_in.clone();
            let d_outc = d_out.clone();
            let nc = neurons.clone();
            let reachc = reach.clone();

            let task = Box::new(move || {
                let mut organic = Manifold::new(d_inc, d_outc, reachc, nc);
                organic.weave();
                Arc::new(Mutex::new(organic))
            }) as Box<dyn (FnOnce() -> Arc<Mutex<Manifold>>) + Send>;

            weave_tasks.push(task);
        }

        let woven_manifolds = LoveUno::distributed::<Arc<Mutex<Manifold>>>(weave_tasks);

        for m in woven_manifolds.into_iter() {
            p.push_back(m)
        }

        p
    }

    pub fn reweave_layer_population(
        from: Arc<Mutex<Manifold>>,
        layer_ix: usize,
        count: usize,
    ) -> Population {
        let mut p: Population = VecDeque::new();

        for _ in 0..count {
            let mut parent = from.lock().unwrap();
            let child = parent.reweave_layer(layer_ix.clone());
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
            .map(|x| LoveUno::signalize(&x))
            .enumerate()
            .for_each(|(i, mut x)| {
                manifold.forward(&mut x, &neuros);
                if x.len() != sample_y[i].len() {
                    panic!("Output malformed")
                }
                let loss = LoveUno::mse(&LoveUno::vectorize(&x), &sample_y[i]);
                manifold.accumulate_loss(loss)
            });
    }

    pub fn evaluate(population: &mut Population, basis: &Basis, hyper: &EvolutionHyper) {
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
                Box::new(move || LoveUno::evaluate_one(manifold, sx, sy, neuros))
                    as Box<dyn FnOnce() + Send>
            })
            .collect::<Vec<Box<dyn FnOnce() + Send>>>();

        let _ = LoveUno::distributed::<()>(tasks);

        population.make_contiguous().sort_unstable_by(|m, n| {
            let m = m.lock().unwrap();
            let n = n.lock().unwrap();
            m.loss.partial_cmp(&n.loss).unwrap()
        });
    }
}
