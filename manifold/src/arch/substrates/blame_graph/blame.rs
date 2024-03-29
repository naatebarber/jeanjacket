use std::collections::HashMap;

use crate::f;

pub type BlameGraph = HashMap<usize, f64>;

#[derive(Clone, Debug)]
pub struct Blame {
    pub graph: HashMap<usize, f64>,
}

impl Blame {
    pub fn new() -> Blame {
        Blame {
            graph: BlameGraph::new(),
        }
    }

    pub fn add(&mut self, target: usize, influence: f64) {
        if self.graph.contains_key(&target) {
            let current_influence = self.graph.get_mut(&target).unwrap();
            let average_next = (*current_influence + influence) / 2.;
            *current_influence = average_next;
            return;
        }

        self.graph.insert(target, influence);
    }

    pub fn merge_weighted(&mut self, blame: Blame, wx: f64, wy: f64) {
        for (target, influence) in blame.graph.into_iter() {
            if self.graph.contains_key(&target) {
                let self_influence = self.graph.get_mut(&target).unwrap();
                let next_influence = f::weighted_average(*self_influence, wx, influence, wy);
                *self_influence = next_influence;
                continue;
            }

            let next_influence = f::weighted_average(0., wx, influence, wy);
            self.graph.insert(target, next_influence);
        }
    }

    pub fn merge_sum(&mut self, blame: Blame) {
        for (target, influence) in blame.graph.into_iter() {
            if self.graph.contains_key(&target) {
                let self_influence = self.graph.get_mut(&target).unwrap();
                *self_influence = *self_influence + influence;
                continue;
            }

            self.graph.insert(target, influence);
        }
    }

    pub fn distribute(&self, x: f64) -> Blame {
        let total_blame = self.graph.values().fold(0., |a, v| a + ((*v).abs()));
        let mut distributed_alternative = self.clone();
        for (_, influence) in distributed_alternative.graph.iter_mut() {
            let assignment = (*influence / total_blame).abs();
            *influence = assignment * x;
        }

        distributed_alternative
    }

    pub fn sum(&self) -> f64 {
        self.graph.values().fold(0., |a, v| a + v)
    }

    pub fn max(&self) -> f64 {
        self.graph.values().fold(f64::MIN, |a, v| {
            if *v >= a {
                return *v;
            }
            a
        })
    }

    pub fn iter_with_associated<'a>(&'a self, blame: &'a Blame) -> Vec<(&usize, &f64, &f64)> {
        self.graph
            .keys()
            .filter_map(|k| match (k, self.graph.get(&k), blame.graph.get(k)) {
                (_, Some(x), Some(y)) => Some((k, x, y)),
                _ => None,
            })
            .collect::<Vec<(&usize, &f64, &f64)>>()
    }
}
