use std::collections::VecDeque;

pub trait SignalConversion<S> {
    fn signalize(vec: Vec<f64>) -> VecDeque<S>;
    fn vectorize(sig: VecDeque<S>) -> Vec<f64>;
}
