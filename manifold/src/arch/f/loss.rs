pub fn mean_squared_error(pred: &[f64], actual: &[f64]) -> f64 {
    let diff = actual
        .iter()
        .enumerate()
        .map(|(i, e)| (e - pred[i]).powi(2))
        .collect::<Vec<f64>>();

    diff.into_iter().fold(0. as f64, |a, e| a + e) / pred.len() as f64
}

pub fn binary_cross_entropy(pred: &[f64], actual: &[f64]) -> f64 {
    if pred.len() != actual.len() {
        panic!("Output length mismatch!");
    }

    let eps = 1e-9;

    let loss: f64 = actual
        .iter()
        .zip(pred.iter())
        .map(|(&y, &y_hat)| {
            let y_hat_clipped = y_hat.max(eps).min(1.0 - eps);
            if y == 1.0 {
                -y * y_hat_clipped.ln()
            } else {
                -(1.0 - y) * (1.0 - y_hat_clipped).ln()
            }
        })
        .sum();

    loss / actual.len() as f64
}
