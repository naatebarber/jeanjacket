pub mod arch;
pub use arch::*;

#[cfg(test)]
mod test_manifold {
    use crate::substrates::binary::{Manifold, Neuron, Signal};

    #[test]
    fn manifold_transformations() {
        let reach = vec![5];
        let mut din = 1;
        let mut dout = 10;

        let neuros = Neuron::substrate(100, -1.0..1.0);

        while din <= 10 {
            while dout >= 1 {
                let mut manifold = Manifold::new(din, dout, reach.clone(), neuros.len());
                manifold.weave();

                let mut signals = Signal::_random_normal(din);
                manifold.forward(&mut signals, &neuros);

                assert_eq!(signals.len(), dout);
                dout -= 1;
            }
            din += 1;
        }
    }

    #[test]
    fn manifold_reweave_transformations() {
        let reach = vec![5, 10, 5];
        let din = 1;
        let dout = 10;

        let neuros = Neuron::substrate(100, -1.0..1.);

        let mut manifold = Manifold::new(din, dout, reach.clone(), neuros.len());
        manifold.weave();

        for backtrack in 0..manifold.get_num_layers() {
            manifold = manifold.reweave_backtrack(backtrack);

            let mut signals = Signal::_random_normal(din);
            manifold.forward(&mut signals, &neuros);

            assert_eq!(signals.len(), dout);
        }
    }
}
