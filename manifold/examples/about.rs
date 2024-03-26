use manifold::substrates::binary::{Manifold, Neuron, Signal};

fn main() {
    let mesh_len: usize = 100;

    let din = 1;
    let dout = 10;

    let mut manifold = Manifold::new(din, dout, vec![2, 2, 1], mesh_len);
    manifold.weave();

    println!("{}", manifold._sequence());

    // for backtrack in 0..manifold.get_num_layers() {
    //     manifold = manifold.reweave_backtrack(backtrack);
    // }

    let neuros = Neuron::substrate(mesh_len, -2.0..2.0);

    let mut test_signals = Signal::_random_normal(din);
    manifold.forward(&mut test_signals, &neuros);
    // // println!("{:?}", test_signals);

    // println!("{:?}", test_signals);

    // println!("{}", manifold._sequence());

    // println!("Goal: Light = survival. Learn to seek light.\n");
    // println!("Input signals: \n1. Light sensor input\n2. Previous light sensor input\n");
    // println!("Output signals: \n1. Rotate left\n2. Rotate right\n3. Move forward\n");

    // println!("Each neuron performs a mathematical operation on a signal, \npathways through neurons are evolved based on success of the resulting answer.\n");
    // println!("There are 1,000,000 neurons with pre-set influence. \nCurrently the influence of a neuron does not change, the pathway does.");
    // println!("This might be hella gay and inefficient because it's like a brain with no cell death or growth.\n");
    // println!("F => Push signal from neuron A to neuron B");
    // println!("M => Merge signals from neuron A and neuron B into neuron C");
    // println!("S => Split signals from neuron A into neurons B and C");

    // println!("\nEvolved thought pathway through arbitrary neurons:\n");

    // println!("{}", manifold._sequence())
}
