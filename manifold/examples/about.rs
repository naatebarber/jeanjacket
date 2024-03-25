use manifold::{Manifold, Neuron, Signal};

fn main() {
    let mesh_len: usize = 100;

    let mut manifold = Manifold::new(2, 3, vec![12, 2, 8, 4, 14, 6, 12], mesh_len);
    manifold.weave();

    let neuros = Neuron::substrate(mesh_len);

    let mut test_signals = Signal::_random_normal(2);
    // println!("{:?}", test_signals);

    manifold.forward(&mut test_signals, &neuros);
    // println!("{:?}", test_signals);

    println!("Goal: Light = survival. Learn to seek light.\n");
    println!("Input signals: \n1. Light sensor input\n2. Previous light sensor input\n");
    println!("Output signals: \n1. Rotate left\n2. Rotate right\n3. Move forward\n");

    println!("Each neuron performs a mathematical operation on a signal, \npathways through neurons are evolved based on success of the resulting answer.\n");
    println!("There are 1,000,000 neurons with pre-set influence. \nCurrently the influence of a neuron does not change, the pathway does.");
    println!("This might be hella gay and inefficient because it's like a brain with no cell death or growth.\n");
    println!("F => Push signal from neuron A to neuron B");
    println!("M => Merge signals from neuron A and neuron B into neuron C");
    println!("S => Split signals from neuron A into neurons B and C");

    println!("\nEvolved thought pathway through arbitrary neurons:\n");

    println!("{}", manifold._sequence())
}
