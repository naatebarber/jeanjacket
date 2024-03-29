# Blame Graph 1

Since signals are independent of a larger structure (matrix) we can track their movement and assemble
a 'blame graph' as they pass through the system. The blame graph can then be used to act upon every
signal and it's mutations after an informed forward pass

Implement:
 - blame graph for high speed training
 - change amplitude to a float, representing coverage of the ringbuffer substrate 0.0..1.0
 - add 'sticky' param to manifold. if sticky signals cannot cycle the sorted substrate and get stuch at each end.
 - add 'shifts' to substrate. since substrate is a ringbuffer it can be shifted at either end in reasonable time.
 shift by adding in a low-value or high-value neuron to either the start or end of the substrate. this will mildly change
 the result of the network. could be the basis of low-speed training.

Challenges:
 - figure out how to assign blame
 - figure out how to update blame after a singal handoff. potentially abstract this into a `Box<dyn Fn>`
 - figure out how to apply blame to signals/neurons/ops
 - currently signals are dropped between a high and low layer size, and created on a low to high layer size. 
 maybe we should use the split / merge functionality of binary manifold to maintain blame records

Have multiple blames. 

one for historical significance / how the degree of mutation compares to previous degrees of mutation
one for immediate significance / what was the degree of mutation attributed to the current neuron.

each signal (component of a vector traveling through a neural network) stores blames in a hashmap. training will be done
on this combination of blame graphs.

MANIFOLD BLAME

Keep the long term blame on Manifold, and have the signals modify it upon a loss step.
This will contribute to the instant loss associated with one forward pass.

TODO
 - if an op is pegged at max (with sticky turns enabled), and blame tells it to go higher, distribute it's blame elsewhere.
 - signal recycling? send that motherfucker back through.
 - manifold-managed blame for long-term decision making. huge for RL and strategy.

Friday march 29th 7pm 

The fucking thing learns, but instead of learning the problem itself it adjusts weights to the average of the problem space.
For example - if the problem is binary classification with correct answers being either 0 or 1, the model will converge at 0.5.

I think i have to make the blame graph out of multiple runs. it can't just optimize the most recent pass.