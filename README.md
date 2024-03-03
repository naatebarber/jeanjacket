### Neuroevolution stuff

I've got parallelized training going on now (thank you cuda for allowing me to bypass GIL).
Now need to do some stuff:

### IN ORDER:  
 - Move all this spaghetti bs into classes.
 - Experiment with a better env (y = max(x)**2) is weak lol.
 - Play with classification (rn im just doing regression)
 - Add a user dialog with the best organism to manually evaluate
 - Instantiate "species" of organisms, with different weights and architectures.
 - Allow only orgs of the same species to crossover.
 - Look into tensor sharing
 - Make em communicate over network

### A Game Of Telephone

With the new interesting Q star + a larger model training a smaller model topics coming out of openAI i wanna devise a quick
strat to shrink required network size for an inference (n) as far as possible.

This should all be done in a q-learning type approach.

 - Start with a larger, more complex model. Train it until it hits a pre-defined cascading loss.
 - Take this larger model, use it as the environment for a couple differently shaped smaller models, train them to the same cascading loss.
 - Select the best performing n models
 - Use them to train smaller models
 - Rinse and repeat with a predefined generation breadth (# of models allowed to act as trainers per generation)
 - Train until a generation fails to hit the min cascading loss

### Manifold 

mumble mumble learn by finding different pathways through a changing neural mesh mumble mumble

first / fast op: train the pathway, use some alg to find the best pathway
second / slow op: evolve the neurons / swap out unused ones

Cool part ab this architecture is that you could technically house 
completely different experts / models in the same neural mesh since a learned trait is 
mostly dictated by path

Thought pool:

 - Maybe we choose the output neurons beforehand (when making manifold) and use pathfinding algorithm to optimize?
 - Maybe neurons need to be connected in order to forward/split/merge?