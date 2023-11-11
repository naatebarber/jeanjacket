### Neuroevolution stuff

I've got parallelized training going on now (thank you cuda for allowing me to bypass GIL).
Now need to do some stuff:

### IN ORDER:  
 - Move all this spaghetti bs into classes.
 - Experiment with a better env (y = max(x)**2) is weak lol.
 - Add a user dialog with the best organism to manually evaluate
 - Instantiate "species" of organisms, with different weights and architectures.
 - Allow only orgs of the same species to crossover.
 - Look into tensor sharing
 - Make em communicate over network