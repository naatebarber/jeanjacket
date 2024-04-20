# radial dichotomy

use hnsw for recall and mutation. mutate nearest k in skiptree, with increasing influence the farther you get out from the root.

Structure:

 - symbols live on layers, which live on the dichotomy. 
 - currently data is represented as a scalar angle around a circle, 
 but i'll definitely make this an n-d sphere with a vec of angles
 - the outermost layer holds core symbols or what for us would be a priori
 - each subsequent layer incorporates either a combination of past symbols, or a past symbol and a new value.
 - everything - including outputs from the system - must be represented in terms of the systems symbols
 - variant of hnsw
 - low P layers (sparse) represent core memories / central / early influence (layer 0 is a priori)
 - influence decreases the higher P you get, along with the number of symbols in layer P
 - symbols can reference past symbols depending on closeness
