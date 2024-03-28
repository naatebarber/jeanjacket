# Fully Connected 1

In this incarnation, an optimizer is not needed or used.

Substrate is stored in a VecDeque and sorted from least to greatest.

Manifold has two training methods, _explode_ and _methodic_.

Explode: 
 - Finds the operation of greatest influence at layer n, determined by some heuristic (highest action potential)
 - Splits the parent manifold into two children, left and right.
 - creates a vector of "next selves"
 - "turns" the left child counterclockwise, by decrementing the neuron index of it's operation of greatest influence.
 - "turns" the right child clockwise, by incrementing the neuron index of it's operation of greatest influence.
 - `explode(left_self, layer n - 1)` and `explode(right_self, layer n - 1)` are appended to "next selves".
 - the recursive explosion of O(2^n) complexity ends when layer <= 0, then all selves are collected into a single vector.
 - this vector is iterated over.
    - the selves are then evaluated by the same loss fn and data the parent was evolved by.
    - if an alternative self has a lower loss than the root manifold, the root cannibalizes the alternative self (steals it's web and loss)
    - this process repeats, and accumulates the web with lowest loss

Methodic:
 - Iterate through the layers of the parent manifold
 - For each layer find the operation of greatest influence, determined by some heuristic (highest action potential)
 - Splits the parent manifold into two children, left and right.
 - Isolate the layer, and turn the operation right and left.
 - Add the isolated layer updates to an "alternative selves" Manifold vector.
    - The difference from explode here is the lack of combinatorial iteration. This op is O(n).
 - this vector is iterated over.
    - the selves are then evaluated by the same loss fn and data the parent was evolved by.
    - if an alternative self has a lower loss than the root manifold, the root cannibalizes the alternative self's 
    modified layer.
- this process repeats, and the positive changes of alternative selves are accumulated on the root.
