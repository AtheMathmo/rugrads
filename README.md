# Rugrads

This project is a proof-of-concept auto differentiation library written in Rust.
The idea behind auto differentiation is to define a function and have the library
compute the derivative of this function for us implicitly.

## What can rugrads do now?

Compute the gradient of simple functions made up of `sin`, `cos`, `add`,
and `mul` operations.

## How to use rugrads?

This is an early draft and the API needs a **lot** of work. Below is an example
of how you would specify a function in rugrads to be differentiated.

```rust
extern crate rugrads;

use std::rc::Rc;
use rugrads::grad;
use rugrads::graph::{Graph, Node};
use rugrads::functions::*;

// Create an Rc pointer to a graph.
let graph = Rc::new(Graph::new());
// Create new variables with values 0.5 and 0.3
let x = Graph::create_variable(graph.clone(), 0.5);
let y = Graph::create_variable(graph.clone(), 0.3);

// Below we build: y * sin(x) + cos(y)
// Take sin of x
let f = Sin::sin(graph.clone(), x.index());
// Multiply f with y
let g = Mul::mul(graph.clone(), y.index(), f.index());
// Take cos(y) and add it to g
let h = Cos::cos(graph.clone(), y.index());
let final = Add::add(graph.clone(), h.index(), g.index());

// Take gradient with respect to x 
// This is 0.3 * cos(0.5) = 0.263
let grad_x = grad(graph.clone(), final.clone(), x.clone());

// We can also take the gradient with respect to y
// This is sin(0.5) - sin(0.3) = 0.184
let grad_x = grad(graph.clone(), final.clone(), y); 
```

As you can see this is pretty verbose. It is also very error prone - you can easily
find yourself with panics and no good reason why (unwrapping of dead Rc values). Additionally
you cannot change the values of x and y once they are set. This is obviously bad and is first
on the list of improvements.

I'll be trying to find ways to improve the API now.

## References

- [autograd](https://github.com/HIPS/autograd/tree/master/autograd)
