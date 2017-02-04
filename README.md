# Rugrads

This project is a proof-of-concept auto differentiation library written in Rust.
The idea behind auto differentiation is to define a function and have the library
compute the derivative of this function for us implicitly.

## What can rugrads do now?

Compute the gradient of simple functions made up of `sin`, `cos`, `add`,
and `mul` operations.

It is designed to be easy to extend rugrads to include user defined
functions.

## How to use rugrads?

This is an early draft and the API needs a lot more work. Below is an example
of how you would specify a function in rugrads to be differentiated.

```rust
extern crate rugrads;

// Here we import `Context` and some functions
use rugrads::*;

// First we define our context and function variables
let mut context = Context::new();
let x = context.create_variable(0.5);
let y = context.create_variable(0.3);

// Below we build: y * sin(x) + cos(y)
// Take sin of x
let f = Sin(x);

// Multiply f with y
let g = Mul(y, f);

// Take cos(y) and add it to g
let h = Cos(y);
let fin = Add(h, g);

let mut grad = Gradient::of(fin, context);

// Take gradient with respect to x - has value: 
grad.grad(x);
grad.grad(y);

// We can also change the initial seed values and recompute:
grad.context().set_variable_value(x, 0.8);
grad.grad(x);
```

The API is still actively evolving to be more flexible. I would love to receive
any suggestions!

## References

- [autograd](https://github.com/HIPS/autograd/tree/master/autograd)
