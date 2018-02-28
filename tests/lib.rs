extern crate rugrads;
extern crate num;

use rugrads::functions::*;
use rugrads::{Context, Gradient};

use num::Float;





#[test]
fn test_grad_twice() {
    // First we define our context and function variables
    let mut context = Context::new();
    let x = context.create_variable(0.5);
    let y = context.create_variable(0.3);
    
    // Below we build: y * sin(x) + cos(y)
    let f = y * sin(x) + cos(y);
    let mut grad = Gradient::of(f, context);
    
    // Take gradient with respect to x - has value: 
    let g1 = grad.grad(&x);
    let g2 = grad.grad(&x);

    assert!((g1 - g2).abs() == 0f64)
    
}

