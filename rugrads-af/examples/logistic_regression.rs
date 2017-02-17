//! This is a super-ugly proof of concept for optimizing auto differentiation
//!
//! TODO: Currently this example panics as there is a bug in the VJP computation somewhere.

extern crate rugrads_af as raf;
use raf::{Array, Dim4};

fn toy_dataset() -> (Array, Array) {
    let input_data = vec![0.52, 1.12,  0.77,
                          0.88, -1.08, 0.15,
                          0.52, 0.06, -1.30,
                          0.74, -2.49, 1.39];
    let target_data = vec![1.0, 1.0, 0.0, 1.0];

    let inputs = raf::new_array(&input_data, raf::Dim2([4, 3]));
    let targets = raf::new_array(&target_data, raf::Dim2([4, 1]));

    (inputs, targets)
}

fn main() {
    /* Need current arrayfire-rust dev branch for this to compile

    // Set the backend and create context for auto diff
    raf::set_backend(raf::Backend::CPU);
    let mut context = raf::Context::new();

    // Create all context variables
    let (inputs, targets) = toy_dataset();
    let x = context.create_variable(inputs);
    let y = context.create_variable(targets);
    let weights = raf::constant(0f64, Dim4::new(&[3, 1, 1, 1]));
    let w = context.create_variable(weights);
    let ones = context.create_variable(raf::constant(1f64, Dim4::new(&[4, 1, 1, 1])));

    // Set up our logistic regression loss function
    let preds = raf::sigmoid(raf::dot(x, w.clone(), raf::MatProp::NONE, raf::MatProp::NONE));
    let label_probs = raf::mul(preds, y.clone() + y.clone() - ones.clone(), false) + ones - y;
    let loss = -raf::sum_all(raf::log(label_probs));

    // Set up Gradient object to auto diff
    let mut g = raf::Gradient::of(loss, context);

    // Optimize the weights
    let alpha = raf::constant(0.01, Dim4::new(&[3, 1, 1, 1]));

    for _ in 0..100 {
        let w_grad = g.grad(w.clone());
        *g.0.get_mut(w.clone()) -= w_grad * &alpha;
    }
    */
}
