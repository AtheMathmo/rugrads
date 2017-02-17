extern crate rugrads_af;
extern crate arrayfire as libaf;

use rugrads_af::*;

#[test]
fn test_sin_cos() {
    libaf::set_backend(libaf::Backend::CPU);
    let mut context = Context::new();
    
    let arr = new_array(&[0.5; 4], Dim2([2,2]));
    let x = context.create_variable(arr);
    let f = sin(cos(x.clone()));

    let mut g = Gradient::of(f, context);
    g.grad(x);    
}

#[test]
fn test_sum_all() {
    libaf::set_backend(libaf::Backend::CPU);
    let mut context = Context::new();
    
    let arr = new_array(&[0.5; 4], Dim2([2,2]));
    let x = context.create_variable(arr);
    let f = sum_all(x.clone());

    let mut g = Gradient::of(f, context);
    g.grad(x);
}

