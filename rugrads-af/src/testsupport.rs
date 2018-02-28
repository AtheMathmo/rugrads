//! Module providing test support functions for arrayfire

use libaf;
use libaf::Array;

use rugrads::{Expression, Variable, Container};

// pub use rugrads::testsupport::finite_diff_grad;

use ::{Context, Gradient};

/// Estimate the gradient using finite differences
pub fn finite_diff_grad<E>(expr: &Container<Array,E>, c: &mut Context, var: &Variable, h: Array) -> Array
    where E: Expression<Array>
{
    let curr_val = c.get_variable_value(var);
    let plus_val = &curr_val + h.clone() / 2.0;
    let minus_val = curr_val - h.clone() / 2.0;

    c.set_variable_value(var, plus_val);
    let plus_eval = expr.eval(c).value().clone();
    c.set_variable_value(var, minus_val);
    let minus_eval = expr.eval(c).value().clone();

    return (plus_eval - minus_eval) / h

}

/// Compare the autodiff gradient to the finite diff gradient
pub fn compare_grads<E>(expr: Container<Array,E>, mut c: Context, var: &Variable, h: Array) -> (Array, Array)
    where E: Expression<Array>
{
    let fin_diff_grad = finite_diff_grad(&expr, &mut c, var, h);

    let mut g = Gradient::of(expr, c);
    let auto_grad = g.grad(var);

    (fin_diff_grad, auto_grad)
}

// Check array equality up to some tolerance
pub fn array_eq(arr1: &Array, arr2: &Array, summed_diff: f64) -> bool {
    let arr_abs_diff = libaf::abs(&libaf::sub(arr1, arr2, false));
    libaf::sum_all(&arr_abs_diff).0 <= summed_diff
}

mod tests {
    use super::*;
    use libaf::{Array, Dim4};
    
    #[test]
    fn test_arr_same_eq() {
        libaf::set_backend(libaf::Backend::CPU);
        let data = [2.0; 4];
        let dims = Dim4::new(&[2,2,1,1]);
        let arr = Array::new(&data, dims);
        let arr2 = Array::new(&data, dims);
        assert!(array_eq(&arr, &arr2, 0f64))
    }

    #[test]
    fn test_arr_eq() {
        libaf::set_backend(libaf::Backend::CPU);
        let data = [3.0; 16];
        let data2 = [2.0; 16];
        let dims = Dim4::new(&[2,2,2,2]);
        let arr = Array::new(&data, dims);
        let arr2 = Array::new(&data2, dims);
        assert!(array_eq(&arr, &arr2, 16f64))
    }
}