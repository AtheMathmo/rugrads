//! Module providing test support functions for arrayfire

use libaf;
use libaf::Array;

use rugrads::{Expression, Node, VecJacProduct};

use ::Context;

pub fn array_eq(arr1: &Array, arr2: &Array, summed_diff: f64) -> bool {
    let arr_abs_diff = libaf::abs(&libaf::sub(arr1, arr2, false));
    libaf::sum_all(&arr_abs_diff).0 <= summed_diff
}

pub struct IdentityVJP;

impl VecJacProduct<Array> for IdentityVJP {
    fn vjp(&self, g: Array, _: &Node<Array>, _: usize) -> Array {
        g
    }
}

pub struct TestVar(pub Array);

impl Expression<Array> for TestVar {
    fn eval(&self, c: &mut Context) -> Node<Array> {
        Node::new(c, self.0.clone(), vec![], vec![], Box::new(IdentityVJP))
    }
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