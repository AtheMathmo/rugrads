use libaf::Array;

use rugrads::Expression;
use ::Container;

use super::*;

macro_rules! univar_wrapper_func {
    ($f_name: ident, $struct_name: ident, $doc: expr) => {
/// Takes the elementwise
#[doc]
/// of an Array
pub fn $f_name<E: Expression<Array>>(input: Container<E>) -> Container<$struct_name<E>> {
    Container::new($struct_name(input.into_inner()))
}
    };
}

univar_wrapper_func!(sin, Sin, "Sine");
univar_wrapper_func!(cos, Cos, "Cosine");
univar_wrapper_func!(tan, Tan, "Tangent");
univar_wrapper_func!(sinh, Sinh, "Hyperbolic Sine");
univar_wrapper_func!(cosh, Cosh, "Hyperbolic Cosine");
univar_wrapper_func!(tanh, Tanh, "Hyperbolic Tangent");
univar_wrapper_func!(asin, Arcsin, "Arcsin");
univar_wrapper_func!(acos, Arccos, "Arccos");
univar_wrapper_func!(atan, Arctan, "Arctan");
univar_wrapper_func!(exp, Exp, "Exponent");
univar_wrapper_func!(log, Log, "Natural Logarithm");

/// Takes the elementwise Power Raising of an Array
pub fn pow<E: Expression<Array>>(input: Container<E>, n: f64) -> Container<Pow<E>> {
    Container::new(Pow(input.into_inner(), n))
}
