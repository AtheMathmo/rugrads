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
univar_wrapper_func!(sigmoid, Sigmoid, "Sum");
univar_wrapper_func!(sum_all, SumAll, "Sum");

/// Takes the elementwise Power Raising of an Array
pub fn pow<E: Expression<Array>>(input: Container<E>, n: f64) -> Container<Pow<E>> {
    Container::new(Pow(input.into_inner(), n))
}

/// Computes the norm of an Array
///
/// Currently only implements the Frobenius norm (`NormType::VECTOR_2`). Other norms
/// will cause panics when computing gradients.
pub fn norm<E: Expression<Array>>(input: Container<E>, ntype: ::NormType, p: f64, q: f64) -> Container<Norm<E>> {
    Container::new(Norm(input.into_inner(), ntype, p, q))
}

/// Computes the dot product of two arrays
///
/// Currently this function can only take a vector array
/// on the rhs. You can also not differentiate with respect
/// to a matrix used on the left.
pub fn dot<E1, E2>(lhs: Container<E1>, rhs: Container<E2>, _: ::MatProp, _: ::MatProp) -> Container<Dot<E1, E2>>
    where E1: Expression<Array>, E2: Expression<Array>
{
    Container::new(Dot(lhs.into_inner(), rhs.into_inner()))
}

/// Computes the elementwise multiplication of two arrays
pub fn mul<E1, E2>(lhs: Container<E1>, rhs: Container<E2>, _: bool) -> Container<AFMul<E1, E2>>
    where E1: Expression<Array>, E2: Expression<Array>
{
    Container::new(AFMul(lhs.into_inner(), rhs.into_inner()))
}

/// Computes the matrix product of two arrays
///
/// Currently this function only supports Matrix-Vector products
/// when the derivative is taken with respect to the vector.
pub fn matmul<E1, E2>(lhs: Container<E1>, rhs: Container<E2>, _: ::MatProp, _: ::MatProp) -> Container<MatMul<E1, E2>>
    where E1: Expression<Array>, E2: Expression<Array>
{
    Container::new(MatMul(lhs.into_inner(), rhs.into_inner()))
}
