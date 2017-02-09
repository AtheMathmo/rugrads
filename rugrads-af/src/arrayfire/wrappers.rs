use libaf::Array;

use rugrads::Expression;
use ::Container;

use super::*;

/// Takes the elementwise Sine of an Array
pub fn sin<E: Expression<Array>>(input: Container<E>) -> Container<Sin<E>> {
    Container::new(Sin(input.into_inner()))
}

/// Takes the elementwise Cosine of an Array
pub fn cos<E: Expression<Array>>(input: Container<E>) -> Container<Cos<E>> {
    Container::new(Cos(input.into_inner()))
}

/// Takes the elementwise Exponent of an Array
pub fn exp<E: Expression<Array>>(input: Container<E>) -> Container<Exp<E>> {
    Container::new(Exp(input.into_inner()))
}

/// Takes the elementwise Natural Logarithm of an Array
pub fn log<E: Expression<Array>>(input: Container<E>) -> Container<Log<E>> {
    Container::new(Log(input.into_inner()))
}

/// Takes the elementwise Power Raising of an Array
pub fn pow<E: Expression<Array>>(input: Container<E>, i: i64) -> Container<Pow<E>> {
    Container::new(Pow(input.into_inner(), i))
}

/// Takes the elementwise Hyperbolic Sine of an Array
pub fn sinh<E: Expression<Array>>(input: Container<E>) -> Container<Sinh<E>> {
    Container::new(Sinh(input.into_inner()))
}

/// Takes the elementwise Hyperbolic Cosine of an Array
pub fn cosh<E: Expression<Array>>(input: Container<E>) -> Container<Cosh<E>> {
    Container::new(Cosh(input.into_inner()))
}
