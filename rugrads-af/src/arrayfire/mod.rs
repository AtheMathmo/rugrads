use rugrads::{Node, VecJacProduct, Expression};

use libaf;
use libaf::Array;

use ::Context;

pub mod wrappers;

struct LinVJP<F>(F)
    where for<'a> F: Fn(&'a Array) -> Array;

impl<F> VecJacProduct<Array> for LinVJP<F>
    where for<'a> F: Fn(&'a Array) -> Array
{
    fn vjp(&self, g: Array, x: &Node<Array>, _: usize) -> Array {
        g * (self.0)(x.value())
    }
}

macro_rules! univariate_wrapper {
    ($name: ident, $af_func: expr, $vjp: expr) => {
pub struct $name<X: Expression<Array>>(X);

impl<X: Expression<Array>> Expression<Array> for $name<X> {
    fn eval(&self, c: &mut Context) -> Node<Array> {
        let x_eval = self.0.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node::new(c, $af_func(parents[0].value()),
                    parents, progenitors, Box::new(LinVJP($vjp)))
    }
}
    };
}

univariate_wrapper!(Sin, libaf::sin, libaf::cos);
univariate_wrapper!(Cos, libaf::cos, |x| -libaf::sin(x));
univariate_wrapper!(Tan, libaf::tan, |x| {
    let ones = libaf::constant(1f64, x.dims());
    let cos_x = libaf::cos(x);
    libaf::div(&ones, &libaf::mul(&cos_x, &cos_x, false), false)
});
univariate_wrapper!(Sinh, libaf::sinh, libaf::cosh);
univariate_wrapper!(Cosh, libaf::cosh, libaf::sinh);
univariate_wrapper!(Tanh, libaf::tanh, |x| {
    let ones = libaf::constant(1f64, x.dims());
    let cosh_x = libaf::cosh(x);
    libaf::div(&ones, &libaf::mul(&cosh_x, &cosh_x, false), false)
});
univariate_wrapper!(Arcsin, libaf::asin, |x| {
    let ones = libaf::constant(1f64, x.dims());
    let x_sq = libaf::sub(&ones, &libaf::pow(x, &2f64, false), false);
    libaf::div(&ones, &x_sq, false)
});
univariate_wrapper!(Arccos, libaf::acos, |x| {
    let ones = libaf::constant(1f64, x.dims());
    let x_sq = libaf::sub(&ones, &libaf::pow(x, &2f64, false), false);
    -libaf::div(&ones, &x_sq, false)
});
univariate_wrapper!(Arctan, libaf::atan, |x| {
    let ones = libaf::constant(1f64, x.dims());
    let x_sq = libaf::add(&ones, &libaf::pow(x, &2f64, false), false);
    libaf::div(&ones, &x_sq, false)
});
univariate_wrapper!(Exp, libaf::exp, libaf::exp);
univariate_wrapper!(Log, libaf::log, move |x| libaf::pow(x, &-1f64, false));

pub struct Pow<X: Expression<Array>>(X, f64);

impl<X: Expression<Array>> Expression<Array> for Pow<X> {
    fn eval(&self, c: &mut Context) -> Node<Array> {
        let x_eval = self.0.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node::new(c, libaf::pow(parents[0].value(), &self.1, false),
                    parents, progenitors, Box::new(LinVJP(move |x| libaf::pow(x, &(self.1 - 1f64), false) * self.1)))
    }
}
