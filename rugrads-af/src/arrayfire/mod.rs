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
univariate_wrapper!(Sinh, libaf::sinh, libaf::cosh);
univariate_wrapper!(Cosh, libaf::cosh, libaf::sinh);

pub struct Tan<X: Expression<Array>>(X);

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
