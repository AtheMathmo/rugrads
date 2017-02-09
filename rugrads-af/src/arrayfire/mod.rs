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

pub struct Sin<X: Expression<Array>>(X);

impl<X: Expression<Array>> Expression<Array> for Sin<X> {
    fn eval(&self, c: &mut Context) -> Node<Array> {
        let x_eval = self.0.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node::new(c, libaf::sin(parents[0].value()),
                    parents, progenitors, Box::new(LinVJP(libaf::cos)))
    }
}

pub struct Cos<X: Expression<Array>>(X);

impl<X: Expression<Array>> Expression<Array> for Cos<X> {
    fn eval(&self, c: &mut Context) -> Node<Array> {
        let x_eval = self.0.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node::new(c, libaf::cos(parents[0].value()),
                    parents, progenitors, Box::new(LinVJP(|x| -libaf::sin(x))))
    }
}

pub struct Exp<X: Expression<Array>>(X);

impl<X: Expression<Array>> Expression<Array> for Exp<X> {
    fn eval(&self, c: &mut Context) -> Node<Array> {
        let x_eval = self.0.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node::new(c, libaf::exp(parents[0].value()),
                    parents, progenitors, Box::new(LinVJP(libaf::exp)))
    }
}

pub struct Log<X: Expression<Array>>(X);

impl<X: Expression<Array>> Expression<Array> for Log<X> {
    fn eval(&self, c: &mut Context) -> Node<Array> {
        let x_eval = self.0.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node::new(c, libaf::log(parents[0].value()),
                    parents, progenitors, Box::new(LinVJP(move |x| libaf::pow(x, &-1, false))))
    }
}

pub struct Pow<X: Expression<Array>>(X, i64);

impl<X: Expression<Array>> Expression<Array> for Pow<X> {
    fn eval(&self, c: &mut Context) -> Node<Array> {
        let x_eval = self.0.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node::new(c, libaf::pow(parents[0].value(), &self.1, false),
                    parents, progenitors, Box::new(LinVJP(move |x| libaf::pow(x, &(self.1 - 1), false) * self.1)))
    }
}

pub struct Sinh<X: Expression<Array>>(X);

impl<X: Expression<Array>> Expression<Array> for Sinh<X> {
    fn eval(&self, c: &mut Context) -> Node<Array> {
        let x_eval = self.0.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node::new(c, libaf::sinh(parents[0].value()),
                    parents, progenitors, Box::new(LinVJP(libaf::cosh)))
    }
}

pub struct Cosh<X: Expression<Array>>(X);

impl<X: Expression<Array>> Expression<Array> for Cosh<X> {
    fn eval(&self, c: &mut Context) -> Node<Array> {
        let x_eval = self.0.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node::new(c, libaf::cosh(parents[0].value()),
                    parents, progenitors, Box::new(LinVJP(libaf::sinh)))
    }
}
