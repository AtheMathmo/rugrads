use num::Float;

use std::marker::PhantomData;

use ::{Expression, VecJacProduct, Node};
use ::{Container, Context};

struct LinVJP<T: Float, F: Fn(T) -> T> {
    f: F,
    _marker: PhantomData<T>
}

impl<T: Float, F: Fn(T) -> T> LinVJP<T, F> {
    fn new(f: F) -> Self {
        LinVJP {
            f: f,
            _marker: PhantomData::<T>
        }
    }
}

impl<T: Float, F: Fn(T) -> T> VecJacProduct<T> for LinVJP<T, F> {
    fn vjp(&self, g: T, _:&Node<T>, x: &Node<T>, _: usize) -> T {
        g * (self.f)(x.value)
    }
}

/// Sine operator
pub struct Sin<T: Float, X: Expression<T>> {
    x: X,
    _marker: PhantomData<T>
}

impl<T: Float, X: Expression<T>> Sin<T, X> {
    fn new(x: X) -> Self {
        Sin {
            x: x,
            _marker: PhantomData::<T>
        }
    }
}

impl<T: Float, X: Expression<T>> Expression<T> for Sin<T, X> {
    fn eval(&self, c: &mut Context<T>) -> Node<T> {
        let x_eval = self.x.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node {
            index: c.get_index(),
            value: T::sin(parents[0].value),
            parents: parents,
            progenitors: progenitors,
            _vjp: Box::new(LinVJP::new(T::cos)),
        }
    }
}

/// Sine function
pub fn sin<T, E>(x: Container<T, E>) -> Container<T, Sin<T, E>>
    where T: Float, E: Expression<T>
{
    Container::new(Sin::new(x.inner))
}

/// Cosine operator
pub struct Cos<T: Float, X: Expression<T>> {
    x: X,
    _marker: PhantomData<T>
}

impl<T: Float, X: Expression<T>> Cos<T, X> {
    fn new(x: X) -> Self {
        Cos {
            x: x,
            _marker: PhantomData::<T>
        }
    }
}

impl<T: Float, X: Expression<T>> Expression<T> for Cos<T, X> {
    fn eval(&self, c: &mut Context<T>) -> Node<T> {
        let x_eval = self.x.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node {
            index: c.get_index(),
            value: T::cos(parents[0].value),
            parents: parents,
            progenitors: progenitors,
            _vjp: Box::new(LinVJP::new(|x| -T::sin(x))),
        }
    }
}

/// Cosine function
pub fn cos<T, E>(x: Container<T, E>) -> Container<T, Cos<T, E>>
    where T:Float, E: Expression<T>
{
    Container::new(Cos::new(x.inner))
}

/// Exponential operator
pub struct Exp<T: Float, X: Expression<T>> {
    x: X,
    _marker: PhantomData<T>
}

impl<T: Float, X: Expression<T>> Exp<T, X> {
    fn new(x: X) -> Self {
        Exp {
            x: x,
            _marker: PhantomData::<T>
        }
    }
}

impl<T: Float, X: Expression<T>> Expression<T> for Exp<T, X> {
    fn eval(&self, c: &mut Context<T>) -> Node<T> {
        let x_eval = self.x.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node {
            index: c.get_index(),
            value: T::exp(parents[0].value),
            parents: parents,
            progenitors: progenitors,
            _vjp: Box::new(LinVJP::new(T::exp)),
        }
    }
}

/// Exponential function
pub fn exp<T, E>(x: Container<T, E>) -> Container<T, Exp<T, E>>
    where T:Float, E: Expression<T>
{
    Container::new(Exp::new(x.inner))
}

/// Natural Logarithm operator
pub struct Ln<T: Float, X: Expression<T>> {
    x: X,
    _marker: PhantomData<T>
}

impl<T: Float, X: Expression<T>> Ln<T, X> {
    fn new(x: X) -> Self {
        Ln {
            x: x,
            _marker: PhantomData::<T>
        }
    }
}

impl<T: Float, X: Expression<T>> Expression<T> for Ln<T, X> {
    fn eval(&self, c: &mut Context<T>) -> Node<T> {
        let x_eval = self.x.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node {
            index: c.get_index(),
            value: T::ln(parents[0].value),
            parents: parents,
            progenitors: progenitors,
            _vjp: Box::new(LinVJP::new(T::recip)),
        }
    }
}

/// Natural Logarithm function
pub fn ln<T, E>(x: Container<T, E>) -> Container<T, Ln<T, E>>
    where T:Float, E: Expression<T>
{
    Container::new(Ln::new(x.inner))
}

/// Power raising operator
pub struct Powf<T: Float, X: Expression<T>> {
    x: X,
    n: T
}

impl<T: Float, X: Expression<T>> Powf<T, X> {
    fn new(x: X, n: T) -> Self {
        Powf {
            x: x,
            n: n
        }
    }
}

impl<T: Float, X: Expression<T>> Expression<T> for Powf<T, X> {
    fn eval(&self, c: &mut Context<T>) -> Node<T> {
        let x_eval = self.x.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        let n = self.n;
        Node {
            index: c.get_index(),
            value: T::powf(parents[0].value, n),
            parents: parents,
            progenitors: progenitors,
            _vjp: Box::new(LinVJP::new(move |x| n * T::powf(x, n - T::one()))),
        }
    }
}

/// Natural Logarithm function
pub fn powf<T, E>(x: Container<T, E>, n: T) -> Container<T, Powf<T, E>>
    where T:Float, E: Expression<T>
{
    Container::new(Powf::new(x.inner, n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::functions::{Add, Mul};
    use ::{Context, Expression};

    use ::LeafVar;
    

    #[test]
    fn test_add() {
        let mut c = Context::new();
        let f = Add::new(LeafVar(1.0), LeafVar(1.0));
        let node = f.eval(&mut c);
        // Just a dummy node
        let x = &node.parents[0];
        assert!((node.value - 2.0).abs() < 1e-5);
        assert!((node.vjp(1.0, x, 0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mul() {
        let mut c = Context::new();
        let f = Mul::new(LeafVar(0.5), LeafVar(0.3));
        let node = f.eval(&mut c);
        // Just a dummy node
        let x = &node.parents[0];
        assert!((node.value - 0.5*0.3).abs() < 1e-5);
        // 2 * 0.3
        assert!((node.vjp(2.0, x, 0) - 0.6).abs() < 1e-5);
        // 3 * 0.5
        assert!((node.vjp(3.0, x, 1) - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_sin() {
        let mut c = Context::new();
        let f = Sin::new(LeafVar(0.5));
        let node = f.eval(&mut c);
        let x = &node.parents[0];
        assert!((node.value - f64::sin(0.5)).abs() < 1e-5);
        assert!((node.vjp(1.0, x, 0) - f64::cos(0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_cos() {
        let mut c = Context::new();
        let f = Cos::new(LeafVar(0.5));
        let node = f.eval(&mut c);
        let x = &node.parents[0];
        assert!((node.value - f64::cos(0.5)).abs() < 1e-5);
        assert!((node.vjp(1.0, x, 0) + f64::sin(0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_exp() {
        let mut c = Context::new();
        let f = Exp::new(LeafVar(0.5));
        let node = f.eval(&mut c);
        let x = &node.parents[0];
        assert!((node.value - f64::exp(0.5)).abs() < 1e-5);
        assert!((node.vjp(1.0, x, 0) - f64::exp(0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_ln() {
        let mut c = Context::new();
        let f = Ln::new(LeafVar(0.5));
        let node = f.eval(&mut c);
        let x = &node.parents[0];
        assert!((node.value - f64::ln(0.5)).abs() < 1e-5);
        assert!((node.vjp(1.0, x, 0) - f64::recip(0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_powf_integer() {
        let mut c = Context::new();
        let f = Powf::new(LeafVar(3.0), 2.0);
        let node = f.eval(&mut c);
        let x = &node.parents[0];
        assert!((node.value - f64::powf(3.0, 2.0)).abs() < 1e-5);
        assert!((node.vjp(1.0, x, 0) - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_powf_neg_non_int() {
        let mut c = Context::new();
        let f = Powf::new(LeafVar(3.0), -1.3);
        let node = f.eval(&mut c);
        let x = &node.parents[0];
        assert!((node.value - f64::powf(3.0, -1.3)).abs() < 1e-5);
        assert!((node.vjp(1.0, x, 0) + 1.3 * f64::powf(3.0, -2.3)).abs() < 1e-5);
    }
}
