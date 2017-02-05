//! Functions module
//!
//! This module contains differentiable wrapper functions.

mod op_overrides;
mod ext;

use ::{Node, Context, Expression, VecJacProduct, IdentityVJP};

pub use self::ext::{sin, cos, exp, ln, powf};

/// Addition operation
pub struct Add<X: Expression, Y: Expression>(X, Y);

impl<X: Expression, Y: Expression> Expression for Add<X, Y> {
    fn eval(&self, c: &mut Context) -> Node {
        let x_eval = self.0.eval(c);
        let y_eval = self.1.eval(c);

        let parents = vec![x_eval, y_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node {
            index: c.get_index(),
            value: parents[0].value + parents[1].value,
            parents: parents,
            progenitors: progenitors,
            _vjp: Box::new(IdentityVJP)
        }
    }
}

/// Multiplication operation
pub struct Mul<X: Expression, Y: Expression>(X, Y);

struct MulVJP(f64, f64);

impl VecJacProduct for MulVJP {
    fn vjp(&self, g: f64, _: &Node, argnum: usize) -> f64 {
        match argnum {
            0 => g * self.1,
            1 => g * self.0,
            _ => panic!("Invalid argnum fed to Mul VJP"),
        }
    }
}

impl<X: Expression, Y: Expression> Expression for Mul<X, Y> {
    fn eval(&self, c: &mut Context) -> Node {
        let x_eval = self.0.eval(c);
        let y_eval = self.1.eval(c);

        let parents = vec![x_eval, y_eval];
        let progenitors = Node::get_progenitors(&parents);
        let (v1, v2) = (parents[0].value, parents[1].value);
        Node {
            index: c.get_index(),
            value: v1 * v2,
            parents: parents,
            progenitors: progenitors,
            _vjp: Box::new(MulVJP(v1, v2))
        }
    }
}

/// Division operation
pub struct Div<X: Expression, Y: Expression>(X, Y);

struct DivVJP(f64, f64);

impl VecJacProduct for DivVJP {
    fn vjp(&self, g: f64, _: &Node, argnum: usize) -> f64 {
        match argnum {
            0 => g / self.1,
            1 => - g * self.0 / (self.1 * self.1),
            _ => panic!("Invalid argnum fed to Mul VJP"),
        }
    }
}

impl<X: Expression, Y: Expression> Expression for Div<X, Y> {
    fn eval(&self, c: &mut Context) -> Node {
        let x_eval = self.0.eval(c);
        let y_eval = self.1.eval(c);

        let parents = vec![x_eval, y_eval];
        let progenitors = Node::get_progenitors(&parents);
        let (v1, v2) = (parents[0].value, parents[1].value);
        Node {
            index: c.get_index(),
            value: v1 / v2,
            parents: parents,
            progenitors: progenitors,
            _vjp: Box::new(DivVJP(v1, v2))
        }
    }
}

/// Subtraction operator
pub struct Sub<X: Expression, Y: Expression>(X, Y);

struct SubVJP;

impl VecJacProduct for SubVJP {
    fn vjp(&self, g: f64, _: &Node, argnum: usize) -> f64 {
        match argnum {
            0 => g,
            1 => -g,
            _ => panic!("Invalid argnum fed to Mul VJP"),
        }
    }
}

impl<X: Expression, Y: Expression> Expression for Sub<X, Y> {
    fn eval(&self, c: &mut Context) -> Node {
        let x_eval = self.0.eval(c);
        let y_eval = self.1.eval(c);

        let parents = vec![x_eval, y_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node {
            index: c.get_index(),
            value: parents[0].value - parents[1].value,
            parents: parents,
            progenitors: progenitors,
            _vjp: Box::new(SubVJP)
        }
    }
}

/// Negative unary operator
pub struct Neg<X: Expression>(X);

struct NegVJP;

impl VecJacProduct for NegVJP {
    fn vjp(&self, g: f64, _: &Node, _: usize) -> f64 {
        -g
    }
}

impl<X: Expression> Expression for Neg<X> {
    fn eval(&self, c: &mut Context) -> Node {
        let x_eval = self.0.eval(c);

        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node {
            index: c.get_index(),
            value: -parents[0].value,
            parents: parents,
            progenitors: progenitors,
            _vjp: Box::new(NegVJP)
        }
    }
}
