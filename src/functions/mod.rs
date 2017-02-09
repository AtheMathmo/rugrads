//! Functions module
//!
//! This module contains differentiable wrapper functions.

use std::marker::PhantomData;
use std::ops;

use num::Float;

use ::{Node, Context, Expression, VecJacProduct, IdentityVJP};

mod op_overrides;
mod float;

pub use self::float::{sin, cos, exp, ln, powf};

/// Addition operation
pub struct Add<T, X, Y>
    where for<'a, 'b> &'a T: ops::Add<&'b T, Output=T>,
            X: Expression<T>,
            Y: Expression<T>
{
    x: X,
    y: Y,
    _marker: PhantomData<T>,
}

impl<T, X, Y> Add<T, X, Y>
    where for<'a, 'b> &'a T: ops::Add<&'b T, Output=T>,
            X: Expression<T>,
            Y: Expression<T>
{
    fn new(x: X, y: Y) -> Self {
        Add {
            x: x,
            y: y,
            _marker: PhantomData::<T>
        }
    }
}

impl<T, X, Y> Expression<T> for Add<T, X, Y>
    where for<'a, 'b> &'a T: ops::Add<&'b T, Output=T>,
            X: Expression<T>,
            Y: Expression<T>
{
    fn eval(&self, c: &mut Context<T>) -> Node<T> {
        let x_eval = self.x.eval(c);
        let y_eval = self.y.eval(c);

        let parents = vec![x_eval, y_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node {
            index: c.get_index(),
            value: &parents[0].value + &parents[1].value,
            parents: parents,
            progenitors: progenitors,
            _vjp: Box::new(IdentityVJP)
        }
    }
}

/// Multiplication operation
pub struct Mul<T, X, Y>
    where T: Float,
            X: Expression<T>,
            Y: Expression<T>
{
    x: X,
    y: Y,
    _marker: PhantomData<T>,
}

struct MulVJP<T>(T, T)
    where T: Float;

impl<T> VecJacProduct<T> for MulVJP<T>
    where T: Float
{
    fn vjp(&self, g: T, _: &Node<T>, argnum: usize) -> T {
        match argnum {
            0 => g * self.1,
            1 => g * self.0,
            _ => panic!("Invalid argnum fed to Mul VJP"),
        }
    }
}

impl<T, X, Y> Mul<T, X, Y>
    where T: Float,
            X: Expression<T>,
            Y: Expression<T>
{
    fn new(x: X, y: Y) -> Self {
        Mul {
            x: x,
            y: y,
            _marker: PhantomData::<T>
        }
    }
}

impl<T, X, Y> Expression<T> for Mul<T, X, Y>
    where T: Float,
            X: Expression<T>,
            Y: Expression<T>
{
    fn eval(&self, c: &mut Context<T>) -> Node<T> {
        let x_eval = self.x.eval(c);
        let y_eval = self.y.eval(c);

        let parents = vec![x_eval, y_eval];
        let progenitors = Node::get_progenitors(&parents);

        let v1 = parents[0].value;
        let v2 = parents[1].value;

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
pub struct Div<T, X, Y>
    where T: Float,
            X: Expression<T>,
            Y: Expression<T>
{
    x: X,
    y: Y,
    _marker: PhantomData<T>
}

impl<T, X, Y> Div<T, X, Y>
    where T: Float,
            X: Expression<T>,
            Y: Expression<T>
{
    fn new(x: X, y: Y) -> Self {
        Div {
            x: x,
            y: y,
            _marker: PhantomData::<T>
        }
    }
}

struct DivVJP<T: Float>(T, T);

impl<T: Float> VecJacProduct<T> for DivVJP<T> {
    fn vjp(&self, g: T, _: &Node<T>, argnum: usize) -> T {
        match argnum {
            0 => g / self.1,
            1 => - g * self.0 / (self.1 * self.1),
            _ => panic!("Invalid argnum fed to Div VJP"),
        }
    }
}

impl<T, X, Y> Expression<T> for Div<T, X, Y>
    where T: Float,
            X: Expression<T>,
            Y: Expression<T>
{
    fn eval(&self, c: &mut Context<T>) -> Node<T> {
        let x_eval = self.x.eval(c);
        let y_eval = self.y.eval(c);

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

/// Subtraction operation
pub struct Sub<T, X, Y>
    where for<'a, 'b> &'a T: ops::Sub<&'b T, Output=T>,
            T: ops::Neg<Output=T>,
            X: Expression<T>,
            Y: Expression<T>
{
    x: X,
    y: Y,
    _marker: PhantomData<T>,
}

impl<T, X, Y> Sub<T, X, Y>
    where for<'a, 'b> &'a T: ops::Sub<&'b T, Output=T>,
            T: ops::Neg<Output=T>,
            X: Expression<T>,
            Y: Expression<T>
{
    fn new(x: X, y: Y) -> Self {
        Sub {
            x: x,
            y: y,
            _marker: PhantomData::<T>
        }
    }
}

impl<T, X, Y> Expression<T> for Sub<T, X, Y>
    where for<'a, 'b> &'a T: ops::Sub<&'b T, Output=T>,
            T: ops::Neg<Output=T>,
            X: Expression<T>,
            Y: Expression<T>
{
    fn eval(&self, c: &mut Context<T>) -> Node<T> {
        let x_eval = self.x.eval(c);
        let y_eval = self.y.eval(c);

        let parents = vec![x_eval, y_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node {
            index: c.get_index(),
            value: &parents[0].value - &parents[1].value,
            parents: parents,
            progenitors: progenitors,
            _vjp: Box::new(SubVJP(PhantomData::<T>))
        }
    }
}

struct SubVJP<T: ops::Neg<Output=T>>(PhantomData<T>);

impl<T: ops::Neg<Output=T>> VecJacProduct<T> for SubVJP<T> {
    fn vjp(&self, g: T, _: &Node<T>, argnum: usize) -> T {
        match argnum {
            0 => g,
            1 => -g,
            _ => panic!("Invalid argnum fed to Sub VJP"),
        }
    }
}


/// Negative unary operator
pub struct Neg<T, X>(X, PhantomData<T>)
    where T: Clone + ops::Neg<Output=T>,
          X: Expression<T>;

struct NegVJP<T: ops::Neg<Output=T>>(PhantomData<T>);

impl<T: ops::Neg<Output=T>> VecJacProduct<T> for NegVJP<T> {
    fn vjp(&self, g: T, _: &Node<T>, _: usize) -> T {
        -g
    }
}

impl<T, X> Expression<T> for Neg<T, X>
    where T: Clone + ops::Neg<Output=T>,
          X: Expression<T>
{
    fn eval(&self, c: &mut Context<T>) -> Node<T> {
        let x_eval = self.0.eval(c);

        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node {
            index: c.get_index(),
            value: -parents[0].value.clone(),
            parents: parents,
            progenitors: progenitors,
            _vjp: Box::new(NegVJP(PhantomData::<T>))
        }
    }
}
