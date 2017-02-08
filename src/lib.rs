//! Rugrads
//!
//! Automatic differentiation in Rust
//!
//! Right now this library is a proof of concept with a messy API.
//!
//! # Example
//!
//! ```
//! use rugrads::{Context, Gradient};
//! use rugrads::functions::*;
//!
//! // First we define our context and function variables
//! let mut context = Context::new();
//! let x = context.create_variable(0.5);
//! let y = context.create_variable(0.3);
//!
//! // Below we build: y * sin(x) + cos(y)
//! let f = y * sin(x) + cos(y);
//!
//! let mut grad = Gradient::of(f, context);
//!
//! // Take gradient with respect to x - has value: 
//! grad.grad(x);
//! grad.grad(y);
//!
//! // We can also change the initial seed values and recompute:
//! grad.context().set_variable_value(x, 0.8);
//! grad.grad(x);
//! ``` 

#![deny(missing_docs)]

extern crate num;

pub mod functions;
mod iter;

use std::collections::HashMap;
use std::marker::PhantomData;
use std::f64;

use iter::reverse_topology;

/// Container which wraps an expression
///
/// This container exists to bypass Rust's
/// 'Orphan Rules'. By using the container we can
/// overload arithmetic operations to give a nicer
/// user experience. 
#[derive(Clone, Copy)]
pub struct Container<T, E: Expression<T>> {
    inner: E,
    _marker: PhantomData<T>,
}

impl<T, E: Expression<T>> Expression<T> for Container<T, E> {
    fn eval(&self, c: &mut Context<T>) -> Node<T> {
        self.inner.eval(c)
    }
}

impl<T, E: Expression<T>> Container<T, E> {
    fn new(e: E) -> Self {
        Container {
            inner: e,
            _marker: PhantomData::<T>,
        }
    }
}

/// The Gradient of an Expression
///
/// This struct can be used to evaluate the gradient of
/// a given expression.
pub struct Gradient<T, E: Expression<T>> {
    expr: E,
    context: Context<T>
}

impl<T, E: Expression<T>> Gradient<T, E> {
    /// Take the gradient of an expression in some context
    ///
    /// # Examples
    ///
    /// ```
    /// use rugrads::{Gradient, Context};
    ///
    /// let mut context = Context::new();
    /// let x = context.create_variable(2.0);
    /// let f = x + x;
    ///
    /// let grad = Gradient::of(f, context);
    /// ```
    pub fn of(expr: Container<T, E>, context: Context<T>) -> Self {
        Gradient {
            expr: expr.inner,
            context: context,
        }
    }

    /// Gets the context for this gradient
    /// 
    /// You can use the context to set variable values.
    pub fn context(&mut self) -> &mut Context<T> {
        &mut self.context
    }
}

impl<E: Expression<f64>> Gradient<f64, E> {
    /// Compute the gradient with respect to the given
    /// `Variable`.
    pub fn grad(&mut self, wrt: Container<f64, Variable>) -> f64 {
        // Reset the context
        self.context.node_count = 0;

        // Forward prop
        let end = self.expr.eval(&mut self.context);

        // Backward prop
        let mut node_in_grads = HashMap::new();
        node_in_grads.insert(end.index, vec![1f64]);
        let mut cur_in_grad = 1f64;

        for node in reverse_topology(&end, wrt.inner.0) {
            if !node_in_grads.contains_key(&node.index) {
                continue;
            } else {
                cur_in_grad = node_in_grads[&node.index].iter().sum();
                for (argnum, p_node) in node.parents.iter().enumerate() {
                    let in_grad = node.vjp(cur_in_grad, p_node, argnum);
                    node_in_grads.entry(p_node.index).or_insert(vec![]).push(in_grad);
                }
            }
        }

        return cur_in_grad
    }
}

/// An expression which can be evaluated
pub trait Expression<T> {
    /// Evaluate the expression in the given context
    fn eval(&self, c: &mut Context<T>) -> Node<T>;
}

trait VecJacProduct<T> {
    fn vjp(&self, g: T, node: &Node<T>, argnum: usize) -> T;
}

/// The context for a computational expression
///
/// The `Context` stores the variable values which are used in
/// an expression.
///
/// The user is responsible for managing `Variable`s
/// within a context. This means that the user should ensure
/// that they do not mix up variables between different contexts.
pub struct Context<T> {
    vars: Vec<T>,
    node_count: usize,
}

impl<T> Context<T> {
    /// Create a new `Context`
    pub fn new() -> Context<T> {
        Context {
            vars: vec![],
            node_count: 0,
        }
    }

    fn get_index(&mut self) -> usize {
        let idx = self.vars.len() + self.node_count;
        self.node_count += 1;
        idx
    }
}
impl<T: Clone> Context<T> {
    /// Create a new `Variable` with the given value
    ///
    /// # Examples
    ///
    /// ```
    /// use rugrads::Context;
    ///
    /// let mut c = Context::new();
    /// let x = c.create_variable(2.5);
    /// ```
    pub fn create_variable(&mut self, value: T) -> Container<T, Variable> {
        let var_idx = self.vars.len();
        self.vars.push(value);
        Container::new(Variable(var_idx))
    }

    fn get_variable_value(&self, var: Variable) -> T {
        self.vars[var.0].clone()
    }

    /// Set the given variable value
    ///
    /// # Examples
    ///
    /// ```
    /// use rugrads::Context;
    ///
    /// // Create a new variable with value 2.5
    /// let mut c = Context::new();
    /// let x = c.create_variable(2.5);
    ///
    /// // Actually, lets make that 3.0!
    /// c.set_variable_value(x, 3.0);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if the `Variable` does not belong
    /// to this context.
    ///
    /// More accurately - if the `Variable` has an index which is too
    /// large for this context.
    pub fn set_variable_value(&mut self, var: Container<T, Variable>, value: T) {
        self.vars[var.inner.0] = value;
    }
}

/// A node in a computational graph
///
/// When we evaluate an expression we create
/// a graph made up of nodes.
pub struct Node<'a, T> {
    index: usize,
    value: T,
    parents: Vec<Node<'a, T>>,
    progenitors: Vec<usize>,
    _vjp: Box<VecJacProduct<T> + 'a>,
}

impl<'a, T> VecJacProduct<T> for Node<'a, T> {
    fn vjp(&self, g: T, node: &Node<T>, argnum: usize) -> T {
        self._vjp.vjp(g, node, argnum)
    }
}

impl<'a, T> Node<'a, T> {
    fn get_progenitors(parents: &Vec<Node<'a, T>>) -> Vec<usize> {
        let mut progenitors = parents.iter().map(|p| p.index).collect::<Vec<usize>>();
        for parent in parents.iter() {
            for prog in parent.progenitors.iter() {
                if !progenitors.contains(prog) {
                    progenitors.push(*prog);
                }
            }
        }
        progenitors
    }
}

/// A Variable
///
/// Each variable specifies an index into a Context.
#[derive(Clone, Copy)]
pub struct Variable(usize);

struct IdentityVJP;

impl<T> VecJacProduct<T> for IdentityVJP {
    fn vjp(&self, g: T, _: &Node<T>, _: usize) -> T {
        g
    }
}

impl<T: Clone> Expression<T> for Variable {
    fn eval(&self, c: &mut Context<T>) -> Node<T> {
        Node {
            index: self.0,
            value: c.get_variable_value(*self),
            parents: vec![],
            progenitors: vec![],
            _vjp: Box::new(IdentityVJP)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::functions::*;

    use num::Float;

    #[test]
    fn test_basic_sum() {
        let mut context = Context::new();
        let x = context.create_variable(1.0);
        let f = x + x;

        assert_eq!(f.eval(&mut context).value, 2f64);

        let mut grad = Gradient::of(f, context);
        assert_eq!(grad.grad(x), 2f64);
    }

    #[test]
    fn test_basic_sin() {
        let mut context = Context::new();
        let x = context.create_variable(1.0);
        let f = sin(x);

        assert!((f.eval(&mut context).value - 0.84147098).abs() < 1e-5);

        let mut grad = Gradient::of(f, context);
        assert!((grad.grad(x) - 0.540302305).abs() < 1e-5);

        grad.context().set_variable_value(x, 2.0);
        assert!((grad.grad(x) + 0.416146836).abs() < 1e-5);
    }

    #[test]
    fn test_sum_x_sinx() {
        let mut context = Context::new();
        let x = context.create_variable(0.5);
        let f = x + sin(x);
        
        assert!((f.eval(&mut context).value - 0.979425538f64).abs() < 1e-5);
        
        let mut grad = Gradient::of(f, context);
        assert!((grad.grad(x) - 1.87758256).abs() < 1e-5);
    }

    #[test]
    fn test_multi_var() {
        let mut context = Context::new();
        let x = context.create_variable(0.5);
        let y = context.create_variable(1.0);
        let f = x * y;

        assert!((f.eval(&mut context).value - 0.5).abs() < 1e-5);
        
        let mut grad = Gradient::of(f, context);
        assert!((grad.grad(x) - 1.0) < 1e-5);
        assert!((grad.grad(y) - 0.5) < 1e-5);
    }
}
