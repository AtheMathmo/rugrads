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
mod utils;

use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Add;

use iter::reverse_topology;

/// Container which wraps an expression
///
/// This container exists to bypass Rust's
/// 'Orphan Rules'. By using the container we can
/// overload arithmetic operations to give a nicer
/// user experience. 
pub struct Container<T, E: Expression<T>> {
    inner: E,
    _marker: PhantomData<T>,
}

impl<T, E: Clone + Expression<T>> Clone for Container<T, E> {
    fn clone(&self) -> Self {
        Container {
            inner: self.inner.clone(),
            _marker: PhantomData
        }
    }
}

impl<T, E: Copy + Expression<T>> Copy for Container<T,E> {}

impl<T, E: Expression<T>> Expression<T> for Container<T, E> {
    fn eval(&self, c: &mut Context<T>) -> Node<T> {
        self.inner.eval(c)
    }
}

impl<T, E: Expression<T>> Container<T, E> {
    /// Creates a new container which wraps the expression
    pub fn new(e: E) -> Self {
        Container {
            inner: e,
            _marker: PhantomData::<T>,
        }
    }

    /// Returns a pointer to the inner value
    pub fn inner(&self) -> &E {
        &self.inner
    }

    /// Consumes the container and returns the inner expression
    pub fn into_inner(self) -> E {
        self.inner
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

impl<T: Clone + Add<Output=T>, E: Expression<T>> Gradient<T, E> {
    /// Back propagates the gradient with some starting seed.
    ///
    /// This seed should always be set to 1. 
    pub fn backprop(&mut self, wrt: Container<T, Variable>, seed: T) -> T {
        // Reset the context
        self.context.node_count = 0;

        // Forward prop
        let end = self.expr.eval(&mut self.context);

        // Backward prop
        let mut node_in_grads = HashMap::new();
        node_in_grads.insert(end.index, vec![seed]);

        let mut cur_in_grad = utils::assigning_sum(&node_in_grads[&end.index]);
        for node in reverse_topology(&end, wrt.inner.0) {
            if !node_in_grads.contains_key(&node.index) {
                // This ensures we don't try to sum an empty in_grads vec
                continue;
            } else {
                cur_in_grad = utils::assigning_sum(&node_in_grads[&node.index]);
                for (argnum, p_node) in node.parents.iter().enumerate() {
                    let in_grad = node.vjp(cur_in_grad.clone(), p_node, argnum);
                    node_in_grads.entry(p_node.index).or_insert(vec![]).push(in_grad);
                }
            }
        }

        return cur_in_grad
    }

    /// Returns a mutable reference to a variable value in this gradient
    pub fn get_mut(&mut self, var: Container<T, Variable>) -> &mut T {
        &mut self.context.vars[var.inner.0]
    }

    /// Returns a mutable reference to a variable value in this gradient
    pub fn get(&self, var: Container<T, Variable>) -> &T {
        &self.context.vars[var.inner.0]
    }
}

impl<T: num::Float, E: Expression<T>> Gradient<T, E> {
    /// Compute the gradient with respect to the given
    /// `Variable`.
    pub fn grad(&mut self, wrt: Container<T, Variable>) -> T {
        self.backprop(wrt, T::one())
    }
}

/// An expression which can be evaluated
pub trait Expression<T> {
    /// Evaluate the expression in the given context
    fn eval(&self, c: &mut Context<T>) -> Node<T>;
}

/// The Vector-Jacobian product of gradients
pub trait VecJacProduct<T> {
    /// The vjp function which determines how the gradient is back propagated
    fn vjp(&self, g: T, node: &Node<T>, parent: &Node<T>, argnum: usize) -> T;
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

    fn get_variable_value(&self, var: &Variable) -> T {
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


impl<'a, T> Node<'a, T> {
    /// Vector-Jacobian Product wrapper function
    pub fn vjp(&self, g: T, parent: &Node<T>, argnum: usize) -> T {
        self._vjp.vjp(g, &self, parent, argnum)
    }

    /// Returns a new node in the given context
    pub fn new(c: &mut Context<T>, value: T,
                parents: Vec<Node<'a, T>>, progenitors: Vec<usize>,
                vjp: Box<VecJacProduct<T> + 'a>) -> Self {
        Node {
            index: c.get_index(),
            value: value,
            parents: parents,
            progenitors: progenitors,
            _vjp: vjp
        }
    }

    /// Gets the progenitors of all parents
    pub fn get_progenitors(parents: &Vec<Node<'a, T>>) -> Vec<usize> {
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

    /// Returns a reference to the underlying node value.
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Returns a reference to the nodes parents
    pub fn parents(&self) -> &Vec<Node<T>> {
        &self.parents
    }
}

/// A Variable
///
/// Each variable specifies an index into a Context.
#[derive(Clone, Copy)]
pub struct Variable(usize);

impl Variable {
    /// Returns a reference to the underlying `Variable` value
    /// in this context.
    pub fn value<'a, T: 'a>(&self, c: &'a Context<T>) -> &'a T {
        &c.vars[self.0]
    }
}

struct IdentityVJP;

impl<T> VecJacProduct<T> for IdentityVJP {
    fn vjp(&self, g: T, _:&Node<T>, _: &Node<T>, _: usize) -> T {
        g
    }
}

impl<T: Clone> Expression<T> for Variable {
    fn eval(&self, c: &mut Context<T>) -> Node<T> {
        Node {
            index: self.0,
            value: c.get_variable_value(self),
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
