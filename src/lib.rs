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
//! // Take sin of x
//! let f = Sin(x);
//!     
//! // Multiply f with y
//! let g = Mul(y, f);
//!
//! // Take cos(y) and add it to g
//! let h = Cos(y);
//! let fin = Add(h, g);
//!
//! let mut grad = Gradient::of(fin, context);
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

pub mod functions;
mod iter;

use std::collections::HashMap;
use std::f64;

use iter::reverse_topology;

/// The Gradient of an Expression
///
/// This struct can be used to evaluate the gradient of
/// a given expression.
pub struct Gradient<E: Expression> {
    expr: E,
    context: Context
}

impl<'a, E: Expression> Gradient<E> {
    /// Take the gradient of an expression in some context
    ///
    /// # Examples
    ///
    /// ```
    /// use rugrads::{Gradient, Context};
    /// use rugrads::functions::Add;
    ///
    /// let mut context = Context::new();
    /// let x = context.create_variable(2.0);
    /// let f = Add(x, x);
    ///
    /// let grad = Gradient::of(f, context);
    /// ```
    pub fn of(expr: E, context: Context) -> Self {
        Gradient {
            expr: expr,
            context: context,
        }
    }

    /// Gets the context for this gradient
    /// 
    /// You can use the context to set variable values.
    pub fn context(&mut self) -> &mut Context {
        &mut self.context
    }

    /// Compute the gradient with respect to the given
    /// `Variable`.
    pub fn grad(&mut self, wrt: Variable) -> f64 {
        // Reset the context
        self.context.node_count = 0;

        // Forward prop
        let end = self.expr.eval(&mut self.context);

        // Backward prop
        let mut node_in_grads = HashMap::new();
        node_in_grads.insert(end.index, vec![1f64]);
        let mut cur_in_grad = 1f64;

        for node in reverse_topology(&end, wrt.0) {
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
pub trait Expression {
    /// Evaluate the expression in the given context
    fn eval(&self, c: &mut Context) -> Node;
}

trait VecJacProduct {
    fn vjp(&self, g: f64, node: &Node, argnum: usize) -> f64;
}

/// The context for a computational expression
///
/// The `Context` stores the variable values which are used in
/// an expression.
///
/// The user is responsible for managing `Variable`s
/// within a context. This means that the user should ensure
/// that they do not mix up variables between different contexts.
pub struct Context {
    vars: Vec<f64>,
    node_count: usize,
}

impl Context {
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
    pub fn create_variable(&mut self, value: f64) -> Variable {
        let var_idx = self.vars.len();
        self.vars.push(value);
        Variable(var_idx)
    }

    /// Create a new `Context`
    pub fn new() -> Context {
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

    fn get_variable_value(&self, var: Variable) -> f64 {
        self.vars[var.0]
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
    pub fn set_variable_value(&mut self, var: Variable, value: f64) {
        self.vars[var.0] = value;
    }
}

/// A node in a computational graph
///
/// When we evaluate an expression we create
/// a graph made up of nodes.
pub struct Node {
    index: usize,
    value: f64,
    parents: Vec<Node>,
    progenitors: Vec<usize>,
    _vjp: Box<VecJacProduct>,
}

impl VecJacProduct for Node {
    fn vjp(&self, g: f64, node: &Node, argnum: usize) -> f64 {
        self._vjp.vjp(g, node, argnum)
    }
}

impl Node {
    fn get_progenitors(parents: &Vec<Node>) -> Vec<usize> {
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

impl VecJacProduct for IdentityVJP {
    fn vjp(&self, g: f64, _: &Node, _: usize) -> f64 {
        g
    }
}

impl Expression for Variable {
    fn eval(&self, c: &mut Context) -> Node {
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

    #[test]
    fn test_basic_sum() {
        let mut context = Context::new();
        let x = context.create_variable(1.0);
        let f = Add(x, x);

        assert_eq!(f.eval(&mut context).value, 2f64);

        let mut grad = Gradient::of(f, context);
        assert_eq!(grad.grad(x), 2f64);
    }

    #[test]
    fn test_basic_sin() {
        let mut context = Context::new();
        let x = context.create_variable(1.0);
        let f = Sin(x);

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
        let f = Sin(x);
        let g = Add(x, f);
        
        assert!((g.eval(&mut context).value - 0.979425538f64).abs() < 1e-5);
        
        let mut grad = Gradient::of(g, context);
        assert!((grad.grad(x) - 1.87758256).abs() < 1e-5);
    }

    #[test]
    fn test_multi_var() {
        let mut context = Context::new();
        let x = context.create_variable(0.5);
        let y = context.create_variable(1.0);
        let f = Mul(x, y);

        assert!((f.eval(&mut context).value - 0.5).abs() < 1e-5);
        
        let mut grad = Gradient::of(f, context);
        assert!((grad.grad(x) - 1.0) < 1e-5);
        assert!((grad.grad(y) - 0.5) < 1e-5);
    }
}
