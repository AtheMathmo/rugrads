//! Rugrads
//!
//! Automatic differentiation in Rust
//!
//! Right now this library is a proof of concept with a messy API.
//!
//! # Example
//!
//! ```
//! use rugrads::*;
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


use std::collections::HashMap;

use std::f64;

pub struct Gradient<E: Expression> {
    expr: E,
    context: Context
}

impl<'a, E: Expression> Gradient<E> {
    pub fn of(expr: E, context: Context) -> Self {
        Gradient {
            expr: expr,
            context: context,
        }
    }

    pub fn context(&mut self) -> &mut Context {
        &mut self.context
    }

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

fn relevant_parents(parents: &Vec<Node>, start_idx: usize) -> Vec<&Node> {
    parents.iter()
            .filter(|p| {
                p.progenitors.contains(&start_idx) || p.index == start_idx
            })
            .collect()
}

fn reverse_topology<'a>(end: &'a Node, start_idx: usize) -> RevTopology<'a> {
    let mut child_counts = HashMap::new();
    {
        let mut stack = vec![end];

        while let Some(node) = stack.pop() {
            let cc = child_counts.entry(node.index).or_insert(0);
            *cc += 1;

            stack.extend(relevant_parents(&node.parents, start_idx));
        }
    }
    let mut data = Vec::<&'a Node>::new();
    data.push(end);

    RevTopology {
        start: start_idx,
        child_counts: child_counts,
        childless_nodes: data,
    }

}

struct RevTopology<'a> {
    start: usize,
    child_counts: HashMap<usize, usize>,
    childless_nodes: Vec<&'a Node>
}

impl<'a> Iterator for RevTopology<'a> {
    type Item = &'a Node;

    fn next(&mut self) -> Option<&'a Node> {
        if let Some(node) = self.childless_nodes.pop() {
            for p in relevant_parents(&node.parents, self.start) {
                let mut cc = self.child_counts.get_mut(&p.index)
                                            .expect("All child counts should be present");
                if *cc == 1 {
                    self.childless_nodes.push(p);
                } else {
                    *cc -= 1;
                }
            }
            Some(node)

        } else {
            None
        }
    }
}


pub trait Expression {
    fn eval(&self, c: &mut Context) -> Node;
}

trait VecJacProduct {
    fn vjp(&self, g: f64, node: &Node, argnum: usize) -> f64;
}

pub struct Context {
    vars: Vec<f64>,
    node_count: usize,
}

impl Context {
    pub fn create_variable(&mut self, value: f64) -> Variable {
        let var_idx = self.vars.len();
        self.vars.push(value);
        Variable(var_idx)
    }

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

    pub fn set_variable_value(&mut self, var: Variable, value: f64) {
        self.vars[var.0] = value;
    }
}

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

#[derive(Clone, Copy)]
pub struct Variable(usize);

pub struct IdentityVJP;

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

pub struct Add<X: Expression, Y: Expression>(pub X, pub Y);

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

pub struct Mul<X: Expression, Y: Expression>(pub X, pub Y);

pub struct MulVJP(f64, f64);

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

pub struct LinVJP<F: Fn(f64) -> f64>(F);

impl<F: Fn(f64) -> f64> VecJacProduct for LinVJP<F> {
    fn vjp(&self, g: f64, x: &Node, _: usize) -> f64 {
        g * self.0(x.value)
    }
}

pub struct Sin<X: Expression>(pub X);

impl<X: Expression> Expression for Sin<X> {
    fn eval(&self, c: &mut Context) -> Node {
        let x_eval = self.0.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node {
            index: c.get_index(),
            value: f64::sin(parents[0].value),
            parents: parents,
            progenitors: progenitors,
            _vjp: Box::new(LinVJP(f64::cos)),
        }
    }
}

pub struct Cos<X: Expression>(pub X);

impl<X: Expression> Expression for Cos<X> {
    fn eval(&self, c: &mut Context) -> Node {
        let x_eval = self.0.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node {
            index: c.get_index(),
            value: f64::cos(parents[0].value),
            parents: parents,
            progenitors: progenitors,
            _vjp: Box::new(LinVJP(|x| -f64::sin(x))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_multi_var_complex() {
        // First we define our context and function variables
        let mut context = Context::new();
        let x = context.create_variable(0.5);
        let y = context.create_variable(0.3);

        // Below we build: y * sin(x) + cos(y)
        // Take sin of x
        let f = Sin(x);

        // Multiply f with y
        let g = Mul(y, f);

        // Take cos(y) and add it to g
        let h = Cos(y);
        let fin = Add(h, g);

        let mut grad = Gradient::of(fin, context);

        // Take gradient with respect to x - has value: 
        grad.grad(x);
        grad.grad(y);
    }
}
