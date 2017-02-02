extern crate num;

pub mod graph;
pub mod functions;

use graph::{Graph, Node, reverse_topology};
use std::collections::HashMap;
use std::rc::Rc;

pub fn grad(graph: Rc<Graph>, end: Rc<Node>, start: Rc<Node>) -> f64 {
    let mut node_in_grads = HashMap::new();
    node_in_grads.insert(end.index(), vec![1f64]);
    let mut cur_in_grad = 1f64;

    for node in reverse_topology(graph.clone(), end.index(), start.index()) {
        if !node_in_grads.contains_key(&node.index()) {
            continue;
        } else {
            cur_in_grad = node_in_grads[&node.index()].iter().sum();
            for (argnum, p_idx) in node.parents().into_iter().enumerate() {
                let in_grad = node.vjp(cur_in_grad, graph.get(p_idx), argnum);
                let in_grads = node_in_grads.entry(p_idx).or_insert(vec![]);
                in_grads.push(in_grad);
            }
        }
    }

    return cur_in_grad
}

#[cfg(test)]
mod tests {
    use super::grad;
    use super::graph::{Graph, Node};
    use super::functions::*;
    use std::rc::Rc;

    #[test]
    fn test_basic_sum() {
        let graph = Rc::new(Graph::new());
        let x = Graph::create_variable(graph.clone(), 1.0);
        let f = Add::add(graph.clone(), x.index(), x.index());

        assert_eq!(f.eval(), 2f64);
        assert_eq!(grad(graph, f, x), 2f64);
    }

    #[test]
    fn test_basic_sin() {
        let graph = Rc::new(Graph::new());
        let x = Graph::create_variable(graph.clone(), 1.0);
        let f = Sin::sin(graph.clone(), x.index());

        assert!((f.eval() - 0.84147098).abs() < 1e-5);
        assert!((grad(graph, f, x) - 0.540302305).abs() < 1e-5);
    }

    #[test]
    fn test_sum_x_sinx() {
        let graph = Rc::new(Graph::new());
        let x = Graph::create_variable(graph.clone(), 0.5);
        let f = Sin::sin(graph.clone(), x.index());
        let g = Add::add(graph.clone(), x.index(), f.index());

        assert!((g.eval() - 0.979425538f64).abs() < 1e-5);
        assert!((grad(graph, g, x) - 1.87758256).abs() < 1e-5);
    }

    #[test]
    fn test_multi_var() {
        let graph = Rc::new(Graph::new());
        let x = Graph::create_variable(graph.clone(), 0.5);
        let y = Graph::create_variable(graph.clone(), 1.0);

        let f = Mul::mul(graph.clone(), x.index(), y.index());

        assert!((f.eval() - 0.5) < 1e-5);
        assert!((grad(graph.clone(), f.clone(), x.clone()) - 1.0) < 1e-5);
        assert!((grad(graph, f, y) - 0.5) < 1e-5);
    }

    #[test]
    fn test_multi_var_complex() {
        // Create an Rc pointer to a graph.
        let graph = Rc::new(Graph::new());
        // Create new variables with values 0.5 and 0.3
        let x = Graph::create_variable(graph.clone(), 0.5);
        let y = Graph::create_variable(graph.clone(), 0.3);

        // Below we build: y * sin(x) + cos(y)
        // Take sin of x
        let f = Sin::sin(graph.clone(), x.index());
        // Multiply f with y
        let g = Mul::mul(graph.clone(), y.index(), f.index());
        // Take cos(y) and add it to g
        let h = Cos::cos(graph.clone(), y.index());
        let fin = Add::add(graph.clone(), h.index(), g.index());

        // Take gradient with respect to x - has value: 
        let grad_x = grad(graph.clone(), fin.clone(), x.clone());
        println!("{}", grad_x);

        let grad_y = grad(graph, fin, y);
        println!("{}", grad_y);
        assert!(false);
    }
}
