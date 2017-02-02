use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::cell::RefCell;
use super::functions::Variable;

pub struct Graph {
    values: RefCell<Vec<Weak<Node>>>,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            values: RefCell::new(vec![])
        }
    }

    pub fn values(&self) -> &RefCell<Vec<Weak<Node>>> {
        &self.values
    }
    
    pub fn get(&self, node_idx: usize) -> Rc<Node> {
        self.values.borrow_mut()[node_idx].upgrade().unwrap()
    }

    pub fn create_variable(graph: Rc<Graph>, value: f64) -> Rc<Variable> {
        let new_index = graph.values.borrow().len();
        
        let var = Rc::new(Variable {
            graph: graph.clone(),
            index: new_index,
            value: value,
        });
        let node_ptr : Rc<Node> = var.clone();
        graph.values.borrow_mut().push(Rc::downgrade(&node_ptr));
        var
    }
}

pub trait Node {
    fn owner(&self) -> &Rc<Graph>;
    
    fn parents(&self) -> Vec<usize>;
    
    fn index(&self) -> usize;
    
    fn eval(&self) -> f64;

    fn vjp(&self, g: f64, node: Rc<Node>, argnum: usize) -> f64;
    
    fn progenitors(&self) -> Vec<usize> {
        let mut progenitors = self.parents();
        for parent in progenitors.clone() {
            for desc in self.owner().get(parent).progenitors() {
                if !progenitors.contains(&desc) {
                    progenitors.push(desc);
                }
            }
        }
        progenitors
    }
}

fn valid_parents(graph: &Rc<Graph>, child_idx: usize, base_idx: usize) -> Vec<usize> {
    graph.get(child_idx).parents()
                        .iter()
                        .filter(|p_idx| {
                            graph.get(**p_idx).progenitors().contains(&base_idx) ||
                            **p_idx == base_idx
                        })
                        .cloned()
                        .collect()
}

pub fn reverse_topology(graph: Rc<Graph>, end: usize, start: usize) -> RevTopology {
    let mut child_counts = HashMap::new();
    let mut stack = vec![end];

    while let Some(node_idx) = stack.pop() {
        let cc = child_counts.entry(node_idx).or_insert(0);
        *cc += 1;

        stack.extend_from_slice(&valid_parents(&graph, node_idx, start));
    }

    RevTopology {
        graph: graph,
        start: start,
        child_counts: child_counts,
        childless_nodes: vec![end],
    }
}

pub struct RevTopology {
    graph: Rc<Graph>,
    start: usize,
    child_counts: HashMap<usize, usize>,
    childless_nodes: Vec<usize>
}

impl Iterator for RevTopology {
    type Item = Rc<Node>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node_idx) = self.childless_nodes.pop() {
            for p in valid_parents(&self.graph, node_idx, self.start) {
                let mut cc = self.child_counts.get_mut(&p).expect("All child counts should be present");
                if *cc == 1 {
                    self.childless_nodes.push(p);
                } else {
                    *cc -= 1;
                }
            }
            Some(self.graph.get(node_idx))             
        } else {
            None
        }
    }
}
