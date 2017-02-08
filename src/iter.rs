use super::Node;

use std::collections::HashMap;

fn relevant_parents<'a, 'b, T>(parents: &'b Vec<Node<'a, T>>, start_idx: usize) -> Vec<&'b Node<'a, T>> {
    parents.iter()
            .filter(|p| {
                p.progenitors.contains(&start_idx) || p.index == start_idx
            })
            .collect()
}

pub fn reverse_topology<'a, 'b, T: 'a>(end: &'a Node<'b, T>, start_idx: usize) -> RevTopology<'a, 'b, T> {
    let mut child_counts = HashMap::new();
    {
        let mut stack = vec![end];

        while let Some(node) = stack.pop() {
            let cc = child_counts.entry(node.index).or_insert(0);
            *cc += 1;

            stack.extend(relevant_parents(&node.parents, start_idx));
        }
    }

    RevTopology {
        start: start_idx,
        child_counts: child_counts,
        childless_nodes: vec![end],
    }
}

pub struct RevTopology<'a, 'b: 'a, T: 'a> {
    start: usize,
    child_counts: HashMap<usize, usize>,
    childless_nodes: Vec<&'a Node<'b, T>>
}

impl<'a, 'b, T: 'a> Iterator for RevTopology<'a, 'b, T> {
    type Item = &'a Node<'b, T>;

    fn next(&mut self) -> Option<&'a Node<'b, T>> {
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
