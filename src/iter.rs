use super::Node;

use std::collections::HashMap;

fn relevant_parents(parents: &Vec<Node>, start_idx: usize) -> Vec<&Node> {
    parents.iter()
            .filter(|p| {
                p.progenitors.contains(&start_idx) || p.index == start_idx
            })
            .collect()
}

pub fn reverse_topology<'a>(end: &'a Node, start_idx: usize) -> RevTopology<'a> {
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

pub struct RevTopology<'a> {
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
