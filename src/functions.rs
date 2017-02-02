use std::f64;
use graph::{Graph, Node};
use std::rc::Rc;

// Contains index of variable with respect to context
pub struct Variable {
    pub graph: Rc<Graph>,
    pub index: usize,
    pub value: f64
}

impl Node for Variable {
    fn owner(&self) -> &Rc<Graph> {
        &self.graph
    }
    
    fn index(&self) -> usize {
        self.index
    }
    
    fn eval(&self) -> f64 {
        self.value
    }

    fn vjp(&self, g: f64, _: Rc<Node>, _: usize) -> f64 {
        g
    }
    
    fn parents(&self) -> Vec<usize> {
        vec![]
    }
}

pub struct Add {
    graph: Rc<Graph>,
    index: usize,
    x: usize,
    y: usize
}

impl Add {
    pub fn add(graph: Rc<Graph>, x: usize, y: usize) -> Rc<Add> {
        let new_index = graph.values().borrow().len();
        assert!(x < new_index);
        assert!(y < new_index);
        
        let add = Rc::new(Add {
            graph: graph.clone(),
            index: new_index,
            x: x,
            y: y
        });
        let node_ptr : Rc<Node> = add.clone();
        graph.values().borrow_mut().push(Rc::downgrade(&node_ptr));
        add
    }
}

impl Node for Add {
    fn owner(&self) -> &Rc<Graph> {
        &self.graph
    }
    
    fn index(&self) -> usize {
        self.index
    }
    
    fn eval(&self) -> f64 {
        self.graph.get(self.x).eval() + self.graph.get(self.y).eval()
    }

    fn vjp(&self, g: f64, _: Rc<Node>, _: usize) -> f64 {
        g
    }
    
    fn parents(&self) -> Vec<usize> {
        vec![self.x, self.y]
    }
}

pub struct Mul {
    graph: Rc<Graph>,
    index: usize,
    x: usize,
    y: usize
}

impl Mul {
    pub fn mul(graph: Rc<Graph>, x: usize, y: usize) -> Rc<Mul> {
        let new_index = graph.values().borrow().len();
        assert!(x < new_index);
        assert!(y < new_index);
        
        let mul = Rc::new(Mul {
            graph: graph.clone(),
            index: new_index,
            x: x,
            y: y
        });
        let node_ptr : Rc<Node> = mul.clone();
        graph.values().borrow_mut().push(Rc::downgrade(&node_ptr));
        mul
    }
}

impl Node for Mul {
    fn owner(&self) -> &Rc<Graph> {
        &self.graph
    }
    
    fn index(&self) -> usize {
        self.index
    }
    
    fn eval(&self) -> f64 {
        self.graph.get(self.x).eval() * self.graph.get(self.y).eval()
    }

    fn vjp(&self, g: f64, _: Rc<Node>, argnum: usize) -> f64 {
        match argnum {
            0 => g * self.graph.get(self.y).eval(),
            1 => g * self.graph.get(self.x).eval(),
            _ => panic!("Mul vjp must have an argnum of 1 or 2.")
        }
    }
    
    fn parents(&self) -> Vec<usize> {
        vec![self.x, self.y]
    }
}

pub struct Sin {
    graph: Rc<Graph>,
    index: usize,
    x: usize
}

impl Sin {
    pub fn sin(graph: Rc<Graph>, x: usize) -> Rc<Sin> {
        let new_index = graph.values().borrow().len();
        assert!(x < new_index);
        
        let sin = Rc::new(Sin {
            graph: graph.clone(),
            index: new_index,
            x: x
        });
        let node_ptr : Rc<Node> = sin.clone();
        graph.values().borrow_mut().push(Rc::downgrade(&node_ptr));
        sin
    }
}

impl Node for Sin {
    fn owner(&self) -> &Rc<Graph> {
        &self.graph
    }
    
    fn index(&self) -> usize {
        self.index
    }
    
    fn eval(&self) -> f64 {
        f64::sin(self.graph.get(self.x).eval())
    }

    fn vjp(&self, g: f64, node: Rc<Node>, _: usize) -> f64 {
        g * f64::cos(node.eval())
    }
    
    fn parents(&self) -> Vec<usize> {
        vec![self.x]
    }
}

pub struct Cos {
    graph: Rc<Graph>,
    index: usize,
    x: usize
}

impl Cos {
    pub fn cos(graph: Rc<Graph>, x: usize) -> Rc<Cos> {
        let new_index = graph.values().borrow().len();
        assert!(x < new_index);
        
        let cos = Rc::new(Cos {
            graph: graph.clone(),
            index: new_index,
            x: x
        });
        let node_ptr : Rc<Node> = cos.clone();
        graph.values().borrow_mut().push(Rc::downgrade(&node_ptr));
        cos
    }
}

impl Node for Cos {
    fn owner(&self) -> &Rc<Graph> {
        &self.graph
    }
    
    fn index(&self) -> usize {
        self.index
    }
    
    fn eval(&self) -> f64 {
        f64::cos(self.graph.get(self.x).eval())
    }

    fn vjp(&self, g: f64, node: Rc<Node>, _: usize) -> f64 {
        - g * f64::sin(node.eval())
    }
    
    fn parents(&self) -> Vec<usize> {
        vec![self.x]
    }
}
