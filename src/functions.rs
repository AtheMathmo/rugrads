//! Functions module
//!
//! This module contains differentiable wrapper functions.

use super::{Expression, VecJacProduct, Node, Context, IdentityVJP};

/// Addition operation
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

/// Multiplication operation
pub struct Mul<X: Expression, Y: Expression>(pub X, pub Y);

struct MulVJP(f64, f64);

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

struct LinVJP<F: Fn(f64) -> f64>(F);

impl<F: Fn(f64) -> f64> VecJacProduct for LinVJP<F> {
    fn vjp(&self, g: f64, x: &Node, _: usize) -> f64 {
        g * self.0(x.value)
    }
}

/// Sine operator
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

/// Cosine operator
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

/// Exponential operator
pub struct Exp<X: Expression>(pub X);

impl<X: Expression> Expression for Exp<X> {
    fn eval(&self, c: &mut Context) -> Node {
        let x_eval = self.0.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node {
            index: c.get_index(),
            value: f64::exp(parents[0].value),
            parents: parents,
            progenitors: progenitors,
            _vjp: Box::new(LinVJP(f64::exp)),
        }
    }
}

/// Natural Logarithm operator
pub struct Ln<X: Expression>(pub X);

impl<X: Expression> Expression for Ln<X> {
    fn eval(&self, c: &mut Context) -> Node {
        let x_eval = self.0.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node {
            index: c.get_index(),
            value: f64::ln(parents[0].value),
            parents: parents,
            progenitors: progenitors,
            _vjp: Box::new(LinVJP(f64::recip)),
        }
    }
}

/// Power raising operation
pub struct Powf<X: Expression>(pub X, pub f64);

impl<X: Expression> Expression for Powf<X> {
    fn eval(&self, c: &mut Context) -> Node {
        let x_eval = self.0.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);
        let n = self.1;

        Node {
            index: c.get_index(),
            value: f64::powf(parents[0].value, n),
            parents: parents,
            progenitors: progenitors,
            _vjp: Box::new(LinVJP(move |x| n * f64::powf(x, n - 1f64))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::{Node, Context, Expression, IdentityVJP, VecJacProduct};

    pub struct TestVar(f64);

    impl Expression for TestVar {
        fn eval(&self, c: &mut Context) -> Node {
            Node {
                index: c.get_index(),
                value: self.0,
                parents: vec![],
                progenitors: vec![],
                _vjp: Box::new(IdentityVJP),
            }
        }
    }

    #[test]
    fn test_add() {
        let mut c = Context::new();
        let f = Add(TestVar(1.0), TestVar(1.0));
        let node = f.eval(&mut c);
        // Just a dummy node
        let x = &node.parents[0];
        assert!((node.value - 2.0).abs() < 1e-5);
        assert!((node.vjp(1.0, x, 0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mul() {
        let mut c = Context::new();
        let f = Mul(TestVar(0.5), TestVar(0.3));
        let node = f.eval(&mut c);
        // Just a dummy node
        let x = &node.parents[0];
        assert!((node.value - 0.5*0.3).abs() < 1e-5);
        // 2 * 0.3
        assert!((node.vjp(2.0, x, 0) - 0.6).abs() < 1e-5);
        // 3 * 0.5
        assert!((node.vjp(3.0, x, 1) - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_sin() {
        let mut c = Context::new();
        let f = Sin(TestVar(0.5));
        let node = f.eval(&mut c);
        let x = &node.parents[0];
        assert!((node.value - f64::sin(0.5)).abs() < 1e-5);
        assert!((node.vjp(1.0, x, 0) - f64::cos(0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_cos() {
        let mut c = Context::new();
        let f = Cos(TestVar(0.5));
        let node = f.eval(&mut c);
        let x = &node.parents[0];
        assert!((node.value - f64::cos(0.5)).abs() < 1e-5);
        assert!((node.vjp(1.0, x, 0) + f64::sin(0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_exp() {
        let mut c = Context::new();
        let f = Exp(TestVar(0.5));
        let node = f.eval(&mut c);
        let x = &node.parents[0];
        assert!((node.value - f64::exp(0.5)).abs() < 1e-5);
        assert!((node.vjp(1.0, x, 0) - f64::exp(0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_ln() {
        let mut c = Context::new();
        let f = Ln(TestVar(0.5));
        let node = f.eval(&mut c);
        let x = &node.parents[0];
        assert!((node.value - f64::ln(0.5)).abs() < 1e-5);
        assert!((node.vjp(1.0, x, 0) - f64::recip(0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_powf_integer() {
        let mut c = Context::new();
        let f = Powf(TestVar(3.0), 2.0);
        let node = f.eval(&mut c);
        let x = &node.parents[0];
        assert!((node.value - f64::powf(3.0, 2.0)).abs() < 1e-5);
        assert!((node.vjp(1.0, x, 0) - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_powf_neg_non_int() {
        let mut c = Context::new();
        let f = Powf(TestVar(3.0), -1.3);
        let node = f.eval(&mut c);
        let x = &node.parents[0];
        assert!((node.value - f64::powf(3.0, -1.3)).abs() < 1e-5);
        assert!((node.vjp(1.0, x, 0) + 1.3 * f64::powf(3.0, -2.3)).abs() < 1e-5);
    }
}
