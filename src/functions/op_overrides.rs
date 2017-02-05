use std::ops;

use ::{Expression, Container};
use super::{Add, Mul, Div, Sub, Neg};

impl<E1: Expression, E2: Expression> ops::Add<Container<E2>> for Container<E1> {
    type Output = Container<Add<E1,E2>>;

    fn add(self, rhs: Container<E2>)-> Container<Add<E1,E2>> {
        Container(Add(self.0, rhs.0))
    }
}

impl<E1: Expression, E2: Expression> ops::Mul<Container<E2>> for Container<E1> {
    type Output = Container<Mul<E1,E2>>;

    fn mul(self, rhs: Container<E2>)-> Container<Mul<E1,E2>> {
        Container(Mul(self.0, rhs.0))
    }
}

impl<E1: Expression, E2: Expression> ops::Div<Container<E2>> for Container<E1> {
    type Output = Container<Div<E1,E2>>;

    fn div(self, rhs: Container<E2>)-> Container<Div<E1,E2>> {
        Container(Div(self.0, rhs.0))
    }
}

impl<E1: Expression, E2: Expression> ops::Sub<Container<E2>> for Container<E1> {
    type Output = Container<Sub<E1,E2>>;

    fn sub(self, rhs: Container<E2>)-> Container<Sub<E1,E2>> {
        Container(Sub(self.0, rhs.0))
    }
}

impl<E: Expression> ops::Neg for Container<E> {
    type Output = Container<Neg<E>>;

    fn neg(self)-> Container<Neg<E>> {
        Container(Neg(self.0))
    }
}

#[cfg(test)]
mod tests {
    use ::{Expression, Gradient, Context};
    use std::f64;

    #[test]
    fn test_add_op() {
        let mut c = Context::new();
        let x = c.create_variable(1.0);
        let y = c.create_variable(1.5);

        let f = x + y;
        assert!((f.eval(&mut c).value - 2.5).abs() < 1e-5);
        let mut g = Gradient::of(f, c);

        assert!((g.grad(x) - 1.0).abs() < 1e-5);
        assert!((g.grad(y) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mul_op() {
        let mut c = Context::new();
        let x = c.create_variable(1.0);
        let y = c.create_variable(1.5);

        let f = x * y;
        assert!((f.eval(&mut c).value - 1.5).abs() < 1e-5);
        let mut g = Gradient::of(f, c);

        assert!((g.grad(x) - 1.5).abs() < 1e-5);
        assert!((g.grad(y) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_div_op() {
        let mut c = Context::new();
        let x = c.create_variable(1.0);
        let y = c.create_variable(1.5);

        let f = x / y;
        assert!((f.eval(&mut c).value - f64::recip(1.5)).abs() < 1e-5);
        let mut g = Gradient::of(f, c);

        assert!((g.grad(x) - f64::recip(1.5)).abs() < 1e-5);
        assert!((g.grad(y) + f64::recip(2.25)).abs() < 1e-5);
    }

    #[test]
    fn test_sub_op() {
        let mut c = Context::new();
        let x = c.create_variable(1.0);
        let y = c.create_variable(1.5);

        let f = x - y;
        assert!((f.eval(&mut c).value + 0.5).abs() < 1e-5);
        let mut g = Gradient::of(f, c);

        assert!((g.grad(x) - 1.0).abs() < 1e-5);
        assert!((g.grad(y) + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_neg_op() {
        let mut c = Context::new();
        let x = c.create_variable(1.5);

        let f = -x;
        assert!((f.eval(&mut c).value + 1.5).abs() < 1e-5);
        let mut g = Gradient::of(f, c);
        assert!((g.grad(x) + 1.0).abs() < 1e-5);
    }
}
