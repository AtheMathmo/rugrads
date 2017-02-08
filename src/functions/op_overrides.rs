use std::ops;
use std::marker::PhantomData;

use num::Float;

use ::{Expression, Container};
use super::{Add, Mul, Div, Sub, Neg};

impl<T, E1, E2> ops::Add<Container<T, E2>> for Container<T, E1>
    where for<'a, 'b> &'a T: ops::Add<&'b T, Output=T>,
        E1: Expression<T>,
        E2: Expression<T> 
{
    type Output = Container<T, Add<T, E1,E2>>;

    fn add(self, rhs: Container<T, E2>)-> Container<T, Add<T, E1,E2>> {
        Container::new(Add::new(self.inner, rhs.inner))
    }
}

impl<T, E1, E2> ops::Mul<Container<T, E2>> for Container<T, E1>
    where T: Float,
        E1: Expression<T>,
        E2: Expression<T> 
{
    type Output = Container<T, Mul<T, E1, E2>>;

    fn mul(self, rhs: Container<T, E2>)-> Container<T, Mul<T, E1, E2>> {
        Container::new(Mul::new(self.inner, rhs.inner))
    }
}

impl<T, E1, E2> ops::Div<Container<T, E2>> for Container<T, E1>
    where T: Float,
        E1: Expression<T>,
        E2: Expression<T>        
{
    type Output = Container<T, Div<T, E1,E2>>;

    fn div(self, rhs: Container<T, E2>)-> Container<T, Div<T, E1, E2>> {
        Container::new(Div::new(self.inner, rhs.inner))
    }
}

impl<T, E1, E2> ops::Sub<Container<T, E2>> for Container<T, E1>
    where for<'a, 'b> &'a T: ops::Sub<&'b T, Output=T>,
        T: ops::Neg<Output=T>,
        E1: Expression<T>,
        E2: Expression<T> 
{
    type Output = Container<T, Sub<T, E1,E2>>;

    fn sub(self, rhs: Container<T, E2>)-> Container<T, Sub<T, E1,E2>> {
        Container::new(Sub::new(self.inner, rhs.inner))
    }
}

impl<T, E> ops::Neg for Container<T, E>
    where T: Clone + ops::Neg<Output=T>,
          E: Expression<T>
{
    type Output = Container<T, Neg<T, E>>;

    fn neg(self)-> Container<T, Neg<T, E>> {
        Container::new(Neg(self.inner, PhantomData::<T>))
    }
}

#[cfg(test)]
mod tests {
    use ::{Expression, Gradient, Context};
    use std::f64;
    use num::Float;

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
