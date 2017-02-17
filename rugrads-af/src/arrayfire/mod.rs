use rugrads::{Node, VecJacProduct, Expression};

use libaf;
use libaf::Array;

use ::Context;

pub mod wrappers;

struct LinVJP<F>(F)
    where for<'a> F: Fn(&'a Array) -> Array;

impl<F> VecJacProduct<Array> for LinVJP<F>
    where for<'a> F: Fn(&'a Array) -> Array
{
    fn vjp(&self, g: Array, x: &Node<Array>, _: usize) -> Array {
        g * (self.0)(x.value())
    }
}

macro_rules! univariate_wrapper {
    ($name: ident, $af_func: expr, $vjp: expr) => {
pub struct $name<X: Expression<Array>>(X);

impl<X: Expression<Array>> Expression<Array> for $name<X> {
    fn eval(&self, c: &mut Context) -> Node<Array> {
        let x_eval = self.0.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node::new(c, $af_func(parents[0].value()),
                    parents, progenitors, Box::new(LinVJP($vjp)))
    }
}
    };
}

univariate_wrapper!(Sin, libaf::sin, libaf::cos);
univariate_wrapper!(Cos, libaf::cos, |x| -libaf::sin(x));
univariate_wrapper!(Tan, libaf::tan, |x| {
    let ones = libaf::constant(1f64, x.dims());
    let cos_x = libaf::cos(x);
    libaf::div(&ones, &libaf::mul(&cos_x, &cos_x, false), false)
});
univariate_wrapper!(Sinh, libaf::sinh, libaf::cosh);
univariate_wrapper!(Cosh, libaf::cosh, libaf::sinh);
univariate_wrapper!(Tanh, libaf::tanh, |x| {
    let ones = libaf::constant(1f64, x.dims());
    let cosh_x = libaf::cosh(x);
    libaf::div(&ones, &libaf::mul(&cosh_x, &cosh_x, false), false)
});
univariate_wrapper!(Arcsin, libaf::asin, |x| {
    let ones = libaf::constant(1f64, x.dims());
    let x_sq = libaf::sub(&ones, &libaf::pow(x, &2f64, false), false);
    libaf::div(&ones, &x_sq, false)
});
univariate_wrapper!(Arccos, libaf::acos, |x| {
    let ones = libaf::constant(1f64, x.dims());
    let x_sq = libaf::sub(&ones, &libaf::pow(x, &2f64, false), false);
    -libaf::div(&ones, &x_sq, false)
});
univariate_wrapper!(Arctan, libaf::atan, |x| {
    let ones = libaf::constant(1f64, x.dims());
    let x_sq = libaf::add(&ones, &libaf::pow(x, &2f64, false), false);
    libaf::div(&ones, &x_sq, false)
});
univariate_wrapper!(Exp, libaf::exp, libaf::exp);
univariate_wrapper!(Log, libaf::log, move |x| libaf::pow(x, &-1f64, false));
univariate_wrapper!(Sigmoid, libaf::sigmoid, |x| {
    let exp = libaf::exp(x);
    let ones = libaf::constant(1f64, x.dims());

    let denom = libaf::pow(&libaf::add(&ones, &exp, false), &2f64, false);
    libaf::div(&exp, &denom, false)
});

pub struct Pow<X: Expression<Array>>(X, f64);

impl<X: Expression<Array>> Expression<Array> for Pow<X> {
    fn eval(&self, c: &mut Context) -> Node<Array> {
        let x_eval = self.0.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        Node::new(c, libaf::pow(parents[0].value(), &self.1, false),
                    parents, progenitors, Box::new(LinVJP(move |x| libaf::pow(x, &(self.1 - 1f64), false) * self.1)))
    }
}

pub struct SumAllVJP;

impl VecJacProduct<Array> for SumAllVJP {
    fn vjp(&self, g: Array, x: &Node<Array>, _: usize) -> Array {
        let output_dims = x.value().dims();
        libaf::tile(&g, output_dims)
    }
}

pub struct SumAll<X: Expression<Array>>(X);

impl<X: Expression<Array>> Expression<Array> for SumAll<X> {
    fn eval(&self, c: &mut Context) -> Node<Array> {
        let x_eval = self.0.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        let sum = libaf::sum_all(parents[0].value()).0;

        Node::new(c, libaf::constant(sum, libaf::Dim4::new(&[1,1,1,1])),
                    parents, progenitors, Box::new(SumAllVJP))
    }
}

pub struct NormVJP(libaf::NormType);

impl VecJacProduct<Array> for NormVJP {
    fn vjp(&self, g: Array, x: &Node<Array>, _: usize) -> Array {
        let output_dims = x.value().dims();
        let full_g = libaf::tile(&g, output_dims);
        match self.0 {
            libaf::NormType::VECTOR_2 => {
                libaf::mul(&full_g, x.value(), false) * 2
            },
            _ => panic!("Only Frobenius norm is supported currently")
        }
    }
}

pub struct Norm<X: Expression<Array>>(X, libaf::NormType, f64, f64);

impl<X: Expression<Array>> Expression<Array> for Norm<X> {
    fn eval(&self, c: &mut Context) -> Node<Array> {
        let x_eval = self.0.eval(c);
        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);

        let norm = libaf::norm(parents[0].value(), self.1, self.2, self.3);

        Node::new(c, libaf::constant(norm, libaf::Dim4::new(&[1,1,1,1])),
                    parents, progenitors, Box::new(NormVJP(self.1)))
    }
}

pub struct DotVJP(Array, Array);

impl VecJacProduct<Array> for DotVJP {
    fn vjp(&self, g: Array, _: &Node<Array>, argnum: usize) -> Array {
        let lhs_dim = self.0.dims().ndims();
        let rhs_dim = self.1.dims().ndims();

        match (argnum, lhs_dim, rhs_dim) {
            // LHS Vector derivative
            (0, 1, 1) => {
                libaf::mul(&g, &self.1, false)
            },
            // RHS Vector dot derivative
            (1, 1, 1) => {
                libaf::mul(&g, &self.0, false)
            },
            // RHS Matrix-Vector dot derivative
            (1, 2, 1) => {
                libaf::mul(&g, &self.0, false)
            },
            _ => panic!("dot product does not (yet) support matrix differentation")
        }
    }
}

pub struct Dot<X: Expression<Array>, Y: Expression<Array>>(X, Y);

impl<X, Y> Expression<Array> for Dot<X, Y>
    where X: Expression<Array>, Y: Expression<Array>
{
    fn eval(&self, c: &mut Context) -> Node<Array> {
        let x_eval = self.0.eval(c);
        let y_eval = self.1.eval(c);
        let parents = vec![x_eval, y_eval];
        let progenitors = Node::get_progenitors(&parents);

        let dot = libaf::dot(parents[0].value(), parents[1].value(),
                              libaf::MatProp::NONE, libaf::MatProp::NONE);
        
        let lhs_clone = parents[0].value().clone();
        let rhs_clone = parents[1].value().clone();
        Node::new(c, dot, parents, progenitors, Box::new(DotVJP(lhs_clone, rhs_clone)))
    }
}

pub struct AFMulVJP(Array, Array);

impl VecJacProduct<Array> for AFMulVJP {
    fn vjp(&self, g: Array, _: &Node<Array>, argnum: usize) -> Array {
        match argnum {
            0 => libaf::mul(&g, &self.1, false),
            1 => libaf::mul(&g, &self.0, false),
            _ => panic!("Invalid argnum fed to AFMul VJP")
        }
    }
}

pub struct AFMul<X: Expression<Array>, Y: Expression<Array>>(X, Y);

impl<X, Y> Expression<Array> for AFMul<X, Y>
    where X: Expression<Array>, Y: Expression<Array>
{
    fn eval(&self, c: &mut Context) -> Node<Array> {
        let x_eval = self.0.eval(c);
        let y_eval = self.1.eval(c);
        let parents = vec![x_eval, y_eval];
        let progenitors = Node::get_progenitors(&parents);

        let dot = libaf::mul(parents[0].value(), parents[1].value(), false);
        
        let lhs_clone = parents[0].value().clone();
        let rhs_clone = parents[1].value().clone();
        Node::new(c, dot, parents, progenitors, Box::new(AFMulVJP(lhs_clone, rhs_clone)))
    }
}

#[cfg(test)]
mod tests {
    use libaf;
    use libaf::{Array, Dim4};
    use rugrads::VecJacProduct;

    use ::testsupport::*;
    use ::Context;

    use super::*;

    macro_rules! test_univar_func {
        ($test_name: ident, $expr: ident, $func: expr, $grad: expr) => {
#[test]
fn $test_name() {
    libaf::set_backend(libaf::Backend::CPU);
    let dims = Dim4::new(&[2,2,1,1]);
    let arr = Array::new(&[0.5, 0.5, 0.25, 0.25], dims);
    let var = TestVar(arr.clone());
    let expr = $expr(var);

    let mut c = Context::new();
    let node = expr.eval(&mut c);
    assert!(array_eq(node.value(), &$func(&arr), 1e-5));

    let p = &node.parents()[0];
    let ones = libaf::constant(1f64, dims);
    let vjp = node.vjp(ones, &p, 0);
    assert!(array_eq(&vjp, &($grad)(&arr), 1e-5));
}
        };
    }

    test_univar_func!(test_sin, Sin, libaf::sin, libaf::cos);
    test_univar_func!(test_cos, Cos, libaf::cos, |x| -libaf::sin(x));
    test_univar_func!(test_sinh, Sinh, libaf::sinh, libaf::cosh);
    test_univar_func!(test_cosh, Cosh, libaf::cosh, libaf::sinh);
}