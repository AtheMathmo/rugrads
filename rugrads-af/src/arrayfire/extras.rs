use rugrads;
use rugrads::{Node, VecJacProduct, Expression, LeafVar};

use libaf;
use libaf::{Array, Dim4};

use ::{Context, Container};
use super::utils::repeat_to_match_dims;

#[derive(Clone)]
pub struct LogSumExpVJP(Array);

impl VecJacProduct<Array> for LogSumExpVJP {
    fn vjp(&self, g: Array, node: &Node<Array>, _: &Node<Array>, _: usize) -> Array {
        let output_dims = self.0.dims();
        let g_tiled = repeat_to_match_dims(&g, output_dims);
        let node_tiled = repeat_to_match_dims(&node.value(), output_dims);
        return libaf::mul(&g_tiled, &libaf::exp(&(&self.0 - node_tiled)), false)
    }
}

#[derive(Clone, Copy)]
pub struct LogSumExp<X: Expression<Array>>(X, Option<i32>);

impl<X> Expression<Array> for LogSumExp<X> 
    where X: Expression<Array>
{
    fn eval(&self, c: &mut Context) -> Node<Array> {
        let x_eval = self.0.eval(c);
        let x_dims = x_eval.value().dims();
        let max = match self.1 {
            Some(dim) => repeat_to_match_dims(&libaf::max(&x_eval.value(), dim), x_dims),
            None => libaf::constant(libaf::max_all(&x_eval.value()).0, x_dims)
        };
        let exp_values = libaf::exp(&(x_eval.value() - &max));
        let expsum = match self.1 {
            Some(dim) => repeat_to_match_dims(&libaf::sum(&exp_values, dim), x_dims),
            None => libaf::constant(libaf::sum_all(&exp_values).0, x_dims)
        };
        let out_val = max + libaf::log(&expsum);

        let parents = vec![x_eval];
        let progenitors = Node::get_progenitors(&parents);
        let lse_clone = parents[0].value().clone();
        Node::new(c, out_val,
                    parents, progenitors, Box::new(LogSumExpVJP(lse_clone)))
    }
}

/// Takes the elementwise Power Raising of an Array
pub fn logsumexp<E: Expression<Array>>(input: Container<E>, dim: Option<i32>) -> Container<LogSumExp<E>> {
    Container::new(LogSumExp(input.into_inner(), dim))
}

/// Takes the log softmax of a given input array
pub fn logsoftmax<E: Clone + Expression<Array>>(input: Container<E>, dim: Option<i32>)
    -> Container<rugrads::functions::Sub<Array, E, LogSumExp<E>>> {
    input.clone() - logsumexp(input, dim)
}

/// ReLU Activation function
pub fn relu<E: Expression<Array>>(input: Container<E>)
    -> Container<super::MaxOf<LeafVar<Array>, rugrads::Container<Array, E>>>
{
    let zeros = LeafVar(libaf::constant(0.0, Dim4::new(&[1,1,1,1])));
    Container::new(super::MaxOf(zeros, input, true))
}

#[cfg(test)]
mod tests {

    use libaf;
    use libaf::{Array, Dim4};

    use ::{Gradient, Context};

    use super::*;
    use ::testsupport::array_eq;

    #[test]
    fn test_logsumexp() {
        libaf::set_backend(libaf::Backend::CPU);
        let mut context = Context::new();

        let dims = Dim4::new(&[2,2,1,1]);
        let arr = Array::new(&[0.5, 0.5, 0.25, 0.25], dims.clone());
        let var = context.create_variable(arr);
        let loss = logsumexp(var, None);
        {
            let _ = loss.eval(&mut context);
        }

        let mut g = Gradient::of(loss, context);
        
        let _ = g.grad(&var);
    }

    #[test]
    fn test_logsumexp_dim1() {
        libaf::set_backend(libaf::Backend::CPU);
        let mut context = Context::new();

        let dims = Dim4::new(&[2,2,1,1]);
        let arr = Array::new(&[0.5, 0.5, 0.25, 0.25], dims.clone());
        let var = context.create_variable(arr);
        let lse = logsumexp(var, Some(1));
        let loss = ::sum_all(lse);
        {
            let _ = loss.eval(&mut context);
        }
    }

    #[test]
    fn test_relu() {
        libaf::set_backend(libaf::Backend::CPU);
        let mut context = Context::new();

        let dims = Dim4::new(&[2,2,1,1]);
        let arr = Array::new(&[-0.5, 0.5, -0.25, 0.25], dims.clone());
        let out_arr = Array::new(&[0.0, 0.5, 0.0, 0.25] ,dims.clone());
        let var = context.create_variable(arr);
        let loss = relu(var);
        {
            let out = loss.eval(&mut context);
            assert!(array_eq(out.value(), &out_arr, 1e-8));
        }

        let mut g = Gradient::of(loss, context);
        
        let _ = g.grad(&var);
    }
}

