use ::{Expression, Context, Variable, Gradient, Container};
use std::ops::{Add, Sub, Div};
use num;

/// Estimate the gradient using finite differences
pub fn finite_diff_grad<T, E>(expr: &Container<T,E>, c: &mut Context<T>, var: &Variable, h: T) -> T
    where E: Expression<T>,
          T: Clone + Add<Output=T> + Sub<Output=T> + Div<Output=T>
{
    let curr_val = c.get_variable_value(var);
    let new_val = curr_val + h.clone();

    let curr_eval = expr.eval(c).value;
    c.set_variable_value(var, new_val);
    let new_eval = expr.eval(c).value;

    return (new_eval - curr_eval) / h

}

/// Compare the autodiff gradient to the finite diff gradient
pub fn compare_grads<T, E>(expr: Container<T,E>, mut c: Context<T>, var: &Variable, h: T) -> (T, T)
    where E: Expression<T>,
          T: num::Float
{
    let fin_diff_grad = finite_diff_grad(&expr, &mut c, var, h);

    let mut g = Gradient::of(expr, c);
    let auto_grad = g.grad(var);

    (fin_diff_grad, auto_grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num::Float;
    use ::functions::*;

    #[test]
    fn test_linear_grad_compare() {
        let mut context = Context::new();
        let x = context.create_variable(0.5);
        let y = context.create_variable(3.0);

        let f = x * y;

        let (findiff, autodiff) = compare_grads(f, context, &x, 0.01);
        assert!((findiff - autodiff).abs() < 1e-8);
    }

    #[test]
    fn test_sin_func_grad_compare() {
        let mut context = Context::new();
        let x = context.create_variable(0.5);
        let y = context.create_variable(3.0);

        let f = x * y + cos(y) * sin(x);

        let (findiff, autodiff) = compare_grads(f, context, &x, 1e-8);
        println!("{}", (findiff - autodiff).abs());
        assert!((findiff - autodiff).abs() < 1e-7);
    }
}
