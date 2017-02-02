use std::marker::PhantomData;

/// A primitive wrapper for an unboxed function
pub struct Primitive<Args, F: FnMut<Args>> {
    pub func: F,
    _marker: PhantomData<Args>
}

impl<Args, F: FnMut<Args>> FnOnce<Args> for Primitive<Args, F> {
    type Output = F::Output;
    extern "rust-call" fn call_once(self, args: Args) -> F::Output {
        self.func.call_once(args)
    }
}

impl<Args, F: FnMut<Args>> FnMut<Args> for Primitive<Args, F> {
    extern "rust-call" fn call_mut(&mut self, args: Args) -> F::Output {
        self.func.call_mut(args)
    }
}

impl<Args, F: FnMut<Args>> From<F> for Primitive<Args, F> {
    fn from(f: F) -> Self {
        Primitive {
            func: f,
            _marker: PhantomData::<Args>
        }
    }
}

use num::Float;

pub fn constant<T: Float + 'static>(c: T) -> Box<FnMut<(), Output=T>> {
    Box::new(Primitive::from(move || c))
}

pub fn identity<T: Float + 'static>() -> Box<FnMut<(T,), Output=T>> {
    Box::new(Primitive::from(|x: T| x))
}

pub fn add<T: Float + 'static>() -> Box<FnMut<(T,T), Output=T>> {
    Box::new(Primitive::from(|x, y| x + y))
}

pub fn mul<T: Float + 'static>() -> Box<FnMut<(T,T), Output=T>> {
    Box::new(Primitive::from(|x, y| x * y))
}
