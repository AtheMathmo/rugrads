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
