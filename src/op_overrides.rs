use std::ops;

use ::{Expression, Container};
use ::{Add, Mul};

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
