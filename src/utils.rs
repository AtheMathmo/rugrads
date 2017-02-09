use std::ops::Add;

/// Computes the sum of the given slice
///
/// Computes the sum by using the AddAssign operator
/// on the first element.
pub fn assigning_sum<T>(xs: &[T]) -> T
    where T: Clone + Add<Output = T>
{
    debug_assert!(xs.len() > 0, "Cannot do an assigning sum if vec is empty");

    let mut sum = xs[0].clone();

    for elmt in &xs[1..] {
        sum = sum + elmt.clone();
    }

    sum
}
