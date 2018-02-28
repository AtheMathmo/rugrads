use libaf;
use libaf::{Array, Dim4};

/// Repeats the input array to match the target dimensions
/// 
/// Note: This should be setup to fail if broadcasting is not possible
pub fn repeat_to_match_dims(a: &Array, target_dim: Dim4) -> Array {
    // Get b's dims (Being able to `Index` Dim4 would be helpful here)
    let tile_dims: Vec<_> = a.dims().get().iter().zip(target_dim.get().iter())
                                          .map(|(x,y)| y / x).collect();
    let mut dim_arr = [0; 4];
    dim_arr.clone_from_slice(&tile_dims);
    // Tile it to fit the first two dimensions
    return libaf::tile(&a, Dim4::new(&dim_arr));
}

/// Balanced equality for gradient of comparison operators
pub fn balanced_eq(x: &Array, z: &Array, y: &Array, batch: bool) -> Array {
    libaf::eq(x, z, batch) / (libaf::eq(x, y, batch) + 1.0)
}