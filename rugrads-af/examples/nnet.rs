//! This is a super-ugly proof of concept for optimizing neural networks
//! 
//! This currently optimize correctly. Bug and/or bad hyperparameters.

extern crate arrayfire as af;
extern crate rugrads_af as raf;
use raf::{Array, Dim4};
use af::{RandomEngine, DEFAULT_RANDOM_ENGINE};

extern crate vision;
use vision::mnist::{MNISTBuilder};

extern crate rugrads;
use rugrads::Expression;

// fn toy_dataset() -> (Array, Array) {
//     let input_data = vec![0.52, 1.12,  0.77,
//                           0.88, 1.08, 0.15,
//                           0.90, 1.32, 0.07,
//                           0.73, 0.91, 0.22,
//                           0.52, 0.06, -1.30,
//                           0.31, 0.00, -1.63,
//                           0.72, 0.02, -1.17,
//                           0.74, -2.49, 1.39,
//                           0.66, -3.01, 1.77];
//     let target_data = vec![1.0, 0.0, 0.0,
//                            1.0, 0.0, 0.0,
//                            1.0, 0.0, 0.0,
//                            1.0, 0.0, 0.0,
//                            0.0, 1.0, 0.0,
//                            0.0, 1.0, 0.0,
//                            0.0, 1.0, 0.0,
//                            0.0, 0.0, 1.0,
//                            0.0, 0.0, 1.0];

//     let inputs = raf::new_array(&input_data, raf::Dim2([9, 3]));
//     let targets = raf::new_array(&target_data, raf::Dim2([9, 3]));

//     (inputs, targets)
// }

fn get_mnist(train_size: Option<usize>) -> (Array, Array) {
    let builder = MNISTBuilder::new();
    let mnist = builder.verbose().get_data().unwrap();

    let data_size = if let Some(x) = train_size {
        x
    } else { mnist.train_imgs.len() };

    let processed_imgs: Vec<f64> = mnist.train_imgs.iter().take(data_size).flat_map(|x| {
        x.iter().map(|y| (*y as f64) / 255.0)
    }).collect();

    let processed_labels: Vec<f64> = mnist.train_labels.iter().take(data_size).flat_map(|i| {
        let mut a = vec![0f64; 10];
        a[*i as usize] = 1.0;
        a.into_iter()
    }).collect();

    // Currently row-major but column-major expected
    let inputs = af::transpose(
        &raf::new_array(&processed_imgs, raf::Dim2([784, data_size as u64])),
        false);
    let targets = af::transpose(
        &raf::new_array(&processed_labels, raf::Dim2([10, data_size as u64])),
        false);

    (inputs, targets)
    
}

fn pred_acc(preds: &Array, targets: &Array) -> f64 {
    let (_, preds) = af::imax(preds, 1);
    let (_, targets) = af::imax(targets, 1);

    af::mean_all(&af::eq(&preds, &targets, false)).0
}

fn main() {
    // Set the backend and create context for auto diff
    raf::set_backend(raf::Backend::CPU);
    let mut context = raf::Context::new();

    // Create all context variables
    let (inputs, targets) = get_mnist(Some(300));
    let x = context.create_variable(inputs);
    let y = context.create_variable(targets.clone());

    let engine = RandomEngine::new(DEFAULT_RANDOM_ENGINE, None);
    let weights_l1 =  af::random_normal::<f64>(Dim4::new(&[784, 300, 1, 1]), engine.clone()) * (2.0 / 784.0);
    let weights_l2 = af::random_normal::<f64>(Dim4::new(&[300, 10, 1, 1]), engine) * (2.0 / 300.0);

    let w1 = context.create_variable(weights_l1);
    let w2 = context.create_variable(weights_l2);

    // Network forward pass
    let h1 = raf::matmul(x, w1, raf::MatProp::NONE, raf::MatProp::NONE);
    let a1 = raf::tanh(h1);

    let h2 = raf::matmul(a1, w2, raf::MatProp::NONE, raf::MatProp::NONE);
    let preds = raf::logsoftmax(h2, Some(1));

    // Set up the loss function
    let loss = -raf::sum_all(raf::mul(preds.clone(), y, false));

    // Set up Gradient object to auto diff
    let mut g = raf::Gradient::of(loss, context);

    // Optimize the weights
    let alpha = raf::constant(0.001, Dim4::new(&[1, 1, 1, 1]));

    for i in 0..50 {
        // println!("Epoch: {}", i);
        let w1_grad = g.grad(&w1);
        let w2_grad = g.grad(&w2);

        *g.0.get_mut(&w1) = g.0.get(&w1) - w1_grad * &alpha;
        *g.0.get_mut(&w2) = g.0.get(&w2) - w2_grad * &alpha;
    }

    // Print the predicted outputs on the training data
    // let train_preds = raf::exp(preds);
    // af::print(train_preds.eval(g.0.context()).value());
    let acc = pred_acc(preds.eval(g.0.context()).value(), &targets);
    println!("{}", acc);
}
