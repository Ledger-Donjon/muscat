use std::{env, iter::zip, path::PathBuf};

use gnuplot::{Figure, PlotOption::Caption};
use muscat::{distinguishers::dpa::DpaProcessor, leakage_model::aes::sbox};
use ndarray::Array2;
use ndarray_npy::read_npy;

fn selection_function(plaintext_byte: u8, guess: usize) -> bool {
    sbox(plaintext_byte ^ guess as u8) & 1 == 1
}

fn main() {
    let traces_dir =
        PathBuf::from(env::var("TRACES_DIR").expect("Missing TRACES_DIR environment variable"));

    let traces: Array2<f64> = read_npy(traces_dir.join("traces.npy")).unwrap();
    let plaintexts: Array2<u8> = read_npy(traces_dir.join("plaintexts.npy")).unwrap();
    assert_eq!(traces.shape()[0], plaintexts.shape()[0]);

    // Let's recover the first byte of the key
    let mut processor = DpaProcessor::new(traces.shape()[1], 256);
    for (trace, plaintext) in zip(traces.rows(), plaintexts.rows()) {
        processor.update(
            // Convert chipwhisperer float to int
            trace.mapv(|x| ((x + 1.) * 1024.) as u16).view(),
            plaintext[0],
            selection_function,
        );
    }

    let dpa = processor.finalize();

    let best_guess = dpa.best_guess();
    println!("Best subkey guess: {:?}", best_guess);

    // Let's plot correlation coefficients of the best guess
    let differential_curves = dpa.differential_curves();
    let differential_curve = differential_curves.row(best_guess);

    let mut fg = Figure::new();
    fg.axes2d().lines(
        0..differential_curve.len(),
        differential_curve,
        &[Caption("Differential curve")],
    );
    fg.show().unwrap();
}
