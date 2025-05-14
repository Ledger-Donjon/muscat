use gnuplot::{Figure, PlotOption::Caption};
use muscat::{distinguishers::cpa::CpaProcessor, leakage_model::aes::sbox};
use ndarray::Array2;
use ndarray_npy::read_npy;
use std::{env, iter::zip, path::PathBuf};

fn leakage_model(plaintext_byte: usize, guess: usize) -> usize {
    sbox((plaintext_byte ^ guess) as u8) as usize
}

fn main() {
    let traces_dir =
        PathBuf::from(env::var("TRACES_DIR").expect("Missing TRACES_DIR environment variable"));

    let traces: Array2<f64> = read_npy(traces_dir.join("traces.npy")).unwrap();
    let plaintexts: Array2<u8> = read_npy(traces_dir.join("plaintexts.npy")).unwrap();
    assert_eq!(traces.shape()[0], plaintexts.shape()[0]);

    // Let's recover the first byte of the key
    let mut processor = CpaProcessor::new(traces.shape()[1], 256);
    for (trace, plaintext) in zip(traces.rows(), plaintexts.rows()) {
        processor.update(
            // Convert chipwhisperer float to int
            trace.mapv(|x| ((x + 1.) * 1024.) as u16).view(),
            plaintext[0],
            leakage_model,
        );
    }

    let cpa = processor.finalize(leakage_model);

    let best_guess = cpa.best_guess();
    println!("Best subkey guess: {:?}", best_guess);

    // Let's plot correlation coefficients of the best guess
    let corr = cpa.corr();
    let corr_best_guess = corr.row(best_guess);

    let mut fg = Figure::new();
    fg.axes2d().lines(
        0..corr_best_guess.len(),
        corr_best_guess,
        &[Caption("Pearson correlation coefficient")],
    );
    fg.show().unwrap();
}
