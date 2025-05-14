use anyhow::{Context, Result};
use gnuplot::{Figure, PlotOption::Caption};
use muscat::{distinguishers::cpa_normal::CpaProcessor, leakage_model::aes::sbox};
use ndarray::{Array2, ArrayView1, Axis};
use ndarray_npy::read_npy;
use std::{env, iter::zip, path::PathBuf};

fn leakage_model(plaintext: ArrayView1<usize>, guess: usize) -> usize {
    sbox((plaintext[0] ^ guess) as u8) as usize
}

fn main() -> Result<()> {
    let traces_dir =
        PathBuf::from(env::var("TRACES_DIR").context("Missing TRACES_DIR environment variable")?);

    let traces: Array2<f64> =
        read_npy(traces_dir.join("traces.npy")).context("Failed to read traces.npy")?;
    let plaintexts: Array2<u8> =
        read_npy(traces_dir.join("plaintexts.npy")).context("Failed to read plaintexts.npy")?;
    assert_eq!(traces.shape()[0], plaintexts.shape()[0]);

    // Let's recover the first byte of the key
    let mut processor = CpaProcessor::new(traces.shape()[1], 100, 256);
    for (trace_batch, plaintext_batch) in zip(
        traces.axis_chunks_iter(Axis(0), 100),
        plaintexts.axis_chunks_iter(Axis(0), 100),
    ) {
        processor.update(
            // Convert chipwhisperer float to int
            trace_batch.mapv(|x| ((x + 1.) * 1024.) as u16).view(),
            plaintext_batch,
            leakage_model,
        );
    }

    let cpa = processor.finalize();

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
    fg.show()?;

    Ok(())
}
