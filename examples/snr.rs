use anyhow::{Context, Result};
use gnuplot::{Figure, PlotOption::Caption};
use muscat::{leakage_detection::SnrProcessor, util::chipwhisperer_float_to_u16};
use ndarray::Array2;
use ndarray_npy::read_npy;
use std::{env, iter::zip, path::PathBuf};

fn main() -> Result<()> {
    let traces_dir =
        PathBuf::from(env::var("TRACES_DIR").context("Missing TRACES_DIR environment variable")?);

    let traces: Array2<f64> =
        read_npy(traces_dir.join("traces.npy")).context("Failed to read traces.npy")?;
    let plaintexts: Array2<u8> =
        read_npy(traces_dir.join("plaintexts.npy")).context("Failed to read plaintexts.npy")?;
    assert_eq!(traces.shape()[0], plaintexts.shape()[0]);

    // Let's find the leakage of P[0]
    let mut processor = SnrProcessor::new(traces.shape()[1], 256);
    for (trace, plaintext) in zip(traces.rows(), plaintexts.rows()) {
        processor.process(
            trace.mapv(chipwhisperer_float_to_u16).view(),
            plaintext[0].into(),
        );
    }

    let snr = processor.snr();

    let mut fg = Figure::new();
    fg.axes2d().lines(0..snr.len(), snr, &[Caption("SNR")]);
    fg.show()?;

    Ok(())
}
