use gnuplot::{Figure, PlotOption::Caption};
use muscat::leakage_detection::SnrProcessor;
use ndarray::Array2;
use ndarray_npy::read_npy;
use std::{env, iter::zip, path::PathBuf};

fn main() {
    let traces_dir =
        PathBuf::from(env::var("TRACES_DIR").expect("Missing TRACES_DIR environment variable"));

    let traces: Array2<f64> = read_npy(traces_dir.join("traces.npy")).unwrap();
    let plaintexts: Array2<u8> = read_npy(traces_dir.join("plaintexts.npy")).unwrap();
    assert_eq!(traces.shape()[0], plaintexts.shape()[0]);

    // Let's find the leakage of P[0]
    let mut processor = SnrProcessor::new(traces.shape()[1], 256);
    for (trace, plaintext) in zip(traces.rows(), plaintexts.rows()) {
        processor.process(
            // Convert chipwhisperer float to int
            trace.mapv(|x| ((x + 1.) * 1024.) as u16).view(),
            plaintext[0].into(),
        );
    }

    let snr = processor.snr();

    let mut fg = Figure::new();
    fg.axes2d().lines(0..snr.len(), snr, &[Caption("SNR")]);
    fg.show().unwrap();
}
