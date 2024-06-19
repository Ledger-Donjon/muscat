use anyhow::Result;
use indicatif::ProgressIterator;
use muscat::cpa_normal::CpaProcessor;
use muscat::leakage::{hw, sbox};
use muscat::util::{progress_bar, read_array2_from_npy_file, save_array2};
use ndarray::*;
use ndarray_npy::write_npy;
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::time;

// leakage model
pub fn leakage_model(value: ArrayView1<usize>, guess: usize) -> usize {
    hw(sbox((value[1] ^ guess) as u8) as usize)
}

// traces format
type FormatTraces = f64;
type FormatMetadata = u8;

#[allow(dead_code)]
fn cpa() -> Result<()> {
    let start_sample = 0;
    let end_sample = 5000;
    let size = end_sample - start_sample; // Number of samples
    let batch = 500;
    let guess_range = 256; // 2**(key length)
    let folder = String::from("../../data/cw");
    let dir_l = format!("{folder}/leakages.npy");
    let dir_p = format!("{folder}/plaintexts.npy");
    let leakages = read_array2_from_npy_file::<FormatTraces>(&dir_l)?;
    let plaintext = read_array2_from_npy_file::<FormatMetadata>(&dir_p)?;
    let len_traces = leakages.shape()[0];

    let cpa_parallel = ((0..len_traces).step_by(batch))
        .progress_with(progress_bar(len_traces))
        .par_bridge()
        .map(|row_number| {
            let mut cpa = CpaProcessor::new(size, batch, guess_range, leakage_model);
            let range_rows = row_number..row_number + batch;
            let range_samples = start_sample..end_sample;
            let sample_traces = leakages
                .slice(s![range_rows.clone(), range_samples])
                .map(|l| *l as f32);
            let sample_metadata = plaintext.slice(s![range_rows, ..]).map(|p| *p as usize);
            cpa.update(sample_traces.view(), sample_metadata.view());
            cpa
        })
        .reduce(
            || CpaProcessor::new(size, batch, guess_range, leakage_model),
            |x, y| x + y,
        );

    let cpa = cpa_parallel.finalize();
    println!("Guessed key = {}", cpa.best_guess());

    save_array2("results/corr.npy", cpa.corr())?;

    Ok(())
}

#[allow(dead_code)]
fn success() -> Result<()> {
    let start_sample = 0;
    let end_sample = 5000;
    let size = end_sample - start_sample; // Number of samples
    let batch = 500;
    let guess_range = 256; // 2**(key length)
    let folder = String::from("../data/log_584012"); // "../../../intenship/scripts/log_584012"
    let nfiles = 13; // Number of files in the directory. TBD: Automating this value
    let rank_traces = 1000;

    let mut cpa = CpaProcessor::new(size, batch, guess_range, leakage_model);

    let mut rank = Array1::zeros(guess_range);
    let mut processed_traces = 0;
    for i in (0..nfiles).progress() {
        let dir_l = format!("{folder}/l/{i}.npy");
        let dir_p = format!("{folder}/p/{i}.npy");
        let leakages = read_array2_from_npy_file::<FormatTraces>(&dir_l)?;
        let plaintext = read_array2_from_npy_file::<FormatMetadata>(&dir_p)?;
        for row in (0..leakages.shape()[0]).step_by(batch) {
            let range_samples = start_sample..end_sample;
            let range_rows = row..row + batch;
            let range_metadata = 0..plaintext.shape()[1];
            let sample_traces = leakages
                .slice(s![range_rows.clone(), range_samples])
                .map(|l| *l as f32);
            let sample_metadata = plaintext.slice(s![range_rows, range_metadata]);

            cpa.update(sample_traces.view(), sample_metadata);
            processed_traces += sample_traces.len();
            if processed_traces % rank_traces == 0 {
                // rank can be saved to get its evolution
                rank = cpa.finalize().rank();
            }
        }
    }

    let cpa = cpa.finalize();
    println!("Guessed key = {}", cpa.best_guess());

    println!("{:?}", rank);

    // save corr key curves in npy
    write_npy("results/success.npy", &cpa.rank().map(|&x| x as u64))?;

    Ok(())
}

fn main() -> Result<()> {
    let t = time::Instant::now();
    cpa()?;
    println!("{:?}", t.elapsed());

    Ok(())
}
