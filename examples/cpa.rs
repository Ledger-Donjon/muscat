use cpa::cpa_normal::*;
use cpa::leakage::{hw, sbox};
use cpa::tools::{progress_bar, read_array_2_from_npy_file, write_array};
use indicatif::ProgressIterator;
use ndarray::*;
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::time::{self};


// leakage model
pub fn leakage_model(value: ArrayView1<usize>, guess: usize) -> usize {
    hw(sbox((value[1] ^ guess) as u8) as usize)
}


// traces format
type FormatTraces = f64;
type FormatMetadata = u8;

#[allow(dead_code)]
fn cpa() {
    let start_sample: usize = 0;
    let end_sample: usize = 5000;
    let size: usize = end_sample - start_sample; // Number of samples
    let patch: usize = 500;
    let guess_range = 256; // 2**(key length)
    let folder = String::from("../data/cw");
    let dir_l = format!("{folder}/leakages.npy");
    let dir_p = format!("{folder}/plaintexts.npy");
    let leakages: Array2<FormatTraces> = read_array_2_from_npy_file::<FormatTraces>(&dir_l);
    let plaintext: Array2<FormatMetadata> = read_array_2_from_npy_file::<FormatMetadata>(&dir_p);
    let len_traces = leakages.shape()[0];
    let mut cpa_parallel = ((0..len_traces).step_by(patch)).progress_with(progress_bar(len_traces))
        .map(|row| row)
        .par_bridge()
        .map(|row_number| {
            let mut cpa = Cpa::new(size, patch, guess_range, leakage_model);
            let range_rows = row_number..row_number + patch;
            let range_samples = start_sample..end_sample;
            let sample_traces = leakages
                .slice(s![range_rows.clone(), range_samples])
                .map(|l| *l as f32);
            let sample_metadata: ArrayBase<OwnedRepr<usize>, Dim<[usize; 2]>> =
                plaintext.slice(s![range_rows, ..]).map(|p| *p as usize);
            cpa.update(sample_traces, sample_metadata);
            cpa
        })
        .reduce(
            || Cpa::new(size, patch, guess_range, leakage_model),
            |x, y| x + y,
        );
    cpa_parallel.finalize();
    println!("Guessed key = {}", cpa_parallel.pass_guess());
    write_array("results/corr.npy", cpa_parallel.pass_corr_array().view())
}



#[allow(dead_code)]
fn success() {
    let start_sample: usize = 0;
    let end_sample: usize = 5000;
    let size: usize = end_sample - start_sample; // Number of samples
    let patch: usize = 500;
    let guess_range = 256; // 2**(key length)
    let folder = String::from("../data/log_584012"); // "../../../intenship/scripts/log_584012"
    let nfiles = 13; // Number of files in the directory. TBD: Automating this value
    let rank_traces: usize = 1000;
    let mut cpa = Cpa::new(size, patch, guess_range, leakage_model);
    cpa.success_traces(rank_traces);
    for i in (0..nfiles).progress() {
        let dir_l = format!("{folder}/l/{i}.npy");
        let dir_p = format!("{folder}/p/{i}.npy");
        let leakages: Array2<FormatTraces> = read_array_2_from_npy_file::<FormatTraces>(&dir_l);
        let plaintext: Array2<FormatMetadata> =
            read_array_2_from_npy_file::<FormatMetadata>(&dir_p);
        let len_leakages = leakages.shape()[0];
        for row in (0..len_leakages).step_by(patch) {
            let range_samples = start_sample..end_sample;
            let range_rows: std::ops::Range<usize> = row..row + patch;
            let range_metadat = 0..plaintext.shape()[1];
            let sample_traces = leakages
                .slice(s![range_rows.clone(), range_samples]).map(|l| *l as f32);
            let sample_metadata: Array2<FormatMetadata>= plaintext
                .slice(s![range_rows, range_metadat]).to_owned();
            cpa.update_success(sample_traces, sample_metadata);
        }
    }
    cpa.finalize();
    println!("Guessed key = {}", cpa.pass_guess());
    // save corr key curves in npy
    write_array("results/success.npy", cpa.pass_rank().view());
}



fn main(){
    let t = time::Instant::now();
    cpa();
    println!("{:?}", t.elapsed());
}




