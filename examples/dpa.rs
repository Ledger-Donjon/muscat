use indicatif::ProgressIterator;
use muscat::dpa::*;
use muscat::leakage::sbox;
use muscat::util::read_array_2_from_npy_file;
use ndarray::*;
use rayon::iter::{ParallelBridge, ParallelIterator};

// traces format
type FormatTraces = f64;
type FormatMetadata = u8;

// leakage model
pub fn leakage_model(value: Array1<FormatMetadata>, guess: usize) -> usize {
    (sbox((value[1] as usize ^ guess) as u8)) as usize
}

fn dpa() {
    let start_sample: usize = 0;
    let end_sample: usize = 2000;
    let size: usize = end_sample - start_sample; // Number of samples
    let guess_range = 256; // 2**(key length)
    let folder = String::from("../../data/cw");
    let dir_l = format!("{folder}/leakages.npy");
    let dir_p = format!("{folder}/plaintexts.npy");
    let leakages: Array2<FormatTraces> = read_array_2_from_npy_file::<FormatTraces>(&dir_l);
    let plaintext: Array2<FormatMetadata> = read_array_2_from_npy_file::<FormatMetadata>(&dir_p);
    let len_traces = 20000; //leakages.shape()[0];
    let mut dpa = Dpa::new(size, guess_range, leakage_model);
    for i in (0..len_traces).progress() {
        let tmp_trace = leakages
            .row(i)
            .slice(s![start_sample..end_sample])
            .map(|t| *t as f32);
        let tmp_metadata = plaintext.row(i).to_owned();
        dpa.update(tmp_trace, tmp_metadata);
    }
    dpa.finalize();
    println!("Guessed key = {}", dpa.pass_guess());
    // let corr = dpa.pass_corr_array();
}

#[allow(dead_code)]
fn dpa_success() {
    let start_sample: usize = 0;
    let end_sample: usize = 2000;
    let size: usize = end_sample - start_sample; // Number of samples
    let guess_range = 256; // 2**(key length)
    let folder = String::from("../../data/cw");
    let dir_l = format!("{folder}/leakages.npy");
    let dir_p = format!("{folder}/plaintexts.npy");
    let leakages: Array2<FormatTraces> = read_array_2_from_npy_file::<FormatTraces>(&dir_l);
    let plaintext: Array2<FormatMetadata> = read_array_2_from_npy_file::<FormatMetadata>(&dir_p);
    let len_traces = leakages.shape()[0];
    let mut dpa = Dpa::new(size, guess_range, leakage_model);
    let rank_traces: usize = 100;
    dpa.assign_rank_traces(rank_traces);

    for i in (0..len_traces).progress() {
        let tmp_trace = leakages
            .row(i)
            .slice(s![start_sample..end_sample])
            .map(|t| *t as f32);
        let tmp_metadata = plaintext.row(i).to_owned();
        dpa.update_success(tmp_trace, tmp_metadata);
    }

    println!("Guessed key = {:?}", dpa.pass_guess());
    // let succss = dpa.pass_rank().to_owned();
}

#[allow(dead_code)]
fn dpa_parallel() {
    let start_sample: usize = 0;
    let end_sample: usize = 2000;
    let size: usize = end_sample - start_sample; // Number of samples
    let guess_range = 256; // 2**(key length)
    let folder = String::from("../../data/cw");
    let dir_l = format!("{folder}/leakages.npy");
    let dir_p = format!("{folder}/plaintexts.npy");
    let leakages: Array2<FormatTraces> = read_array_2_from_npy_file::<FormatTraces>(&dir_l);
    let plaintext: Array2<FormatMetadata> = read_array_2_from_npy_file::<FormatMetadata>(&dir_p);
    let len_traces = 20000; //leakages.shape()[0];
    let patch: usize = 2500;
    let mut dpa = (0..len_traces)
        .step_by(patch)
        .par_bridge()
        .map(|range_rows: usize| {
            let tmp_leakages = leakages
                .slice(s![range_rows..range_rows + patch, start_sample..end_sample])
                .map(|l| *l as f32);
            let tmp_metadata = plaintext
                .slice(s![range_rows..range_rows + patch, ..])
                .to_owned();
            let mut dpa_inner = Dpa::new(size, guess_range, leakage_model);
            for i in 0..patch {
                let trace = tmp_leakages.row(i).to_owned();
                let metadata = tmp_metadata.row(i).to_owned();
                dpa_inner.update(trace, metadata);
            }
            dpa_inner
        })
        .reduce(|| Dpa::new(size, guess_range, leakage_model), |x, y| x + y);

    dpa.finalize();
    println!("{:?}", dpa.pass_guess());
    // let corr = dpa.pass_corr_array();
}

fn main() {
    dpa();
}
