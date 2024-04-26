use anyhow::Result;
use indicatif::ProgressIterator;
use muscat::dpa::DpaProcessor;
use muscat::leakage::sbox;
use muscat::util::read_array2_from_npy_file;
use ndarray::{s, Array1, Array2};
use rayon::iter::{ParallelBridge, ParallelIterator};

// traces format
type FormatTraces = f64;
type FormatMetadata = u8;

// leakage model
pub fn leakage_model(value: Array1<FormatMetadata>, guess: usize) -> usize {
    (sbox((value[1] as usize ^ guess) as u8)) as usize
}

fn dpa() -> Result<()> {
    let start_sample = 0;
    let end_sample = 2000;
    let size = end_sample - start_sample; // Number of samples
    let guess_range = 256; // 2**(key length)
    let folder = String::from("../../data/cw");
    let dir_l = format!("{folder}/leakages.npy");
    let dir_p = format!("{folder}/plaintexts.npy");
    let leakages: Array2<FormatTraces> = read_array2_from_npy_file::<FormatTraces>(&dir_l)?;
    let plaintext: Array2<FormatMetadata> = read_array2_from_npy_file::<FormatMetadata>(&dir_p)?;
    let len_traces = 20000; //leakages.shape()[0];
    let mut dpa = DpaProcessor::new(size, guess_range, leakage_model);
    for i in (0..len_traces).progress() {
        let tmp_trace = leakages
            .row(i)
            .slice(s![start_sample..end_sample])
            .map(|t| *t as f32);
        let tmp_metadata = plaintext.row(i);
        dpa.update(tmp_trace.view(), tmp_metadata.to_owned());
    }
    let dpa = dpa.finalize();
    println!("Guessed key = {:02x}", dpa.pass_guess());
    // let corr = dpa.pass_corr_array();

    Ok(())
}

#[allow(dead_code)]
fn dpa_success() -> Result<()> {
    let start_sample = 0;
    let end_sample = 2000;
    let size = end_sample - start_sample; // Number of samples
    let guess_range = 256; // 2**(key length)
    let folder = String::from("../../data/cw");
    let dir_l = format!("{folder}/leakages.npy");
    let dir_p = format!("{folder}/plaintexts.npy");
    let leakages: Array2<FormatTraces> = read_array2_from_npy_file::<FormatTraces>(&dir_l)?;
    let plaintext: Array2<FormatMetadata> = read_array2_from_npy_file::<FormatMetadata>(&dir_p)?;
    let len_traces = leakages.shape()[0];
    let mut dpa = DpaProcessor::new(size, guess_range, leakage_model);
    let rank_traces: usize = 100;
    dpa.assign_rank_traces(rank_traces);

    for i in (0..len_traces).progress() {
        let tmp_trace = leakages
            .row(i)
            .slice(s![start_sample..end_sample])
            .map(|t| *t as f32);
        let tmp_metadata = plaintext.row(i).to_owned();
        dpa.update_success(tmp_trace.view(), tmp_metadata);
    }

    let dpa = dpa.finalize();
    println!("Guessed key = {:02x}", dpa.pass_guess());
    // let succss = dpa.pass_rank().to_owned();

    Ok(())
}

#[allow(dead_code)]
fn dpa_parallel() -> Result<()> {
    let start_sample = 0;
    let end_sample = 2000;
    let size = end_sample - start_sample; // Number of samples
    let guess_range = 256; // 2**(key length)
    let folder = String::from("../../data/cw");
    let dir_l = format!("{folder}/leakages.npy");
    let dir_p = format!("{folder}/plaintexts.npy");
    let leakages: Array2<FormatTraces> = read_array2_from_npy_file::<FormatTraces>(&dir_l)?;
    let plaintext: Array2<FormatMetadata> = read_array2_from_npy_file::<FormatMetadata>(&dir_p)?;
    let len_traces = 20000; //leakages.shape()[0];
    let batch = 2500;
    let mut dpa = (0..len_traces)
        .step_by(batch)
        .par_bridge()
        .map(|range_rows: usize| {
            let tmp_leakages = leakages
                .slice(s![range_rows..range_rows + batch, start_sample..end_sample])
                .map(|l| *l as f32);
            let tmp_metadata = plaintext
                .slice(s![range_rows..range_rows + batch, ..])
                .to_owned();
            let mut dpa_inner = DpaProcessor::new(size, guess_range, leakage_model);
            for i in 0..batch {
                let trace = tmp_leakages.row(i);
                let metadata = tmp_metadata.row(i).to_owned();
                dpa_inner.update(trace, metadata);
            }
            dpa_inner
        })
        .reduce(
            || DpaProcessor::new(size, guess_range, leakage_model),
            |x, y| x + y,
        );

    let dpa = dpa.finalize();
    println!("{:2x}", dpa.pass_guess());
    // let corr = dpa.pass_corr_array();

    Ok(())
}

fn main() -> Result<()> {
    dpa()
}
