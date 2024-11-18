use anyhow::Result;
use indicatif::ProgressIterator;
use muscat::distinguishers::dpa::DpaProcessor;
use muscat::leakage_model::aes::sbox;
use muscat::util::read_array2_from_npy_file;
use ndarray::{s, Array1};
use rayon::iter::{ParallelBridge, ParallelIterator};

// traces format
type FormatTraces = f64;
type FormatMetadata = u8;

pub fn selection_function(value: Array1<FormatMetadata>, guess: usize) -> bool {
    (sbox((value[1] as usize ^ guess) as u8)) as usize & 1 == 1
}

fn dpa() -> Result<()> {
    let start_sample = 0;
    let end_sample = 2000;
    let size = end_sample - start_sample; // Number of samples
    let guess_range = 256; // 2**(key length)
    let folder = String::from("../../data/cw");
    let dir_l = format!("{folder}/leakages.npy");
    let dir_p = format!("{folder}/plaintexts.npy");
    let traces = read_array2_from_npy_file::<FormatTraces>(&dir_l)?;
    let plaintext = read_array2_from_npy_file::<FormatMetadata>(&dir_p)?;
    let len_traces = 20000; //traces.shape()[0];
    let mut dpa_proc = DpaProcessor::new(size, guess_range, selection_function);
    for i in (0..len_traces).progress() {
        let tmp_trace = traces
            .row(i)
            .slice(s![start_sample..end_sample])
            .mapv(|t| t as f32);
        let tmp_metadata = plaintext.row(i);
        dpa_proc.update(tmp_trace.view(), tmp_metadata.to_owned());
    }
    let dpa = dpa_proc.finalize();
    println!("Guessed key = {:02x}", dpa.best_guess());
    // let corr = dpa.corr();

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
    let traces = read_array2_from_npy_file::<FormatTraces>(&dir_l)?;
    let plaintext = read_array2_from_npy_file::<FormatMetadata>(&dir_p)?;
    let len_traces = traces.shape()[0];
    let mut dpa_proc = DpaProcessor::new(size, guess_range, selection_function);
    let rank_traces: usize = 100;

    let mut rank = Array1::zeros(guess_range);
    for i in (0..len_traces).progress() {
        let tmp_trace = traces
            .row(i)
            .slice(s![start_sample..end_sample])
            .mapv(|t| t as f32);
        let tmp_metadata = plaintext.row(i).to_owned();
        dpa_proc.update(tmp_trace.view(), tmp_metadata);

        if i % rank_traces == 0 {
            // rank can be saved to get its evolution
            rank = dpa_proc.finalize().rank();
        }
    }

    let dpa = dpa_proc.finalize();
    println!("Guessed key = {:02x}", dpa.best_guess());
    println!("{:?}", rank);

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
    let traces = read_array2_from_npy_file::<FormatTraces>(&dir_l)?;
    let plaintext = read_array2_from_npy_file::<FormatMetadata>(&dir_p)?;
    let len_traces = 20000; // traces.shape()[0];
    let batch = 2500;
    let dpa = (0..len_traces)
        .step_by(batch)
        .par_bridge()
        .map(|range_rows| {
            let tmp_traces = traces
                .slice(s![range_rows..range_rows + batch, start_sample..end_sample])
                .mapv(|l| l as f32);
            let tmp_metadata = plaintext
                .slice(s![range_rows..range_rows + batch, ..])
                .to_owned();

            let mut dpa_inner = DpaProcessor::new(size, guess_range, selection_function);
            for i in 0..batch {
                let trace = tmp_traces.row(i);
                let metadata = tmp_metadata.row(i).to_owned();
                dpa_inner.update(trace, metadata);
            }
            dpa_inner
        })
        .reduce(
            || DpaProcessor::new(size, guess_range, selection_function),
            |x, y| x + y,
        )
        .finalize();

    println!("{:2x}", dpa.best_guess());
    // let corr = dpa.corr();

    Ok(())
}

fn main() -> Result<()> {
    dpa()
}
