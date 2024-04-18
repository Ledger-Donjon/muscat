use anyhow::Result;
use indicatif::ProgressIterator;
use muscat::cpa::CpaProcessor;
use muscat::leakage::{hw, sbox};
use muscat::util::{progress_bar, read_array2_from_npy_file, save_array};
use ndarray::s;
use rayon::prelude::{ParallelBridge, ParallelIterator};

// traces format
type FormatTraces = i16;
type FormatMetadata = i32;

// leakage model
pub fn leakage_model(value: usize, guess: usize) -> usize {
    hw(sbox((value ^ guess) as u8) as usize)
}

fn rank() -> Result<()> {
    let size = 5000; // Number of samples
    let guess_range = 256; // 2**(key length)
    let target_byte = 1;
    let folder = String::from("../../data");
    let nfiles = 5;
    let chunk = 3000;
    let mut rank = CpaProcessor::new(size, guess_range, target_byte, leakage_model);
    for file in (0..nfiles).progress_with(progress_bar(nfiles)) {
        let dir_l = format!("{folder}/l{file}.npy");
        let dir_p = format!("{folder}/p{file}.npy");
        let leakages = read_array2_from_npy_file::<FormatTraces>(&dir_l)?;
        let plaintext = read_array2_from_npy_file::<FormatMetadata>(&dir_p)?;
        let len_file = leakages.shape()[0];
        for sample in (0..len_file).step_by(chunk) {
            let l_sample: ndarray::ArrayBase<
                ndarray::ViewRepr<&FormatTraces>,
                ndarray::Dim<[usize; 2]>,
            > = leakages.slice(s![sample..sample + chunk, ..]);
            let p_sample = plaintext.slice(s![sample..sample + chunk, ..]);
            let x = (0..chunk)
                .par_bridge()
                .fold(
                    || CpaProcessor::new(size, guess_range, target_byte, leakage_model),
                    |mut r: CpaProcessor, n| {
                        r.update(
                            l_sample.row(n).map(|l| *l as usize).view(),
                            p_sample.row(n).map(|p| *p as usize).view(),
                        );
                        r
                    },
                )
                .reduce(
                    || CpaProcessor::new(size, guess_range, target_byte, leakage_model),
                    |lhs, rhs| lhs + rhs,
                );
            rank = rank + x;
        }
    }

    let rank = rank.finalize();

    // save rank key curves in npy
    save_array("../results/rank.npy", &rank.pass_rank())?;

    Ok(())
}

fn main() -> Result<()> {
    rank()?;

    Ok(())
}
