use muscat::cpa::*;
use muscat::leakage::{hw, sbox};
use muscat::util::{progress_bar, read_array_2_from_npy_file, save_array};
use ndarray::*;
use rayon::prelude::{ParallelBridge, ParallelIterator};
use simple_bar::ProgressBar;
use std::time::Instant;

// traces format
type FormatTraces = i16;
type FormatMetadata = i32;

// leakage model
pub fn leakage_model(value: usize, guess: usize) -> usize {
    hw(sbox((value ^ guess) as u8) as usize)
}

fn rank() {
    let size: usize = 5000; // Number of samples
    let guess_range = 256; // 2**(key length)
    let target_byte = 1;
    let folder = String::from("../../data");
    let nfiles = 5;
    // let mut bar = ProgressBar::default(nfiles as u32, 50, false);
    let bar = progress_bar(nfiles);
    let chunk = 3000;
    let mut rank = Cpa::new(size, guess_range, target_byte, leakage_model);
    for file in 0..nfiles {
        let dir_l = format!("{folder}/l{file}.npy");
        let dir_p = format!("{folder}/p{file}.npy");
        let leakages: Array2<FormatTraces> = read_array_2_from_npy_file(&dir_l);
        let plaintext: Array2<FormatMetadata> = read_array_2_from_npy_file(&dir_p);
        let len_file = leakages.shape()[0];
        for sample in (0..len_file).step_by(chunk) {
            let l_sample: ndarray::ArrayBase<
                ndarray::ViewRepr<&FormatTraces>,
                ndarray::Dim<[usize; 2]>,
            > = leakages.slice(s![sample..sample + chunk, ..]);
            let p_sample = plaintext.slice(s![sample..sample + chunk, ..]);
            let x = (0..chunk)
                .into_iter()
                .par_bridge()
                .fold(
                    || Cpa::new(size, guess_range, target_byte, leakage_model),
                    |mut r: Cpa, n| {
                        r.update(
                            l_sample.row(n).map(|l: &FormatTraces| *l as usize),
                            p_sample.row(n).map(|p: &FormatMetadata| *p as usize),
                        );
                        r
                    },
                )
                .reduce(
                    || Cpa::new(size, guess_range, target_byte, leakage_model),
                    |lhs, rhs| lhs + rhs,
                );
            rank = rank + x;
            rank.finalize();
        }
        bar.inc(file as u64);
    }
    // save rank key curves in npy
    save_array("../results/rank.npy", &rank.pass_rank());
}

fn main() {
    rank();
}
