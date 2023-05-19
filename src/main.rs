use std::ops::Add;

use indicatif::ProgressIterator;
use processors::Snr;
use quicklog::{BatchIter, Log};
use rayon::prelude::{ParallelBridge, ParallelIterator};
use util::{progress_bar, save_array};
mod processors;
mod quicklog;
mod trace;
mod util;

struct Analysis {
    snr: Snr,
}

impl Analysis {
    fn new(leakage_size: usize) -> Self {
        Self {
            snr: Snr::new(leakage_size, 256),
        }
    }
}

impl Add<Analysis> for Analysis {
    type Output = Self;

    fn add(self, rhs: Analysis) -> Self::Output {
        Self {
            snr: self.snr + rhs.snr
        }
    }
}

fn main() {
    // Open log file
    let log = Log::<i16>::new("log").unwrap();
    let leakage_size = log.leakage_size();
    let trace_count = log.len();

    let result: Analysis = log.into_iter()
        .progress_with(progress_bar(trace_count))
        // Process records sharing same leakage numpy files by batches, so batch files get read only
        // once.
        .batches(|record| record.bytes("new")[0])
        // Use rayon to make processing multithreaded
        .par_bridge()
        .fold(|| Analysis::new(leakage_size), |mut analysis, batch| {
            for trace in batch {
                analysis.snr.process(&trace.leakage, trace.value as usize)
            }
            analysis
        // Merge the results of each processing thread
        }).reduce(|| Analysis::new(leakage_size), |a, b| a + b);

    save_array("result.npy", &result.snr.snr()).unwrap();
}
