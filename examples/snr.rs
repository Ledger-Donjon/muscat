use anyhow::Result;
use indicatif::ProgressIterator;
use muscat::processors::Snr;
use muscat::quicklog::{BatchIter, Log};
use muscat::util::{progress_bar, save_array};
use rayon::prelude::{ParallelBridge, ParallelIterator};

fn main() -> Result<()> {
    // Open log file
    // This uses logs from the python quicklog library.
    let log = Log::<i16>::new("log")?;
    let leakage_size = log.leakage_size();
    let trace_count = log.len();

    let result: Snr = log
        .into_iter()
        .progress_with(progress_bar(trace_count))
        // Process records sharing same leakage numpy files by batches, so batch files get read only
        // once.
        // Log entries may have lots of information. The closure argument here extracts the data
        // required for the analysis, in this case the first byte of an hexadecimal string in the
        // "data" column of the record.
        .batches(|record| record.bytes("data")[0])
        // Use `par_bridge` from rayon crate to make processing multithreaded
        .par_bridge()
        .fold(
            || Snr::new(leakage_size, 256),
            |mut snr, batch| {
                for trace in batch {
                    // `process` takes an `ArrayView1` argument, which makes possible to pass a
                    // trace slice: `traces.leakage.slice(s![100..])` for instance.
                    snr.process(&trace.leakage.view(), trace.value as usize)
                }
                snr
            },
        )
        // Merge the results of each processing thread
        .reduce(|| Snr::new(leakage_size, 256), |a, b| a + b);

    // Save the resulting SNR trace to a numpy file
    save_array("result.npy", &result.snr())?;

    Ok(())
}
