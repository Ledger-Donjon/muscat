use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::{iter::zip, ops::Add};

use crate::util::{argsort_by, max_per_row};

/// Computes the [`Cpa`] of the given traces using [`CpaProcessor`].
///
/// # Panics
/// - Panic if `leakages.shape()[0] != plaintexts.shape()[0]`
/// - Panic if `chunk_size` is 0.
pub fn cpa<T, U, F>(
    leakages: ArrayView2<T>,
    plaintexts: ArrayView2<U>,
    guess_range: usize,
    leakage_func: F,
    chunk_size: usize,
) -> Cpa
where
    T: Into<f32> + Copy + Sync,
    U: Into<usize> + Copy + Sync,
    F: Fn(ArrayView1<usize>, usize) -> usize + Send + Sync + Copy,
{
    assert_eq!(leakages.shape()[0], plaintexts.shape()[0]);
    assert!(chunk_size > 0);

    zip(
        leakages.axis_chunks_iter(Axis(0), chunk_size),
        plaintexts.axis_chunks_iter(Axis(0), chunk_size),
    )
    .par_bridge()
    .fold(
        || CpaProcessor::new(leakages.shape()[1], chunk_size, guess_range, leakage_func),
        |mut cpa, (leakages_chunk, plaintexts_chunk)| {
            cpa.update(leakages_chunk, plaintexts_chunk);

            cpa
        },
    )
    .reduce_with(|x, y| x + y)
    .unwrap()
    .finalize()
}

pub struct Cpa {
    /// Guess range upper excluded bound
    guess_range: usize,
    /// Pearson correlation coefficients
    corr: Array2<f32>,
}

impl Cpa {
    pub fn rank(&self) -> Array1<usize> {
        let rank = argsort_by(&self.max_corr().to_vec()[..], f32::total_cmp);

        Array1::from_vec(rank)
    }

    pub fn corr(&self) -> ArrayView2<f32> {
        self.corr.view()
    }

    pub fn best_guess(&self) -> usize {
        let max_corr = self.max_corr();

        let mut best_guess_corr = 0.0;
        let mut best_guess = 0;

        for guess in 0..self.guess_range {
            if max_corr[guess] > best_guess_corr {
                best_guess_corr = max_corr[guess];
                best_guess = guess;
            }
        }

        best_guess
    }

    pub fn max_corr(&self) -> Array1<f32> {
        max_per_row(self.corr.view())
    }
}

pub struct CpaProcessor<F>
where
    F: Fn(ArrayView1<usize>, usize) -> usize,
{
    /// Number of samples per trace
    num_samples: usize,
    /// Guess range upper excluded bound
    guess_range: usize,
    /// Sum of traces
    sum_leakages: Array1<f32>,
    /// Sum of square of traces
    sum2_leakages: Array1<f32>,
    /// Sum of traces per key guess
    guess_sum_leakages: Array1<f32>,
    /// Sum of square of traces per key guess
    guess_sum2_leakages: Array1<f32>,
    values: Array2<f32>,
    cov: Array2<f32>,
    /// Batch size
    batch_size: usize,
    /// Leakage model
    leakage_func: F,
    /// Number of traces processed
    num_traces: usize,
}

/* This class implements the CPA algorithm shown in:
https://www.iacr.org/archive/ches2004/31560016/31560016.pdf */

impl<F> CpaProcessor<F>
where
    F: Fn(ArrayView1<usize>, usize) -> usize,
{
    pub fn new(num_samples: usize, batch_size: usize, guess_range: usize, leakage_func: F) -> Self {
        Self {
            num_samples,
            guess_range,
            sum_leakages: Array1::zeros(num_samples),
            sum2_leakages: Array1::zeros(num_samples),
            guess_sum_leakages: Array1::zeros(guess_range),
            guess_sum2_leakages: Array1::zeros(guess_range),
            values: Array2::zeros((batch_size, guess_range)),
            cov: Array2::zeros((guess_range, num_samples)),
            batch_size,
            leakage_func,
            num_traces: 0,
        }
    }

    /// # Panics
    /// - Panic in debug if `trace_batch.shape()[0] != plaintext_batch.shape()[0]`.
    /// - Panic in debug if `trace_batch.shape()[1] != self.num_samples`.
    pub fn update<T, U>(&mut self, trace_batch: ArrayView2<T>, plaintext_batch: ArrayView2<U>)
    where
        T: Into<f32> + Copy,
        U: Into<usize> + Copy,
    {
        debug_assert_eq!(trace_batch.shape()[0], plaintext_batch.shape()[0]);
        debug_assert_eq!(trace_batch.shape()[1], self.num_samples);

        /* This function updates the internal arrays of the CPA
        It accepts trace_batch and plaintext_batch to update them*/
        let trace_batch = trace_batch.map(|&t| t.into());
        let plaintext_batch = plaintext_batch.map(|&m| m.into());

        self.update_values(trace_batch.view(), plaintext_batch.view(), self.guess_range);
        self.update_key_leakages(trace_batch.view(), self.guess_range);

        self.num_traces += self.batch_size;
    }

    fn update_values(
        /* This function generates the values and cov arrays */
        &mut self,
        trace: ArrayView2<f32>,
        metadata: ArrayView2<usize>,
        guess_range: usize,
    ) {
        for row in 0..self.batch_size {
            for guess in 0..guess_range {
                let pass_to_leakage = metadata.row(row);
                self.values[[row, guess]] = (self.leakage_func)(pass_to_leakage, guess) as f32;
            }
        }

        self.cov = self.cov.clone() + self.values.t().dot(&trace);
    }

    fn update_key_leakages(&mut self, trace: ArrayView2<f32>, guess_range: usize) {
        for i in 0..self.num_samples {
            self.sum_leakages[i] += trace.column(i).sum(); // trace[i] as usize;
            self.sum2_leakages[i] += trace.column(i).dot(&trace.column(i)); // (trace[i] * trace[i]) as usize;
        }

        for guess in 0..guess_range {
            self.guess_sum_leakages[guess] += self.values.column(guess).sum(); //self.values[guess] as usize;
            self.guess_sum2_leakages[guess] +=
                self.values.column(guess).dot(&self.values.column(guess));
            // (self.values[guess] * self.values[guess]) as usize;
        }
    }

    /// Finalizes the calculation after feeding the overall traces.
    pub fn finalize(&self) -> Cpa {
        let cov_n = self.cov.clone() / self.num_traces as f32;
        let avg_keys = self.guess_sum_leakages.clone() / self.num_traces as f32;
        let std_key = self.guess_sum2_leakages.clone() / self.num_traces as f32;
        let avg_leakages = self.sum_leakages.clone() / self.num_traces as f32;
        let std_leakages = self.sum2_leakages.clone() / self.num_traces as f32;

        let mut corr = Array2::zeros((self.guess_range, self.num_samples));
        for i in 0..self.guess_range {
            for x in 0..self.num_samples {
                let numerator = cov_n[[i, x]] - (avg_keys[i] * avg_leakages[x]);

                let denominator_1 = std_key[i] - (avg_keys[i] * avg_keys[i]);

                let denominator_2 = std_leakages[x] - (avg_leakages[x] * avg_leakages[x]);
                if numerator != 0.0 {
                    corr[[i, x]] = f32::abs(numerator / f32::sqrt(denominator_1 * denominator_2));
                }
            }
        }

        Cpa {
            guess_range: self.guess_range,
            corr,
        }
    }
}

impl<F> Add for CpaProcessor<F>
where
    F: Fn(ArrayView1<usize>, usize) -> usize,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.num_samples, rhs.num_samples);
        debug_assert_eq!(self.batch_size, rhs.batch_size);
        debug_assert_eq!(self.guess_range, rhs.guess_range);

        // WARN: `self.leakage_func` and `rhs.leakage_func` must be the same function

        Self {
            num_samples: self.num_samples,
            guess_range: self.guess_range,
            sum_leakages: self.sum_leakages + rhs.sum_leakages,
            sum2_leakages: self.sum2_leakages + rhs.sum2_leakages,
            guess_sum_leakages: self.guess_sum_leakages + rhs.guess_sum_leakages,
            guess_sum2_leakages: self.guess_sum2_leakages + rhs.guess_sum2_leakages,
            values: self.values + rhs.values,
            cov: self.cov + rhs.cov,
            batch_size: self.batch_size,
            leakage_func: self.leakage_func,
            num_traces: self.num_traces + rhs.num_traces,
        }
    }
}
