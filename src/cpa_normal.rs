use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::{iter::zip, ops::Add};

use crate::cpa::Cpa;

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
        let trace_batch = trace_batch.mapv(|t| t.into());
        let plaintext_batch = plaintext_batch.mapv(|m| m.into());

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

    /// Determine if two [`CpaProcessor`] are compatible for addition.
    ///
    /// If they were created with the same parameters, they are compatible.
    ///
    /// Note: [`CpaProcessor::leakage_func`] cannot be checked for equality, but they must have the
    /// same leakage functions in order to be compatible.
    fn is_compatible_with(&self, other: &Self) -> bool {
        self.num_samples == other.num_samples
            && self.batch_size == other.batch_size
            && self.guess_range == other.guess_range
    }
}

impl<F> Add for CpaProcessor<F>
where
    F: Fn(ArrayView1<usize>, usize) -> usize,
{
    type Output = Self;

    /// Merge computations of two [`CpaProcessor`]. Processors need to be compatible to be merged
    /// together, otherwise it can panic or yield incoherent result (see
    /// [`CpaProcessor::is_compatible_with`]).
    ///
    /// # Panics
    /// Panics in debug if the processors are not compatible.
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(self.is_compatible_with(&rhs));

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
