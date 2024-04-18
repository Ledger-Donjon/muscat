use ndarray::{concatenate, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::{iter::zip, ops::Add};

use crate::util::max_per_row;

/// Computes the [`Cpa`] of the given traces using [`CpaProcessor`].
///
/// # Panics
/// - Panic if `leakages.shape()[0] != plaintexts.shape()[0]`
/// - Panic if `chunk_size` is 0.
pub fn cpa<T, U>(
    leakages: ArrayView2<T>,
    plaintexts: ArrayView2<U>,
    guess_range: usize,
    leakage_func: fn(ArrayView1<usize>, usize) -> usize,
    chunk_size: usize,
) -> Cpa
where
    T: Into<f32> + Copy + Sync,
    U: Into<usize> + Copy + Sync,
{
    assert_eq!(leakages.shape()[0], plaintexts.shape()[0]);
    assert!(chunk_size > 0);

    zip(
        leakages.axis_chunks_iter(Axis(0), chunk_size),
        plaintexts.axis_chunks_iter(Axis(0), chunk_size),
    )
    .par_bridge()
    .map(|(leakages_chunk, plaintexts_chunk)| {
        let mut cpa = CpaProcessor::new(leakages.shape()[1], chunk_size, guess_range, leakage_func);
        cpa.update(leakages_chunk, plaintexts_chunk);
        cpa
    })
    .reduce(
        || CpaProcessor::new(leakages.shape()[1], chunk_size, guess_range, leakage_func),
        |x, y| x + y,
    )
    .finalize()
}

pub struct Cpa {
    /// Guess range upper excluded bound
    guess_range: usize,
    corr: Array2<f32>,
    max_corr: Array1<f32>,
    rank_slice: Array2<f32>,
}

impl Cpa {
    pub fn pass_rank(&self) -> ArrayView2<f32> {
        self.rank_slice.view()
    }

    pub fn pass_corr_array(&self) -> ArrayView2<f32> {
        self.corr.view()
    }

    pub fn pass_guess(&self) -> usize {
        let mut init_value = 0.0;
        let mut guess = 0;

        for i in 0..self.guess_range {
            if self.max_corr[i] > init_value {
                init_value = self.max_corr[i];
                guess = i;
            }
        }

        guess
    }
}

pub struct CpaProcessor {
    /// Number of samples per trace
    len_samples: usize,
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
    rank_traces: usize, // Number of traces to calculate succes rate
    /// Batch size
    batch_size: usize,
    /// Leakage model
    leakage_func: fn(ArrayView1<usize>, usize) -> usize,
    /// Number of traces processed
    len_leakages: usize,
}

/* This class implements the CPA algorithm shown in:
https://www.iacr.org/archive/ches2004/31560016/31560016.pdf */

impl CpaProcessor {
    pub fn new(
        size: usize,
        batch_size: usize,
        guess_range: usize,
        leakage_func: fn(ArrayView1<usize>, usize) -> usize,
    ) -> Self {
        Self {
            len_samples: size,
            guess_range,
            sum_leakages: Array1::zeros(size),
            sum2_leakages: Array1::zeros(size),
            guess_sum_leakages: Array1::zeros(guess_range),
            guess_sum2_leakages: Array1::zeros(guess_range),
            values: Array2::zeros((batch_size, guess_range)),
            cov: Array2::zeros((guess_range, size)),
            rank_traces: 0,
            batch_size,
            leakage_func,
            len_leakages: 0,
        }
    }

    /// # Panics
    /// - Panic in debug if `trace_batch.shape()[0] != plaintext_batch.shape()[0]`.
    /// - Panic in debug if `trace_batch.shape()[1] != self.len_samples`.
    pub fn update<T, U>(&mut self, trace_batch: ArrayView2<T>, plaintext_batch: ArrayView2<U>)
    where
        T: Into<f32> + Copy,
        U: Into<usize> + Copy,
    {
        debug_assert_eq!(trace_batch.shape()[0], plaintext_batch.shape()[0]);
        debug_assert_eq!(trace_batch.shape()[1], self.len_samples);

        /* This function updates the internal arrays of the CPA
        It accepts trace_batch and plaintext_batch to update them*/
        let trace_batch = trace_batch.map(|&t| t.into());
        let plaintext_batch = plaintext_batch.map(|&m| m.into());

        self.update_values(trace_batch.view(), plaintext_batch.view(), self.guess_range);
        self.update_key_leakages(trace_batch.view(), self.guess_range);

        self.len_leakages += self.batch_size;
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
        for i in 0..self.len_samples {
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

    pub fn update_success<T, U>(
        &mut self,
        trace_batch: ArrayView2<T>,
        plaintext_batch: ArrayView2<U>,
    ) -> Option<Cpa>
    where
        T: Into<f32> + Copy,
        U: Into<usize> + Copy,
    {
        /* This function updates the main arrays of the CPA for the success rate*/
        self.update(trace_batch, plaintext_batch);

        // WARN: if self.rank_traces == 0 this function will panic (division by zero)
        // WARN: if self.rank_traces is not divisible by self.batch_size this branch will never be
        // taken
        if self.len_leakages % self.rank_traces == 0 {
            let mut cpa = self.finalize();

            if self.len_leakages == self.rank_traces {
                cpa.rank_slice = cpa
                    .max_corr
                    .clone()
                    .into_shape((cpa.max_corr.shape()[0], 1))
                    .unwrap();
            } else {
                cpa.rank_slice = concatenate![
                    Axis(1),
                    cpa.rank_slice,
                    cpa.max_corr
                        .clone()
                        .into_shape((cpa.max_corr.shape()[0], 1))
                        .unwrap()
                ];
            }

            Some(cpa)
        } else {
            None
        }
    }

    pub fn finalize(&self) -> Cpa {
        /* This function finalizes the calculation after
        feeding all stored acc arrays */
        let cov_n = self.cov.clone() / self.len_leakages as f32;
        let avg_keys = self.guess_sum_leakages.clone() / self.len_leakages as f32;
        let std_key = self.guess_sum2_leakages.clone() / self.len_leakages as f32;
        let avg_leakages = self.sum_leakages.clone() / self.len_leakages as f32;
        let std_leakages = self.sum2_leakages.clone() / self.len_leakages as f32;

        let mut corr = Array2::zeros((self.guess_range, self.len_samples));
        for i in 0..self.guess_range {
            for x in 0..self.len_samples {
                let numerator = cov_n[[i, x]] - (avg_keys[i] * avg_leakages[x]);

                let denominator_1 = std_key[i] - (avg_keys[i] * avg_keys[i]);

                let denominator_2 = std_leakages[x] - (avg_leakages[x] * avg_leakages[x]);
                if numerator != 0.0 {
                    corr[[i, x]] = f32::abs(numerator / f32::sqrt(denominator_1 * denominator_2));
                }
            }
        }

        let max_corr = max_per_row(corr.view());

        Cpa {
            guess_range: self.guess_range,
            corr,
            max_corr,
            rank_slice: Array2::zeros((self.guess_range, 1)),
        }
    }

    pub fn success_traces(&mut self, traces_no: usize) {
        self.rank_traces = traces_no;
    }
}

impl Add for CpaProcessor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.len_samples, rhs.len_samples);
        debug_assert_eq!(self.batch_size, rhs.batch_size);
        debug_assert_eq!(self.guess_range, rhs.guess_range);

        Self {
            len_samples: self.len_samples,
            guess_range: self.guess_range,
            sum_leakages: self.sum_leakages + rhs.sum_leakages,
            sum2_leakages: self.sum2_leakages + rhs.sum2_leakages,
            guess_sum_leakages: self.guess_sum_leakages + rhs.guess_sum_leakages,
            guess_sum2_leakages: self.guess_sum2_leakages + rhs.guess_sum2_leakages,
            values: self.values + rhs.values,
            cov: self.cov + rhs.cov,
            rank_traces: self.rank_traces,
            batch_size: self.batch_size,
            leakage_func: self.leakage_func,
            len_leakages: self.len_leakages + rhs.len_leakages,
        }
    }
}
