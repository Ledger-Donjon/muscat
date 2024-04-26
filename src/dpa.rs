use ndarray::{concatenate, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::{iter::zip, ops::Add};

use crate::util::max_per_row;

pub fn dpa<M, T>(
    leakages: ArrayView2<T>,
    metadata: ArrayView1<M>,
    guess_range: usize,
    leakage_func: fn(M, usize) -> usize,
    chunk_size: usize,
) -> Dpa
where
    T: Into<f32> + Copy + Sync,
    M: Clone + Sync,
{
    zip(
        leakages.axis_chunks_iter(Axis(0), chunk_size),
        metadata.axis_chunks_iter(Axis(0), chunk_size),
    )
    .par_bridge()
    .map(|(leakages_chunk, metadata_chunk)| {
        let mut dpa = DpaProcessor::new(leakages.shape()[1], guess_range, leakage_func);

        for i in 0..leakages_chunk.shape()[0] {
            dpa.update(leakages_chunk.row(i), metadata_chunk[i].clone());
        }

        dpa
    })
    .reduce(
        || DpaProcessor::new(leakages.shape()[1], guess_range, leakage_func),
        |a, b| a + b,
    )
    .finalize()
}

pub struct Dpa {
    /// Guess range upper excluded bound
    guess_range: usize,
    corr: Array2<f32>,
    max_corr: Array1<f32>,
}

impl Dpa {
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

pub struct DpaProcessor<M> {
    /// Number of samples per trace
    len_samples: usize,
    /// Guess range upper excluded bound
    guess_range: usize,
    /// Sum of traces for which the selection function equals 0
    sum_0: Array2<f32>,
    /// Sum of traces for which the selection function equals 1
    sum_1: Array2<f32>,
    /// Number of traces processed for which the selection function equals 0
    count_0: Array1<usize>,
    /// Number of traces processed for which the selection function equals 1
    count_1: Array1<usize>,
    rank_slice: Array2<f32>,
    rank_traces: usize, // Number of traces to calculate succes rate
    /// Selection function
    leakage_func: fn(M, usize) -> usize,
    /// Number of traces processed
    len_leakages: usize,
}

/* This class implements the DPA algorithm shown in:
https://paulkocher.com/doc/DifferentialPowerAnalysis.pdf
https://web.mit.edu/6.857/OldStuff/Fall03/ref/kocher-DPATechInfo.pdf */

impl<M: Clone> DpaProcessor<M> {
    pub fn new(size: usize, guess_range: usize, f: fn(M, usize) -> usize) -> Self {
        Self {
            len_samples: size,
            guess_range,
            sum_0: Array2::zeros((guess_range, size)),
            sum_1: Array2::zeros((guess_range, size)),
            count_0: Array1::zeros(guess_range),
            count_1: Array1::zeros(guess_range),
            rank_slice: Array2::zeros((guess_range, 1)),
            rank_traces: 0,
            leakage_func: f,
            len_leakages: 0,
        }
    }

    /// # Panics
    /// Panic in debug if `trace.shape()[0] != self.len_samples`.
    pub fn update<T>(&mut self, trace: ArrayView1<T>, metadata: M)
    where
        T: Into<f32> + Copy,
    {
        debug_assert_eq!(trace.shape()[0], self.len_samples);

        /* This function updates the internal arrays of the DPA
        It accepts trace_batch and plaintext_batch to update them*/
        for guess in 0..self.guess_range {
            let index = (self.leakage_func)(metadata.clone(), guess);
            if index & 1 == 1 {
                // classification is performed based on the lsb
                // let tmp_row: Array1<f32> = self.sum_1.row(guess).to_owned() + tmp_trace.clone();
                // self.sum_1.row_mut(guess).assign(&tmp_row);
                for i in 0..self.len_samples {
                    self.sum_1[[guess, i]] += trace[i].into();
                }
                self.count_1[guess] += 1;
            } else {
                // let tmp_row: Array1<f32> = self.sum_0.row(guess).to_owned() + tmp_trace.clone();
                // self.sum_0.row_mut(guess).assign(&tmp_row);

                for i in 0..self.len_samples {
                    self.sum_0[[guess, i]] += trace[i].into();
                }
                self.count_0[guess] += 1;
            }
        }
        self.len_leakages += 1;
    }

    pub fn update_success<T>(&mut self, trace_batch: ArrayView1<T>, plaintext_batch: M)
    where
        T: Into<f32> + Copy,
    {
        /* This function updates the main arrays of the DPA for the success rate*/
        self.update(trace_batch, plaintext_batch);

        if self.len_leakages % self.rank_traces == 0 {
            let dpa = self.finalize();

            if self.len_leakages == self.rank_traces {
                self.rank_slice = dpa
                    .max_corr
                    .clone()
                    .into_shape((dpa.max_corr.shape()[0], 1))
                    .unwrap();
            } else {
                self.rank_slice = concatenate![
                    Axis(1),
                    self.rank_slice,
                    dpa.max_corr
                        .clone()
                        .into_shape((dpa.max_corr.shape()[0], 1))
                        .unwrap()
                ];
            }
        }
    }

    pub fn assign_rank_traces(&mut self, value: usize) {
        self.rank_traces = value;
    }

    pub fn finalize(&mut self) -> Dpa {
        /* This function finalizes the calculation after feeding all stored acc arrays */
        let mut tmp_avg_0 = Array2::zeros((self.guess_range, self.len_samples));
        let mut tmp_avg_1 = Array2::zeros((self.guess_range, self.len_samples));

        for row in 0..self.guess_range {
            let tmp_row_0 = self.sum_0.row(row).to_owned() / self.count_0[row] as f32;
            let tmp_row_1 = self.sum_1.row(row).to_owned() / self.count_1[row] as f32;
            tmp_avg_0.row_mut(row).assign(&tmp_row_0);
            tmp_avg_1.row_mut(row).assign(&tmp_row_1);
        }

        let corr = (tmp_avg_0 - tmp_avg_1).map(|e| f32::abs(*e));
        let max_corr = max_per_row(corr.view());

        Dpa {
            guess_range: self.guess_range,
            corr,
            max_corr,
        }
    }

    pub fn success_traces(&mut self, traces_no: usize) {
        self.rank_traces = traces_no;
    }

    pub fn pass_rank(&self) -> ArrayView2<f32> {
        self.rank_slice.view()
    }
}

impl<M> Add for DpaProcessor<M> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.len_samples, rhs.len_samples);
        debug_assert_eq!(self.guess_range, rhs.guess_range);
        debug_assert_eq!(self.leakage_func, rhs.leakage_func);

        Self {
            len_samples: self.len_samples,
            guess_range: self.guess_range,
            sum_0: self.sum_0 + rhs.sum_0,
            sum_1: self.sum_1 + rhs.sum_1,
            count_0: self.count_0 + rhs.count_0,
            count_1: self.count_1 + rhs.count_1,
            rank_slice: self.rank_slice,
            rank_traces: self.rank_traces,
            leakage_func: self.leakage_func,
            len_leakages: self.len_leakages + rhs.len_leakages,
        }
    }
}
