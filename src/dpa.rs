use ndarray::{concatenate, Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::ops::Add;

use crate::util::max_per_row;

pub struct Dpa<T> {
    /* List of internal class variables */
    sum_0: Array2<f32>,
    sum_1: Array2<f32>,
    count_0: Array1<usize>,
    count_1: Array1<usize>,
    guess_range: i32,
    corr: Array2<f32>,
    max_corr: Array1<f32>,
    rank_slice: Array2<f32>,
    leakage_func: fn(T, usize) -> usize,
    len_samples: usize,
    rank_traces: usize, // Number of traces to calculate succes rate
    len_leakages: usize,
}

/* This class implements the DPA algorithm shown in:
https://paulkocher.com/doc/DifferentialPowerAnalysis.pdf
https://web.mit.edu/6.857/OldStuff/Fall03/ref/kocher-DPATechInfo.pdf */

impl<T: Clone> Dpa<T> {
    pub fn new(size: usize, guess_range: i32, f: fn(T, usize) -> usize) -> Self {
        Self {
            len_samples: size,
            guess_range, //fixing clippy warning
            sum_0: Array2::zeros((guess_range as usize, size)),
            sum_1: Array2::zeros((guess_range as usize, size)),
            count_0: Array1::zeros(guess_range as usize),
            count_1: Array1::zeros(guess_range as usize),
            corr: Array2::zeros((guess_range as usize, size)),
            max_corr: Array1::zeros(guess_range as usize),
            rank_slice: Array2::zeros((guess_range as usize, 1)),
            leakage_func: f,
            rank_traces: 0,
            len_leakages: 0,
        }
    }

    /// # Panics
    /// Panic in debug if `trace.shape()[0] != self.len_samples`.
    pub fn update<U>(&mut self, trace: ArrayView1<U>, metadata: T)
    where
        U: Into<f32> + Copy,
    {
        debug_assert_eq!(trace.shape()[0], self.len_samples);

        /* This function updates the internal arrays of the DPA
        It accepts trace_batch and plaintext_batch to update them*/
        for guess in 0..self.guess_range as i16 {
            let index = (self.leakage_func)(metadata.clone(), guess as usize);
            if index & 1 == 1 {
                // classification is performed based on the lsb
                // let tmp_row: Array1<f32> = self.sum_1.row(guess as usize).to_owned() + tmp_trace.clone();
                // self.sum_1.row_mut(guess as usize).assign(&tmp_row);
                for i in 0..self.len_samples {
                    self.sum_1[[guess as usize, i]] += trace[i].into();
                }
                self.count_1[guess as usize] += 1;
            } else {
                // let tmp_row: Array1<f32> = self.sum_0.row(guess as usize).to_owned() + tmp_trace.clone();
                // self.sum_0.row_mut(guess as usize).assign(&tmp_row);

                for i in 0..self.len_samples {
                    self.sum_0[[guess as usize, i]] += trace[i].into();
                }
                self.count_0[guess as usize] += 1;
            }
        }
        self.len_leakages += 1;
    }

    pub fn update_success<U>(&mut self, trace_batch: ArrayView1<U>, plaintext_batch: T)
    where
        U: Into<f32> + Copy,
    {
        /* This function updates the main arrays of the DPA for the success rate*/
        self.update(trace_batch, plaintext_batch);

        if self.len_leakages % self.rank_traces == 0 {
            self.finalize();

            if self.len_leakages == self.rank_traces {
                self.rank_slice = self
                    .max_corr
                    .clone()
                    .into_shape((self.max_corr.shape()[0], 1))
                    .unwrap();
            } else {
                self.rank_slice = concatenate![
                    Axis(1),
                    self.rank_slice,
                    self.max_corr
                        .clone()
                        .into_shape((self.max_corr.shape()[0], 1))
                        .unwrap()
                ];
            }
        }
    }

    pub fn assign_rank_traces(&mut self, value: usize) {
        self.rank_traces = value;
    }

    pub fn finalize(&mut self) {
        /* This function finalizes the calculation after
        feeding all stored acc arrays */
        let mut tmp_avg_0 = Array2::zeros((self.guess_range as usize, self.len_samples));
        let mut tmp_avg_1 = Array2::zeros((self.guess_range as usize, self.len_samples));

        for row in 0..self.guess_range as usize {
            let tmp_row_0 = self.sum_0.row(row).to_owned() / self.count_0[row] as f32;
            let tmp_row_1 = self.sum_1.row(row).to_owned() / self.count_1[row] as f32;
            tmp_avg_0.row_mut(row).assign(&tmp_row_0);
            tmp_avg_1.row_mut(row).assign(&tmp_row_1);
        }
        let diff = tmp_avg_0.clone() - tmp_avg_1;

        self.corr = diff.map(|e| f32::abs(*e));

        self.max_corr = max_per_row(self.corr.view());
    }

    pub fn success_traces(&mut self, traces_no: usize) {
        self.rank_traces = traces_no;
    }

    pub fn pass_rank(&self) -> ArrayView2<f32> {
        self.rank_slice.view()
    }

    pub fn pass_corr_array(&self) -> ArrayView2<f32> {
        self.corr.view()
    }

    pub fn pass_guess(&self) -> i32 {
        let mut init_value: f32 = 0.0;
        let mut guess: i32 = 0;
        for i in 0..self.guess_range {
            if self.max_corr[i as usize] > init_value {
                init_value = self.max_corr[i as usize];
                guess = i;
            }
        }
        guess
    }
}

impl<T> Add for Dpa<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.len_samples, rhs.len_samples);
        debug_assert_eq!(self.guess_range, rhs.guess_range);
        debug_assert_eq!(self.leakage_func, rhs.leakage_func);

        Self {
            sum_0: self.sum_0 + rhs.sum_0,
            sum_1: self.sum_1 + rhs.sum_1,
            count_0: self.count_0 + rhs.count_0,
            count_1: self.count_1 + rhs.count_1,
            guess_range: self.guess_range,
            corr: self.corr + rhs.corr,
            max_corr: self.max_corr,
            rank_slice: self.rank_slice,
            len_samples: self.len_samples,
            leakage_func: self.leakage_func,
            rank_traces: self.rank_traces,
            len_leakages: self.len_leakages + rhs.len_leakages,
        }
    }
}
