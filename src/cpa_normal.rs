use ndarray::{concatenate, Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::ops::Add;
pub struct Cpa {
    /* List of internal class variables */
    sum_leakages: Array1<f32>,
    sum2_leakages: Array1<f32>,
    sum_keys: Array1<f32>,
    sum2_keys: Array1<f32>,
    values: Array2<f32>,
    len_leakages: usize,
    guess_range: usize,
    cov: Array2<f32>,
    corr: Array2<f32>,
    max_corr: Array2<f32>,
    rank_slice: Array2<f32>,
    leakage_func: fn(ArrayView1<usize>, usize) -> usize,
    len_samples: usize,
    chunk: usize,
    rank_traces: usize, // Number of traces to calculate succes rate
}

/* This class implements the CPA algorithm shown in:
https://www.iacr.org/archive/ches2004/31560016/31560016.pdf */

impl Cpa {
    pub fn new(
        size: usize,
        patch: usize,
        guess_range: usize,
        leakage_func: fn(ArrayView1<usize>, usize) -> usize,
    ) -> Self {
        Self {
            len_samples: size,
            chunk: patch,
            guess_range,
            sum_leakages: Array1::zeros(size),
            sum2_leakages: Array1::zeros(size),
            sum_keys: Array1::zeros(guess_range),
            sum2_keys: Array1::zeros(guess_range),
            values: Array2::zeros((patch, guess_range)),
            cov: Array2::zeros((guess_range, size)),
            corr: Array2::zeros((guess_range, size)),
            max_corr: Array2::zeros((guess_range, 1)),
            rank_slice: Array2::zeros((guess_range, 1)),
            leakage_func,
            len_leakages: 0,
            rank_traces: 0,
        }
    }

    pub fn update<T, U>(&mut self, trace_patch: Array2<T>, plaintext_patch: Array2<U>)
    where
        T: Into<f32> + Copy,
        U: Into<usize> + Copy,
    {
        /* This function updates the internal arrays of the CPA
        It accepts trace_patch and plaintext_patch to update them*/
        let tmp_traces = trace_patch.map(|&t| t.into());
        let metadat = plaintext_patch.map(|&m| m.into());
        self.len_leakages += self.chunk;
        self.update_values(&metadat, &tmp_traces, self.guess_range);
        self.update_key_leakages(tmp_traces, self.guess_range);
    }

    pub fn update_values(
        /* This function generates the values and cov arrays */
        &mut self,
        metadata: &Array2<usize>,
        trace: &Array2<f32>,
        guess_range: usize,
    ) {
        for row in 0..self.chunk {
            for guess in 0..guess_range {
                let pass_to_leakage: ArrayView1<usize> = metadata.row(row);
                self.values[[row, guess]] = (self.leakage_func)(pass_to_leakage, guess) as f32;
            }
        }

        self.cov = self.cov.clone() + self.values.t().dot(trace);
    }

    pub fn update_key_leakages(&mut self, trace: Array2<f32>, guess_range: usize) {
        for i in 0..self.len_samples {
            self.sum_leakages[i] += trace.column(i).sum(); // trace[i] as usize;
            self.sum2_leakages[i] += trace.column(i).dot(&trace.column(i)); // (trace[i] * trace[i]) as usize;
        }

        for guess in 0..guess_range {
            self.sum_keys[guess] += self.values.column(guess).sum(); //self.values[guess] as usize;
            self.sum2_keys[guess] += self.values.column(guess).dot(&self.values.column(guess));
            // (self.values[guess] * self.values[guess]) as usize;
        }
    }

    pub fn update_success<T: Copy, U: Copy>(
        &mut self,
        trace_patch: Array2<T>,
        plaintext_patch: Array2<U>,
    ) where
        f32: From<T>,
        usize: From<U>,
    {
        /* This function updates the main arrays of the CPA for the success rate*/
        self.update(trace_patch, plaintext_patch);
        if self.len_leakages % self.rank_traces == 0 {
            self.finalize();
            if self.len_leakages == self.rank_traces {
                self.rank_slice = self.max_corr.clone();
            } else {
                self.rank_slice = concatenate![Axis(1), self.rank_slice, self.max_corr];
            }
        }
    }

    pub fn finalize(&mut self) {
        /* This function finalizes the calculation after
        feeding all stored acc arrays */
        let cov_n = self.cov.clone() / self.len_leakages as f32;
        let avg_keys = self.sum_keys.clone() / self.len_leakages as f32;
        let std_key = self.sum2_keys.clone() / self.len_leakages as f32;
        let avg_leakages = self.sum_leakages.clone() / self.len_leakages as f32;
        let std_leakages = self.sum2_leakages.clone() / self.len_leakages as f32;

        for i in 0..self.guess_range {
            for x in 0..self.len_samples {
                let numerator = cov_n[[i, x]] - (avg_keys[i] * avg_leakages[x]);

                let denominator_1 = std_key[i] - (avg_keys[i] * avg_keys[i]);

                let denominator_2 = std_leakages[x] - (avg_leakages[x] * avg_leakages[x]);
                if numerator != 0.0 {
                    self.corr[[i, x]] =
                        f32::abs(numerator / f32::sqrt(denominator_1 * denominator_2));
                }
            }
        }
        self.select_max();
    }

    pub fn select_max(&mut self) {
        for i in 0..self.guess_range {
            let row = self.corr.row(i);
            // Calculating the max value in the row
            let max_value = row
                .into_iter()
                .reduce(|a, b| {
                    let mut tmp = a;
                    if tmp < b {
                        tmp = b;
                    }
                    tmp
                })
                .unwrap();
            self.max_corr[[i, 0]] = *max_value;
        }
    }

    pub fn success_traces(&mut self, traces_no: usize) {
        self.rank_traces = traces_no;
    }

    pub fn pass_rank(&self) -> ArrayView2<f32> {
        self.rank_slice.view()
    }

    pub fn pass_corr_array(&self) -> Array2<f32> {
        self.corr.clone()
    }

    pub fn pass_guess(&self) -> usize {
        let mut init_value = 0.0;
        let mut guess = 0;
        for i in 0..self.guess_range {
            if self.max_corr[[i, 0]] > init_value {
                init_value = self.max_corr[[i, 0]];
                guess = i;
            }
        }
        guess
    }
}

impl Add for Cpa {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            sum_leakages: self.sum_leakages + rhs.sum_leakages,
            sum2_leakages: self.sum2_leakages + rhs.sum2_leakages,
            sum_keys: self.sum_keys + rhs.sum_keys,
            sum2_keys: self.sum2_keys + rhs.sum2_keys,
            values: self.values + rhs.values,
            len_leakages: self.len_leakages + rhs.len_leakages,
            guess_range: rhs.guess_range,
            chunk: rhs.chunk,
            cov: self.cov + rhs.cov,
            corr: self.corr + rhs.corr,
            max_corr: self.max_corr,
            rank_slice: self.rank_slice,
            len_samples: rhs.len_samples,
            leakage_func: self.leakage_func,
            rank_traces: self.rank_traces,
        }
    }
}
