use crate::util::max_per_row;
use ndarray::{concatenate, s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::{
    iter::ParallelBridge,
    prelude::{IntoParallelIterator, ParallelIterator},
};
use std::{iter::zip, ops::Add};

/// Computes the [`Cpa`] of the given traces using [`CpaProcessor`].
///
/// # Panics
/// - Panic if `leakages.shape()[0] != plaintexts.shape()[0]`
/// - Panic if `chunk_size` is 0.
pub fn cpa<T>(
    leakages: ArrayView2<T>,
    plaintexts: ArrayView2<T>,
    guess_range: usize,
    target_byte: usize,
    leakage_func: fn(usize, usize) -> usize,
    chunk_size: usize,
) -> Cpa
where
    T: Into<usize> + Copy + Sync,
{
    assert_eq!(leakages.shape()[0], plaintexts.shape()[0]);
    assert!(chunk_size > 0);

    zip(
        leakages.axis_chunks_iter(Axis(0), chunk_size),
        plaintexts.axis_chunks_iter(Axis(0), chunk_size),
    )
    .par_bridge()
    .map(|(leakages_chunk, plaintexts_chunk)| {
        let mut cpa =
            CpaProcessor::new(leakages.shape()[1], guess_range, target_byte, leakage_func);

        for i in 0..leakages_chunk.shape()[0] {
            cpa.update(leakages_chunk.row(i), plaintexts_chunk.row(i));
        }

        cpa
    })
    .reduce(
        || CpaProcessor::new(leakages.shape()[1], guess_range, target_byte, leakage_func),
        |a, b| a + b,
    )
    .finalize()
}

pub struct Cpa {
    /// Guess range upper excluded bound
    guess_range: usize,
    /// Pearson correlation coefficients
    corr: Array2<f32>,
    /// Max pearson correlation coefficients
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
    target_byte: usize,
    /// Guess range upper excluded bound
    guess_range: usize,
    /// Sum of traces
    sum_leakages: Array1<usize>,
    /// Sum of square of traces
    sum_squares_leakages: Array1<usize>,
    /// Sum of traces per key guess
    guess_sum_leakages: Array1<usize>,
    /// Sum of square of traces per key guess
    guess_sum_squares_leakages: Array1<usize>,
    a_l: Array2<usize>,
    /// Leakage model
    leakage_func: fn(usize, usize) -> usize,
    /// Number of traces processed
    len_leakages: usize,
}

/* This class implements the CPA shown in this paper: https://eprint.iacr.org/2013/794.pdf */
impl CpaProcessor {
    pub fn new(
        size: usize,
        guess_range: usize,
        target_byte: usize,
        leakage_func: fn(usize, usize) -> usize,
    ) -> Self {
        Self {
            len_samples: size,
            target_byte,
            guess_range,
            sum_leakages: Array1::zeros(size),
            sum_squares_leakages: Array1::zeros(size),
            guess_sum_leakages: Array1::zeros(guess_range),
            guess_sum_squares_leakages: Array1::zeros(guess_range),
            a_l: Array2::zeros((guess_range, size)),
            leakage_func,
            len_leakages: 0,
        }
    }

    /// # Panics
    /// Panic in debug if `trace.shape()[0] != self.len_samples`.
    pub fn update<T>(&mut self, trace: ArrayView1<T>, plaintext: ArrayView1<T>)
    where
        T: Into<usize> + Copy,
    {
        debug_assert_eq!(trace.shape()[0], self.len_samples);

        /* This function updates the main arrays of the CPA, as shown in Alg. 4
        in the paper.*/

        for i in 0..self.len_samples {
            self.sum_leakages[i] += trace[i].into();
            self.sum_squares_leakages[i] += trace[i].into() * trace[i].into();
        }

        for guess in 0..self.guess_range {
            let value = (self.leakage_func)(plaintext[self.target_byte].into(), guess);
            self.guess_sum_leakages[guess] += value;
            self.guess_sum_squares_leakages[guess] += value * value;
        }

        let partition = plaintext[self.target_byte].into();
        for i in 0..self.len_samples {
            self.a_l[[partition, i]] += trace[i].into();
        }

        self.len_leakages += 1;
    }

    pub fn finalize(&self) -> Cpa {
        /* This function finalizes the calculation after feeding the
        overall traces */
        let mut p = Array2::zeros((self.guess_range, self.guess_range));
        for guess in 0..self.guess_range {
            for x in 0..self.guess_range {
                p[[x, guess]] = (self.leakage_func)(x, guess);
            }
        }

        let mut corr = Array2::zeros((self.guess_range, self.len_samples));
        for guess in 0..self.guess_range {
            let mean_key = self.guess_sum_leakages[guess] as f32 / self.len_leakages as f32;
            let mean_squares_key =
                self.guess_sum_squares_leakages[guess] as f32 / self.len_leakages as f32;
            let var_key = mean_squares_key - (mean_key * mean_key);

            /* Parallel operation using multi-threading */
            let tmp: Vec<f32> = (0..self.len_samples)
                .into_par_iter()
                .map(|x| {
                    let mean_leakages = self.sum_leakages[x] as f32 / self.len_leakages as f32;
                    let summult = self.sum_mult(self.a_l.slice(s![.., x]), p.slice(s![.., guess]));
                    let upper1 = summult as f32 / self.len_leakages as f32;
                    let upper = upper1 - (mean_key * mean_leakages);

                    let mean_squares_leakages =
                        self.sum_squares_leakages[x] as f32 / self.len_leakages as f32;
                    let var_leakages = mean_squares_leakages - (mean_leakages * mean_leakages);
                    let lower = f32::sqrt(var_key * var_leakages);

                    f32::abs(upper / lower)
                })
                .collect();

            #[allow(clippy::needless_range_loop)]
            for u in 0..self.len_samples {
                corr[[guess, u]] = tmp[u];
            }
        }

        let max_corr = max_per_row(corr.view());

        let mut rank_slice = Array2::zeros((self.guess_range, 1));
        rank_slice = concatenate![
            Axis(1),
            rank_slice,
            max_corr
                .clone()
                .into_shape((max_corr.shape()[0], 1))
                .unwrap()
        ];

        Cpa {
            guess_range: self.guess_range,
            corr,
            max_corr,
            rank_slice,
        }
    }

    fn sum_mult(&self, a: ArrayView1<usize>, b: ArrayView1<usize>) -> i32 {
        a.dot(&b) as i32
    }
}

impl Add for CpaProcessor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.target_byte, rhs.target_byte);
        debug_assert_eq!(self.guess_range, rhs.guess_range);
        debug_assert_eq!(self.len_samples, rhs.len_samples);
        debug_assert_eq!(self.leakage_func, rhs.leakage_func);

        Self {
            len_samples: self.len_samples,
            target_byte: self.target_byte,
            guess_range: self.guess_range,
            sum_leakages: self.sum_leakages + rhs.sum_leakages,
            sum_squares_leakages: self.sum_squares_leakages + rhs.sum_squares_leakages,
            guess_sum_leakages: self.guess_sum_leakages + rhs.guess_sum_leakages,
            guess_sum_squares_leakages: self.guess_sum_squares_leakages
                + rhs.guess_sum_squares_leakages,
            a_l: self.a_l + rhs.a_l,
            leakage_func: self.leakage_func,
            len_leakages: self.len_leakages + rhs.len_leakages,
        }
    }
}
