use ndarray::{concatenate, s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::ops::Add;

pub struct Cpa {
    sum_leakages: Array1<usize>,
    sig_leakages: Array1<usize>,
    sum_keys: Array1<usize>,
    sig_keys: Array1<usize>,
    values: Array1<usize>,
    a_l: Array2<usize>,
    target_byte: i32,
    len_leakages: usize,
    guess_range: i32,
    corr: Array2<f32>,
    max_corr: Array2<f32>,
    rank_slice: Array2<f32>,
    leakage_func: fn(usize, usize) -> usize,
    len_samples: usize,
}

/* This class implements the CPA shown in this paper: https://eprint.iacr.org/2013/794.pdf */
impl Cpa {
    pub fn new(
        size: usize,
        guess_range: i32,
        target_byte: i32,
        f: fn(usize, usize) -> usize,
    ) -> Self {
        Self {
            len_samples: size,
            a_l: Array2::zeros((guess_range as usize, size)),
            target_byte: target_byte,
            guess_range: guess_range,
            sum_leakages: Array1::zeros(size),
            sig_leakages: Array1::zeros(size),
            sum_keys: Array1::zeros(guess_range as usize),
            sig_keys: Array1::zeros(guess_range as usize),
            values: Array1::zeros(guess_range as usize),
            corr: Array2::zeros((guess_range as usize, size)),
            max_corr: Array2::zeros((guess_range as usize, 1)),
            rank_slice: Array2::zeros((guess_range as usize, 1)),
            leakage_func: f,
            len_leakages: 0,
        }
    }

    pub fn update(&mut self, trace: Array1<usize>, plaintext: Array1<usize>) {
        /* This function updates the main arrays of the CPA, as shown in Alg. 4
        in the paper.*/
        self.len_leakages += 1;
        self.gen_values(plaintext.clone(), self.guess_range, self.target_byte);
        self.go(trace, plaintext.clone(), self.guess_range);
    }

    pub fn gen_values(&mut self, metadata: Array1<usize>, _guess_range: i32, _target_key: i32) {
        for guess in 0.._guess_range {
            self.values[guess as usize] =
                (self.leakage_func)(metadata[_target_key as usize], guess as usize) as usize;
        }
    }

    pub fn go(&mut self, _trace: Array1<usize>, metadata: Array1<usize>, _guess_range: i32) {
        for i in 0..self.len_samples {
            self.sum_leakages[i] += _trace[i] as usize;
            self.sig_leakages[i] += (_trace[i] * _trace[i]) as usize;
        }

        for guess in 0.._guess_range {
            self.sum_keys[guess as usize] += self.values[guess as usize] as usize;
            self.sig_keys[guess as usize] +=
                (self.values[guess as usize] * self.values[guess as usize]) as usize;
        }
        let partition: usize = metadata[self.target_byte as usize] as usize;
        for i in 0..self.len_samples {
            self.a_l[[partition, i]] += _trace[i] as usize;
        }
    }

    pub fn finalize(&mut self) {
        /* This function finalizes the calculation after feeding the
        overall traces */

        let shape_p = self.guess_range as usize;
        let mut p: ndarray::ArrayBase<ndarray::OwnedRepr<usize>, ndarray::Dim<[usize; 2]>> =
            Array2::zeros((shape_p, shape_p));
        for i in 0..self.guess_range {
            for x in 0..self.guess_range {
                p[[x as usize, i as usize]] = (self.leakage_func)(x as usize, i as usize) as usize;
            }
        }
        for i in 0..self.guess_range {
            let _sigkeys = self.sig_keys[i as usize] as f32 / self.len_leakages as f32;
            let _sumkeys = self.sum_keys[i as usize] as f32 / self.len_leakages as f32;
            let lower1: f32 = _sigkeys - (_sumkeys * _sumkeys);
            /* Parallel operation using multi-threading */
            let tmp: Vec<f32> = (0..self.len_samples)
                .into_par_iter()
                .map(|x| {
                    let _sumleakages =
                        self.sum_leakages[x as usize] as f32 / self.len_leakages as f32;
                    let _sigleakages =
                        self.sig_leakages[x as usize] as f32 / self.len_leakages as f32;
                    let slice_a = self.a_l.slice(s![.., x]);
                    let slice_b = p.slice(s![.., i]);
                    let summult: i32 = self.sum_mult(slice_a, slice_b);
                    let upper1: f32 = summult as f32 / self.len_leakages as f32;
                    let upper: f32 = upper1 - (_sumkeys * _sumleakages);
                    let lower2: f32 = _sigleakages - (_sumleakages * _sumleakages);
                    let lower = f32::sqrt(lower1 * lower2);
                    f32::abs(upper / lower)
                })
                .collect();

            for z in 0..self.len_samples {
                self.corr[[i as usize, z]] = tmp[z];
            }
        }
        self.calculation();
    }

    pub fn calculation(&mut self) {
        // let mut max_256: Array2<f32> = Array2::zeros((self._guess_range as usize, 1));
        for i in 0..self.guess_range {
            let row = self.corr.row(i as usize);
            // Calculating the max value in the row
            let max_value = row.into_iter().reduce(|a, b| {
                let mut tmp = a;
                if tmp < b {
                    tmp = b;
                }
                tmp
            }).unwrap();
            self.max_corr[[i as usize, 0]] = *max_value;
        }
        self.rank_slice = concatenate![Axis(1), self.rank_slice, self.max_corr];
    }

    pub fn pass_rank(&self) -> ArrayView2<f32> {
        self.rank_slice.slice(s![.., 1..])
    }

    pub fn pass_corr_array(&self) -> Array2<f32> {
        self.corr.clone()
    }

    pub fn pass_guess(&self) -> i32 {
        let mut init_value: f32 = 0.0;
        let mut guess: i32 = 0;
        for i in 0..self.guess_range {
            if self.max_corr[[i as usize, 0]] > init_value {
                init_value = self.max_corr[[i as usize, 0]];
                guess = i;
            }
        }

        guess
    }

    fn sum_mult(&self, a: ArrayView1<usize>, b: ArrayView1<usize>) -> i32 {
        a.dot(&b) as i32
    }
}

impl Add for Cpa {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            sum_leakages: self.sum_leakages + rhs.sum_leakages,
            sig_leakages: self.sig_leakages + rhs.sig_leakages,
            sum_keys: self.sum_keys + rhs.sum_keys,
            sig_keys: self.sig_keys + rhs.sig_keys,
            values: self.values + rhs.values,
            a_l: self.a_l + rhs.a_l,
            target_byte: rhs.target_byte,
            len_leakages: self.len_leakages + rhs.len_leakages,
            guess_range: rhs.guess_range,
            corr: self.corr + rhs.corr,
            max_corr: self.max_corr,
            rank_slice: self.rank_slice,
            len_samples: rhs.len_samples,
            leakage_func: self.leakage_func,
        }
    }
}
