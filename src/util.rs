//! Convenient utility functions.

use std::{cmp::Ordering, io::Read};

use ndarray::{Array, Array1, ArrayView1, ArrayView2, Axis};
use npyz::{Deserialize, NpyFile};

#[cfg(feature = "progress_bar")]
use indicatif::{ProgressBar, ProgressStyle};
#[cfg(feature = "progress_bar")]
use std::time::Duration;

/// Reads a [`NpyFile`] as a [`Array1`]
///
/// This does the same as [`NpyFile.into_vec`] but faster, as this method reserves the resulting
/// vector to the final size directly instead of relying on `collect`.
///
/// # Panics
/// This function panics in case of IO error.
pub fn read_array1_from_npy_file<T: Deserialize, R: Read>(npy: NpyFile<R>) -> Array1<T> {
    let mut v: Vec<T> = Vec::new();
    v.reserve_exact(npy.shape()[0].try_into().unwrap());
    v.extend(npy.data().unwrap().map(|x| x.unwrap()));
    Array::from_vec(v)
}

/// Creates a [`ProgressBar`] with a predefined default style.
#[cfg(feature = "progress_bar")]
pub fn progress_bar(len: usize) -> ProgressBar {
    let progress_bar = ProgressBar::new(len as u64).with_style(
        ProgressStyle::with_template("{elapsed_precise} {wide_bar} {pos}/{len} ({eta})").unwrap(),
    );
    progress_bar.enable_steady_tick(Duration::new(0, 100000000));
    progress_bar
}

/// Return an array where the i-th element contains the maximum of the i-th row of the input array.
pub fn max_per_row(arr: ArrayView2<f32>) -> Array1<f32> {
    arr.axis_iter(Axis(0))
        .map(|row| {
            *row.into_iter()
                .reduce(|a, b| {
                    let mut tmp = a;
                    if tmp < b {
                        tmp = b;
                    }
                    tmp
                })
                .unwrap()
        })
        .collect()
}

/// Return the indices that would sort the given array with a comparison function.
pub fn argsort_by<T, F>(data: &[T], compare: F) -> Vec<usize>
where
    F: Fn(&T, &T) -> Ordering,
{
    let mut indices: Vec<usize> = (0..data.len()).collect();

    indices.sort_by(|&a, &b| compare(&data[a], &data[b]));

    indices
}

/// Return the index of the maximum value in the given array.
pub fn argmax_by<T, F>(array: ArrayView1<T>, compare: F) -> usize
where
    F: Fn(&T, &T) -> Ordering,
{
    let mut idx_max = 0;

    for i in 0..array.shape()[0] {
        if compare(&array[i], &array[idx_max]).is_gt() {
            idx_max = i;
        }
    }

    idx_max
}

/// Convert a floating point value acquired with a
/// [ChipWhisperer](https://github.com/newaetech/chipwhisperer) to integer.
///
/// # References
/// https://github.com/newaetech/chipwhisperer/blob/918e20d7a6ac7211dac61a70d233872045eb113e/software/chipwhisperer/capture/scopes/cwnano.py#L740-L744
pub fn chipwhisperer_float_to_u16(x: f64) -> u16 {
    debug_assert!((-0.5..=0.5).contains(&x));

    ((x + 1.) * 2048.) as u16
}
