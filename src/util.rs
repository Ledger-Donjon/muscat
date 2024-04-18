//! Convenient utility functions.

use std::{io::Read, path::Path};

use ndarray::{Array, Array1, Array2, ArrayView2};
use ndarray_npy::{read_npy, write_npy, ReadNpyError, ReadableElement, WriteNpyError};
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

/// Writes an [`ndarray::ArrayBase`] to a new file, in npy format.
///
/// # Arguments
///
/// * `path` - Output file path. If a file already exists, it is overwritten.
/// * `array` - Array to be saved.
pub fn save_array<
    T: ndarray_npy::WritableElement,
    S: ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
>(
    path: impl AsRef<Path>,
    array: &ndarray::ArrayBase<S, D>,
) -> Result<(), WriteNpyError> {
    write_npy(path, array)
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

pub fn read_array2_from_npy_file<T: ReadableElement>(
    path: impl AsRef<Path>,
) -> Result<Array2<T>, ReadNpyError> {
    read_npy(path)
}

pub fn save_array2(path: impl AsRef<Path>, array: ArrayView2<f32>) -> Result<(), WriteNpyError> {
    write_npy(path, &array)
}
