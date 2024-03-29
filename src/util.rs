//! Convenient utility functions.

use std::{
    fs::File,
    io::{self, BufWriter},
    time::Duration,
};

use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array, Array1, Array2, ArrayView2};
use ndarray_npy::{write_npy, ReadNpyExt, ReadableElement, WriteNpyExt};
use npyz::{Deserialize, NpyFile, WriterBuilder};

/// Reads a [`NpyFile`] as a [`Array1`]
///
/// This does the same as [`NpyFile.into_vec`] but faster, as this method reserves the resulting
/// vector to the final size directly instead of relying on `collect`. It however panics in
/// case of IO error.
pub fn read_array_1_from_npy_file<T: Deserialize, R: std::io::Read>(npy: NpyFile<R>) -> Array1<T> {
    let mut v: Vec<T> = Vec::new();
    v.reserve_exact(npy.shape()[0] as usize);
    v.extend(npy.data().unwrap().into_iter().map(|x| x.unwrap()));
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
    path: &str,
    array: &ndarray::ArrayBase<S, D>,
) {
    // let dir = BufWriter::new(File::create(path).unwrap());
    write_npy(path, array).unwrap();
}

/// Creates a [`ProgressBar`] with a predefined default style.
pub fn progress_bar(len: usize) -> ProgressBar {
    let progress_bar = ProgressBar::new(len as u64).with_style(
        ProgressStyle::with_template("{elapsed_precise} {wide_bar} {pos}/{len} ({eta})").unwrap(),
    );
    progress_bar.enable_steady_tick(Duration::new(0, 100000000));
    progress_bar
}

pub fn read_array_2_from_npy_file<T: ReadableElement>(dir: &str) -> Array2<T> {
    let reader: File = File::open(dir).unwrap();
    let arr: Array2<T> = Array2::<T>::read_npy(reader).unwrap();
    arr
}

pub fn save_array2(path: &str, ar: ArrayView2<f32>) {
    let writer = BufWriter::new(File::create(path).unwrap());
    ar.write_npy(writer).unwrap();
}
