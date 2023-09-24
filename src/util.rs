//! Convenient utility functions.

use std::{
    fs::File,
    io::{self, BufWriter},
    time::Duration,
};

use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array, Array1, ArrayView2, Array2};
use npyz::{Deserialize, NpyFile, WriterBuilder};
use ndarray_npy::{ReadableElement, WriteNpyExt, ReadNpyExt};

/// Writes an ndarray in npy format.
///
/// This code comes from the npyz crate documentation:
/// https://docs.rs/npyz/latest/npyz/#working-with-ndarray
pub fn write_array<T, S, D>(
    writer: impl io::Write,
    array: &ndarray::ArrayBase<S, D>,
) -> io::Result<()>
where
    T: Clone + npyz::AutoSerialize,
    S: ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
{
    let shape: Vec<u64> = array.shape().iter().map(|&x| x as u64).collect();
    let c_order_items = array.iter();
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&shape)
        .writer(writer)
        .begin_nd()?;
    writer.extend(c_order_items)?;
    writer.finish()
}

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
    T: Clone + npyz::AutoSerialize,
    S: ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
>(
    path: &str,
    array: &ndarray::ArrayBase<S, D>,
) -> io::Result<()> {
    write_array(BufWriter::new(File::create(path).unwrap()), array)
}



/// Creates a [`ProgressBar`] with a predefined default style.
pub fn progress_bar(len: usize) -> ProgressBar {
    let progress_bar = ProgressBar::new(len as u64).with_style(
        ProgressStyle::with_template("{elapsed_precise} {wide_bar} {pos}/{len} ({eta})").unwrap(),
    );
    progress_bar.enable_steady_tick(Duration::new(0, 100000000));
    progress_bar
}


pub fn read_array_2_from_npy_file<T: ReadableElement> (dir: &str)-> Array2<T>{
    let reader: File = File::open(dir).unwrap();
    let arr: Array2<T> = Array2::<T>::read_npy(reader).unwrap();
    arr
}           


pub fn save_array2(path: &str, ar: ArrayView2<f32>){
    let writer = BufWriter::new(File::create(path).unwrap());
    let _ = ar.write_npy(writer);
}
