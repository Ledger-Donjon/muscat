use numpy::{
    array::{PyArray1, PyArray2},
    ndarray::Array1,
    IntoPyArray, PyArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::PyFunction;

#[pyfunction]
pub fn compute_snr<'py>(
    leakages: &Bound<'py, PyArray2<i64>>,
    classes: usize,
    get_class: &Bound<'py, PyFunction>,
    batch_size: usize,
) -> Bound<'py, PyArray1<f32>> {
    let mut leakages_class = Array1::zeros(leakages.readonly().as_array().len());
    for i in 0..leakages_class.len() {
        leakages_class[i] = get_class
            .call((i,), None)
            .unwrap()
            .extract::<usize>()
            .unwrap();
    }

    muscat::leakage_detection::snr(
        leakages.readonly().as_array(),
        classes,
        |i| leakages_class[i],
        batch_size,
    )
    .into_pyarray(leakages.py())
}

#[pyfunction]
pub fn compute_ttest<'py>(
    traces: &Bound<'py, PyArray2<i64>>,
    trace_classes: &Bound<'py, PyArray1<bool>>,
    batch_size: usize,
) -> Bound<'py, PyArray1<f32>> {
    muscat::leakage_detection::ttest(
        traces.readonly().as_array(),
        trace_classes.readonly().as_array(),
        batch_size,
    )
    .into_pyarray(traces.py())
}

#[pymodule]
pub fn leakage_detection(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_snr, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ttest, m)?)?;

    Ok(())
}
