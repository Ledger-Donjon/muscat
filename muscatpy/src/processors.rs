use muscat::processors::MeanVar;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

#[pyfunction]
pub fn compute_mean<'py>(traces: &Bound<'py, PyArray2<i64>>) -> Bound<'py, PyArray1<f32>> {
    let py = traces.py();

    let traces_ro = traces.readonly();
    let traces = traces_ro.as_array();

    let mut meanvar = MeanVar::new(traces.shape()[1]);
    for trace in traces.rows() {
        meanvar.process(trace);
    }

    meanvar.mean().into_pyarray(py)
}

#[pyfunction]
pub fn compute_var<'py>(traces: &Bound<'py, PyArray2<i64>>) -> Bound<'py, PyArray1<f32>> {
    let py = traces.py();

    let traces_ro = traces.readonly();
    let traces = traces_ro.as_array();

    let mut meanvar = MeanVar::new(traces.shape()[1]);
    for trace in traces.rows() {
        meanvar.process(trace);
    }

    meanvar.var().into_pyarray(py)
}

#[pymodule]
pub fn processors(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_mean, m)?)?;
    m.add_function(wrap_pyfunction!(compute_var, m)?)?;

    Ok(())
}
