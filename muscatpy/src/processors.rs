use muscat::processors::MeanVar;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods, dtype,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

#[pyfunction]
pub fn compute_mean<'py>(
    traces: &Bound<'py, PyUntypedArray>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let py = traces.py();

    if traces.ndim() != 2 {
        return Err(PyTypeError::new_err(format!(
            "Invalid traces ndim: {}. traces ndim should be equal to 2.",
            traces.ndim()
        )));
    }

    let sample_dtype = traces.dtype();

    macro_rules! type_dispatch {
        ($($ty:ty),*) => {
            $(
                if sample_dtype.is_equiv_to(&dtype::<$ty>(traces.py())) {
                    let traces = traces.downcast::<PyArray2<$ty>>()?;

                    let traces_ro = traces.readonly();
                    let traces = traces_ro.as_array();

                    let mut meanvar = MeanVar::new(traces.shape()[1]);
                    for trace in traces.rows() {
                        meanvar.process(trace);
                    }

                    return Ok(meanvar.mean().into_pyarray(py));
                }
            )*
        };
    }

    type_dispatch! { u8, u16, u32, u64, i8, i16, i32, i64, f32 }

    Err(PyTypeError::new_err(format!(
        "Unsupported traces dtype: {}.",
        sample_dtype
    )))
}

#[pyfunction]
pub fn compute_var<'py>(
    traces: &Bound<'py, PyUntypedArray>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let py = traces.py();

    if traces.ndim() != 2 {
        return Err(PyTypeError::new_err(format!(
            "Invalid traces ndim: {}. traces ndim should be equal to 2.",
            traces.ndim()
        )));
    }

    let sample_dtype = traces.dtype();

    macro_rules! type_dispatch {
        ($($ty:ty),*) => {
            $(
                if sample_dtype.is_equiv_to(&dtype::<$ty>(traces.py())) {
                    let traces = traces.downcast::<PyArray2<$ty>>()?;

                    let traces_ro = traces.readonly();
                    let traces = traces_ro.as_array();

                    let mut meanvar = MeanVar::new(traces.shape()[1]);
                    for trace in traces.rows() {
                        meanvar.process(trace);
                    }

                    return Ok(meanvar.var().into_pyarray(py));
                }
            )*
        };
    }

    type_dispatch! { u8, u16, u32, u64, i8, i16, i32, i64, f32 }

    Err(PyTypeError::new_err(format!(
        "Unsupported traces dtype: {}.",
        sample_dtype
    )))
}

#[pymodule]
pub fn processors(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_mean, m)?)?;
    m.add_function(wrap_pyfunction!(compute_var, m)?)?;

    Ok(())
}
