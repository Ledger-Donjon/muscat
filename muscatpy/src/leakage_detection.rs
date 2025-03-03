use numpy::{
    IntoPyArray, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods,
    array::{PyArray1, PyArray2},
    dtype,
    ndarray::Array1,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyFunction;

#[pyfunction]
pub fn compute_snr<'py>(
    traces: &Bound<'py, PyUntypedArray>,
    classes: usize,
    get_class: &Bound<'py, PyFunction>,
    batch_size: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let mut traces_class = Array1::zeros(traces.len());
    for i in 0..traces_class.len() {
        traces_class[i] = get_class
            .call((i,), None)
            .unwrap()
            .extract::<usize>()
            .unwrap();
    }

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

                    return Ok(muscat::leakage_detection::snr(
                        traces.readonly().as_array(),
                        classes,
                        |i| traces_class[i],
                        batch_size,
                    ).into_pyarray(traces.py()));
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
pub fn compute_ttest<'py>(
    traces: &Bound<'py, PyUntypedArray>,
    trace_classes: &Bound<'py, PyArray1<bool>>,
    batch_size: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
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

                    return Ok(muscat::leakage_detection::ttest(
                        traces.readonly().as_array(),
                        trace_classes.readonly().as_array(),
                        batch_size,
                    ).into_pyarray(traces.py()));
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
pub fn leakage_detection(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_snr, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ttest, m)?)?;

    Ok(())
}
