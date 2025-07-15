use numpy::{
    IntoPyArray, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods,
    array::{PyArray1, PyArray2},
    dtype,
    ndarray::Array2,
};
use pyo3::types::PyFunction;
use pyo3::{exceptions::PyTypeError, prelude::*};

#[pyfunction]
pub fn compute_cpa<'py>(
    traces: &Bound<'py, PyUntypedArray>,
    plaintexts: &Bound<'py, PyArray2<usize>>,
    guess_range: usize,
    target_byte: usize,
    leakage_func: &Bound<'py, PyFunction>,
    batch_size: usize,
) -> PyResult<Cpa> {
    let mut modeled_leakages = Array2::zeros((guess_range, 256));
    for guess in 0..guess_range {
        for plaintext in 0..256 {
            modeled_leakages[[guess, plaintext]] = leakage_func
                .call((plaintext, guess), None)
                .unwrap()
                .extract::<usize>()
                .unwrap();
        }
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

                    return Ok(Cpa(muscat::distinguishers::cpa::cpa(
                        traces.readonly().as_array(),
                        plaintexts.readonly().as_array(),
                        guess_range,
                        target_byte,
                        |guess, plaintext| modeled_leakages[[guess, plaintext]],
                        batch_size,
                    )));
                }
            )*
        };
    }

    type_dispatch! { u8, u16, u32, u64, i8, i16, i32, i64, f32 }

    Err(PyTypeError::new_err(format!(
        "Unsupported traces dtype: {sample_dtype}."
    )))
}

#[pyfunction]
pub fn compute_cpa_normal<'py>(
    traces: &Bound<'py, PyUntypedArray>,
    plaintexts: &Bound<'py, PyArray2<usize>>,
    guess_range: usize,
    target_byte: usize,
    leakage_func: &Bound<'py, PyFunction>,
    batch_size: usize,
) -> PyResult<Cpa> {
    let mut modeled_leakages = Array2::zeros((guess_range, 256));
    for guess in 0..guess_range {
        for plaintext in 0..256 {
            modeled_leakages[[guess, plaintext]] = leakage_func
                .call((plaintext, guess), None)
                .unwrap()
                .extract::<usize>()
                .unwrap();
        }
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

                    return Ok(Cpa(muscat::distinguishers::cpa_normal::cpa(
                        traces.readonly().as_array(),
                        plaintexts.readonly().as_array(),
                        guess_range,
                        |plaintext, guess| modeled_leakages[[guess, plaintext[target_byte]]],
                        batch_size,
                    )));
                }
            )*
        };
    }

    type_dispatch! { u8, u16, u32, u64, i8, i16, i32, i64, f32 }

    Err(PyTypeError::new_err(format!(
        "Unsupported traces dtype: {sample_dtype}."
    )))
}

#[pyclass]
pub struct Cpa(muscat::distinguishers::cpa::Cpa);

#[pymethods]
impl Cpa {
    pub fn rank<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        self.0.rank().into_pyarray(py)
    }

    pub fn corr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        self.0.corr().to_owned().into_pyarray(py)
    }

    pub fn best_guess(&self) -> usize {
        self.0.best_guess()
    }

    pub fn max_corr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.0.max_corr().into_pyarray(py)
    }
}

#[pyfunction]
pub fn compute_dpa<'py>(
    traces: &Bound<'py, PyUntypedArray>,
    plaintexts: &Bound<'py, PyArray1<usize>>,
    guess_range: usize,
    selection_function: &Bound<'py, PyFunction>,
    batch_size: usize,
) -> PyResult<Dpa> {
    let mut selections = Array2::zeros((256, guess_range));
    for p in 0..selections.shape()[0] {
        for guess in 0..selections.shape()[1] {
            if selection_function
                .call((p, guess), None)
                .unwrap()
                .extract::<bool>()
                .unwrap()
            {
                selections[[p, guess]] = 1;
            } else {
                selections[[p, guess]] = 0;
            }
        }
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

                    return Ok(Dpa(muscat::distinguishers::dpa::dpa(
                        traces.readonly().as_array(),
                        plaintexts.readonly().as_array(),
                        guess_range,
                        |p, guess| selections[[p, guess]] == 1,
                        batch_size,
                    )));
                }
            )*
        };
    }

    type_dispatch! { u8, u16, u32, u64, i8, i16, i32, i64, f32 }

    Err(PyTypeError::new_err(format!(
        "Unsupported traces dtype: {sample_dtype}."
    )))
}

#[pyclass]
pub struct Dpa(muscat::distinguishers::dpa::Dpa);

#[pymethods]
impl Dpa {
    pub fn rank<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        self.0.rank().into_pyarray(py)
    }

    pub fn differential_curves<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        self.0.differential_curves().to_owned().into_pyarray(py)
    }

    pub fn best_guess(&self) -> usize {
        self.0.best_guess()
    }

    pub fn max_differential_curves<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.0.max_differential_curves().into_pyarray(py)
    }
}

#[pymodule]
pub fn distinguishers(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_cpa, m)?)?;
    m.add_function(wrap_pyfunction!(compute_cpa_normal, m)?)?;
    m.add_function(wrap_pyfunction!(compute_dpa, m)?)?;

    m.add_class::<Cpa>()?;
    m.add_class::<Dpa>()?;

    Ok(())
}
