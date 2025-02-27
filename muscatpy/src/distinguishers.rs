use numpy::{
    array::{PyArray1, PyArray2},
    ndarray::Array2,
    IntoPyArray, PyArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::PyFunction;

#[pyfunction]
pub fn compute_cpa<'py>(
    leakages: &Bound<'py, PyArray2<usize>>,
    plaintexts: &Bound<'py, PyArray2<usize>>,
    guess_range: usize,
    target_byte: usize,
    leakage_func: &Bound<'py, PyFunction>,
    batch_size: usize,
) -> Cpa {
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

    Cpa(muscat::distinguishers::cpa::cpa(
        leakages.readonly().as_array(),
        plaintexts.readonly().as_array(),
        guess_range,
        target_byte,
        |guess, plaintext| modeled_leakages[[guess, plaintext]],
        batch_size,
    ))
}

#[pyfunction]
pub fn compute_cpa_normal<'py>(
    leakages: &Bound<'py, PyArray2<f32>>,
    plaintexts: &Bound<'py, PyArray2<usize>>,
    guess_range: usize,
    target_byte: usize,
    leakage_func: &Bound<'py, PyFunction>,
    batch_size: usize,
) -> Cpa {
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

    Cpa(muscat::distinguishers::cpa_normal::cpa(
        leakages.readonly().as_array(),
        plaintexts.readonly().as_array(),
        guess_range,
        |plaintext, guess| modeled_leakages[[guess, plaintext[target_byte]]],
        batch_size,
    ))
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
    leakages: &Bound<'py, PyArray2<f32>>,
    plaintexts: &Bound<'py, PyArray1<usize>>,
    guess_range: usize,
    selection_function: &Bound<'py, PyFunction>,
    batch_size: usize,
) -> Dpa {
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

    Dpa(muscat::distinguishers::dpa::dpa(
        leakages.readonly().as_array(),
        plaintexts.readonly().as_array(),
        guess_range,
        |p, guess| selections[[p, guess]] == 1,
        batch_size,
    ))
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
