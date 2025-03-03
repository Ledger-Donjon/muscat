use muscat::leakage_model;
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyBytes, PyList},
};

#[pyfunction]
pub fn sbox(x: u8) -> u8 {
    leakage_model::aes::sbox(x)
}

#[pyfunction]
pub fn inv_sbox(x: u8) -> u8 {
    leakage_model::aes::inv_sbox(x)
}

#[pyfunction]
pub fn expand_key<'py>(key: &Bound<'py, PyBytes>) -> PyResult<Bound<'py, PyList>> {
    let py = key.py();

    let Ok(key) = key.as_bytes().try_into() else {
        return Err(PyTypeError::new_err("Invalid key length"));
    };

    let mut round_keys = [[0u8; 16]; 11];
    leakage_model::aes::expand_key(key, &mut round_keys);

    PyList::new(py, round_keys.map(|rk| PyBytes::new(py, &rk)))
}

#[pyclass]
pub struct State {
    state: [u8; 16],
}

#[pymethods]
impl State {
    pub fn state<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.state)
    }

    pub fn add_round_key(&mut self, round_key: &Bound<'_, PyBytes>) -> PyResult<()> {
        let Ok(round_key) = round_key.as_bytes().try_into() else {
            return Err(PyTypeError::new_err("Invalid round_key length"));
        };

        leakage_model::aes::add_round_key(&mut self.state, round_key);

        Ok(())
    }

    pub fn sub_bytes(&mut self) {
        leakage_model::aes::sub_bytes(&mut self.state);
    }

    pub fn shift_rows(&mut self) {
        leakage_model::aes::shift_rows(&mut self.state);
    }

    pub fn mix_columns(&mut self) {
        leakage_model::aes::mix_columns(&mut self.state);
    }

    pub fn inv_sub_bytes(&mut self) {
        leakage_model::aes::inv_sub_bytes(&mut self.state);
    }

    pub fn inv_mix_columns(&mut self) {
        leakage_model::aes::inv_mix_columns(&mut self.state);
    }
}

#[pymodule]
pub fn aes(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sbox, m)?)?;
    m.add_function(wrap_pyfunction!(inv_sbox, m)?)?;
    m.add_function(wrap_pyfunction!(expand_key, m)?)?;

    m.add_class::<State>()?;

    Ok(())
}
