use pyo3::prelude::*;

mod aes;

#[pymodule]
pub fn leakage_model(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let aes_module = PyModule::new(py, "aes")?;
    aes::aes(py, &aes_module)?;
    m.add_submodule(&aes_module)?;

    Ok(())
}
