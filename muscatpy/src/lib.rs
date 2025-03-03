use pyo3::prelude::*;

mod distinguishers;
mod leakage_detection;
mod leakage_model;

#[pymodule]
fn muscatpy(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let distinguishers_module = PyModule::new(py, "distinguishers")?;
    distinguishers::distinguishers(py, &distinguishers_module)?;
    m.add_submodule(&distinguishers_module)?;

    let leakage_detection_module = PyModule::new(py, "leakage_detection")?;
    leakage_detection::leakage_detection(py, &leakage_detection_module)?;
    m.add_submodule(&leakage_detection_module)?;

    let leakage_model_module = PyModule::new(py, "leakage_model")?;
    leakage_model::leakage_model(py, &leakage_model_module)?;
    m.add_submodule(&leakage_model_module)?;

    Ok(())
}
