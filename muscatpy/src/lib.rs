use pyo3::prelude::*;

mod distinguishers;
mod leakage_detection;

#[pymodule]
fn muscatpy(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let distinguishers_module = PyModule::new_bound(py, "distinguishers")?;
    distinguishers::distinguishers(py, &distinguishers_module)?;
    m.add_submodule(&distinguishers_module)?;

    let leakage_detection_module = PyModule::new_bound(py, "leakage_detection")?;
    leakage_detection::leakage_detection(py, &leakage_detection_module)?;
    m.add_submodule(&leakage_detection_module)?;

    Ok(())
}
