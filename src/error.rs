use std::io;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Failed to save/load muscat data")]
    SaveLoadError(#[from] serde_json::Error),
    #[error(transparent)]
    IoError(#[from] io::Error),
}
