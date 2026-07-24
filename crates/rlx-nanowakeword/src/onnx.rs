use anyhow::{Context, Result};
use std::path::Path;

pub struct OnnxNanoModel {
    session: ort::session::Session,
}

impl OnnxNanoModel {
    pub fn load(path: &Path) -> Result<Self> {
        let session = ort::session::Session::builder()?
            .commit_from_file(path)
            .with_context(|| format!("load {}", path.display()))?;
        Ok(Self { session })
    }

    pub fn try_load(path: &Path) -> Option<Self> {
        Self::load(path).ok()
    }

    pub fn ok(&self) -> bool {
        !self.session.inputs().is_empty()
    }
}
