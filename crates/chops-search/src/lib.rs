//! Library surface of the build tool. Exists so integration tests (the
//! model2vec-rs parity oracle) can reach the model loader — a pure binary
//! crate's internals are unreachable from tests/.

pub mod artifacts;
pub mod assets;
pub mod calibrate;
pub mod completion;
pub mod config;
pub mod eval;
pub mod explain;
pub mod frontmatter;
pub mod init;
pub mod knob;
pub mod model;
pub mod model_loader;
pub mod pca;
pub mod transcript;
