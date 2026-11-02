//! Shared wire protocol and presentation encoding code.

pub mod endpoint;
pub(crate) mod render_ansi;
pub(crate) mod surface_delta;
pub(crate) mod surface_reuse;
pub(crate) mod surface_scroll;
#[cfg(test)]
mod surface_scroll_fuzz;
mod wire;

pub use wire::*;
