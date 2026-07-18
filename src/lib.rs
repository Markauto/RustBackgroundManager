//! Background Manager's reusable application core.

pub mod analysis;
pub mod cli;
pub mod collection;
pub mod config;
pub mod db;
pub mod doctor;
pub mod filter;
pub mod model;
pub mod move_files;
pub mod paths;
pub mod scan;
pub mod tui;
pub mod wpaperd;

pub use paths::AppPaths;
