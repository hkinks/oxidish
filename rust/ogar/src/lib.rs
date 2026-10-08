//! ogar: a Rust port of the MultiOgar game server (FFA) for fast headless evaluation matches and
//! for a real-time websocket server that speaks MultiOgar's client protocol.

pub mod agent;
pub mod bots;
pub mod config;
pub mod endless;
pub mod matches;
#[cfg(feature = "vision-net")]
pub mod nn;
pub mod protocol;
pub mod record;
pub mod render;
pub mod rng;
pub mod server;
pub mod vision;
#[cfg(feature = "vision-net")]
pub mod visionbot;
pub mod world;
