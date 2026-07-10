//! kirie CLI entry point.
//!
//! A thin shell over [`kirie::run`]: it forwards the full process argv (so the
//! compat parser sees `argv[0]` for its `Running with:` banner and error
//! suffixes, docs/compat-cli.md §1.2, §4.7) and returns its exit code.

use std::process::ExitCode;

fn main() -> ExitCode {
    kirie::run(std::env::args_os().collect())
}
