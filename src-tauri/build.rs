use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let local_lib_path = PathBuf::from(manifest_dir).join("local_lib");
    println!(
        "cargo:rustc-link-search=native={}",
        local_lib_path.display()
    );

    tauri_build::build()
}
