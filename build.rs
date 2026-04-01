fn main() {
    println!("cargo:rustc-link-search=/opt/cuda/lib64"); // default Arch Linux location
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64"); // default Ubuntu location
    println!("cargo:rustc-link-lib=dylib=cudart");
}
