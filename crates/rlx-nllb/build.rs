fn main() {
    // Host LM head is a vocab×d GEMV every decode step — Accelerate/AMX on Apple.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
