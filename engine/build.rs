fn main() {
    println!("cargo::rerun-if-changed=../.githooks");
    std::process::Command::new("git")
        .args(["config", "core.hooksPath", ".githooks"])
        .current_dir("..")
        .status()
        .ok();
}
