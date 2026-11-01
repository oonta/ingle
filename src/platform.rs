use crate::load::StartupError;

/// CPU class the SIMD rule branches on.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CpuArch {
    X86_64,
    Aarch64,
    Other,
}

/// Resolved `--kv-pool-mb`, including the `auto` fallback.
#[derive(Debug, PartialEq, Eq)]
pub struct ResolvedPool {
    pub mb: u32,
    pub ram_mb: Option<u64>,
    pub fallback: bool,
}

/// `auto` or an explicit MiB value. An explicit value does not use `detected_ram_mb`.
pub enum PoolRequest {
    Auto,
    Explicit(u32),
}

/// On x86_64, missing AVX2 is a startup error. Other arches do not require it.
pub fn require_avx2(arch: CpuArch, has_avx2: bool) -> Result<(), StartupError> {
    if arch == CpuArch::X86_64 && !has_avx2 {
        Err(StartupError::MissingAvx2)
    } else {
        Ok(())
    }
}

/// `auto_mb = clamp(physical_ram_mb / 8, 256, 8192)`. A missing report uses 2048.
pub fn resolve_kv_pool(
    request: PoolRequest,
    detected_ram_mb: Option<u64>,
) -> Result<ResolvedPool, StartupError> {
    match request {
        PoolRequest::Explicit(mb) if mb < 64 => Err(StartupError::KvPoolTooSmall),
        PoolRequest::Explicit(mb) => Ok(ResolvedPool {
            mb,
            ram_mb: None,
            fallback: false,
        }),
        PoolRequest::Auto => match detected_ram_mb {
            Some(ram) if ram > 0 => {
                let divided = u32::try_from(ram / 8).unwrap_or(u32::MAX);
                Ok(ResolvedPool {
                    mb: divided.clamp(256, 8192),
                    ram_mb: Some(ram),
                    fallback: false,
                })
            }
            _ => Ok(ResolvedPool {
                mb: 2048,
                ram_mb: None,
                fallback: true,
            }),
        },
    }
}

/// One English startup line. It does not include prompt text.
pub fn startup_line(triple: &str, pool: &ResolvedPool) -> String {
    match pool.ram_mb {
        Some(ram) => format!(
            "target {triple}, physical RAM {ram} MiB, KV pool {} MiB",
            pool.mb
        ),
        None if pool.fallback => {
            format!(
                "target {triple}, physical RAM unavailable, KV pool {} MiB",
                pool.mb
            )
        }
        None => format!("target {triple}, KV pool {} MiB", pool.mb),
    }
}

/// The version 1 triple for an arch/os pair from `std::env::consts`.
pub fn known_triple(arch: &str, os: &str) -> &'static str {
    match (arch, os) {
        ("x86_64", "windows") => "x86_64-pc-windows-msvc",
        ("x86_64", "linux") => "x86_64-unknown-linux-gnu",
        ("aarch64", "macos") => "aarch64-apple-darwin",
        _ => "unknown",
    }
}

/// This process's version 1 triple, or `unknown` when the host is outside the matrix.
pub fn target_triple() -> &'static str {
    known_triple(std::env::consts::ARCH, std::env::consts::OS)
}

/// AVX2 on this process. Non-x86_64 hosts pass the check.
pub fn require_host_simd() -> Result<(), StartupError> {
    let arch = match std::env::consts::ARCH {
        "x86_64" => CpuArch::X86_64,
        "aarch64" => CpuArch::Aarch64,
        _ => CpuArch::Other,
    };
    require_avx2(arch, host_has_avx2())
}

/// Physical RAM in MiB, or `None` when the OS does not report a size.
pub fn detect_ram_mb() -> Option<u64> {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let bytes = system.total_memory();
    if bytes == 0 {
        None
    } else {
        Some(bytes / 1024 / 1024)
    }
}

fn host_has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CpuArch, PoolRequest, ResolvedPool, known_triple, require_avx2, resolve_kv_pool,
        startup_line,
    };
    use crate::load::StartupError;

    #[test]
    fn x86_64_without_avx2_names_the_feature() {
        let error = require_avx2(CpuArch::X86_64, false).unwrap_err();
        assert_eq!(error, StartupError::MissingAvx2);
        assert!(error.to_string().contains("AVX2"));
    }

    #[test]
    fn x86_64_with_avx2_and_other_arches_continue() {
        assert!(require_avx2(CpuArch::X86_64, true).is_ok());
        assert!(require_avx2(CpuArch::Aarch64, false).is_ok());
        assert!(require_avx2(CpuArch::Other, false).is_ok());
    }

    #[test]
    fn sixteen_gib_resolves_to_2048() {
        let pool = resolve_kv_pool(PoolRequest::Auto, Some(16 * 1024)).unwrap();
        assert_eq!(
            pool,
            ResolvedPool {
                mb: 2048,
                ram_mb: Some(16 * 1024),
                fallback: false,
            }
        );
    }

    #[test]
    fn auto_clamps_to_the_closed_range() {
        let low = resolve_kv_pool(PoolRequest::Auto, Some(1024)).unwrap();
        let high = resolve_kv_pool(PoolRequest::Auto, Some(100 * 1024)).unwrap();
        assert_eq!(low.mb, 256);
        assert_eq!(high.mb, 8192);
        assert!(!low.fallback);
    }

    #[test]
    fn a_missing_ram_report_falls_back_to_2048() {
        for ram in [None, Some(0)] {
            let pool = resolve_kv_pool(PoolRequest::Auto, ram).unwrap();
            assert_eq!(pool.mb, 2048);
            assert!(pool.fallback);
            assert_eq!(pool.ram_mb, None);
        }
    }

    #[test]
    fn an_explicit_pool_skips_detection_and_rejects_a_value_below_64() {
        let pool = resolve_kv_pool(PoolRequest::Explicit(64), Some(16 * 1024)).unwrap();
        assert_eq!(
            pool,
            ResolvedPool {
                mb: 64,
                ram_mb: None,
                fallback: false,
            }
        );
        assert_eq!(
            resolve_kv_pool(PoolRequest::Explicit(63), None).unwrap_err(),
            StartupError::KvPoolTooSmall
        );
    }

    #[test]
    fn the_startup_line_names_the_triple_ram_and_pool() {
        let pool = ResolvedPool {
            mb: 2048,
            ram_mb: Some(16 * 1024),
            fallback: false,
        };
        let line = startup_line("x86_64-pc-windows-msvc", &pool);
        assert!(line.contains("x86_64-pc-windows-msvc"), "{line}");
        assert!(line.contains("16384"), "{line}");
        assert!(line.contains("2048"), "{line}");
        assert!(!line.contains("unavailable"), "{line}");
    }

    #[test]
    fn the_startup_line_records_the_ram_fallback() {
        let pool = ResolvedPool {
            mb: 2048,
            ram_mb: None,
            fallback: true,
        };
        let line = startup_line("aarch64-apple-darwin", &pool);
        assert!(line.contains("unavailable"), "{line}");
        assert!(line.contains("2048"), "{line}");
        assert!(line.contains("aarch64-apple-darwin"), "{line}");
    }

    #[test]
    fn known_triples_match_the_version_1_matrix() {
        assert_eq!(known_triple("x86_64", "windows"), "x86_64-pc-windows-msvc");
        assert_eq!(known_triple("x86_64", "linux"), "x86_64-unknown-linux-gnu");
        assert_eq!(known_triple("aarch64", "macos"), "aarch64-apple-darwin");
        assert_eq!(known_triple("x86_64", "macos"), "unknown");
    }
}
