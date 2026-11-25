use serde::{Deserialize, Serialize};
use sysinfo::System;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuInfo {
    pub name: String,
    pub arch: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_fma: bool,
}

pub fn detect_cpu(sys: &System) -> CpuInfo {
    let name = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Unknown CPU".to_string());

    let logical_cores = sys.cpus().len().max(1);
    let physical_cores = sys.physical_core_count().unwrap_or(logical_cores);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let (has_avx, has_avx2, has_avx512, has_fma) = (
        std::is_x86_feature_detected!("avx"),
        std::is_x86_feature_detected!("avx2"),
        std::is_x86_feature_detected!("avx512f"),
        std::is_x86_feature_detected!("fma"),
    );

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let (has_avx, has_avx2, has_avx512, has_fma) = (false, false, false, false);

    CpuInfo {
        name,
        arch: std::env::consts::ARCH.to_string(),
        physical_cores,
        logical_cores,
        has_avx,
        has_avx2,
        has_avx512,
        has_fma,
    }
}
