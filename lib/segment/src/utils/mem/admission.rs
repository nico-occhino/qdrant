//! Memory available for a bounded, optional build. Ordinary memory reporting is unchanged.
//! Kernel references: docs.kernel.org/admin-guide/cgroup-v2.html (memory.stat)
//! and docs.kernel.org/admin-guide/cgroup-v1/memory.html (hierarchical counters).
use std::io;

#[derive(Debug, serde::Serialize)]
pub struct AdmissionMemory {
    pub host_available_bytes: u64,
    pub available_bytes: u64,
    pub cgroups: Vec<CgroupAdmission>,
}

#[derive(Debug, serde::Serialize)]
pub struct CgroupAdmission {
    pub path: String,
    pub limit_bytes: Option<u64>,
    pub high_bytes: Option<u64>,
    pub usage_bytes: u64,
    pub peak_bytes: Option<u64>,
    pub inactive_file_bytes: Option<u64>,
    pub clean_inactive_file_bytes: u64,
    pub credited_reclaimable_bytes: u64,
    pub available_bytes: Option<u64>,
    pub stat_status: String,
    pub memory_stat: Option<String>,
    pub memory_events: Option<String>,
}

/// Credit only half of clean inactive file pages. Subtract dirty, writeback,
/// shmem and unevictable counts conservatively (even where categories overlap).
/// Active cache, anonymous memory, swap and reclaimable slab get no credit.
/// Unknown/malformed counters give zero credit, never an optimistic fallback.
fn cache_credit(stat: &str, v2: bool, usage: u64) -> Option<(u64, u64, u64)> {
    let mut values = std::collections::BTreeMap::new();
    for line in stat.lines() {
        let mut parts = line.split_whitespace();
        let key = parts.next()?;
        let value = parts.next()?.parse::<u64>().ok()?;
        if parts.next().is_some() || values.insert(key, value).is_some() {
            return None;
        }
    }
    let key = |v2_key: &'static str, v1_key: &'static str| {
        values.get(if v2 { v2_key } else { v1_key }).copied()
    };
    let inactive = key("inactive_file", "total_inactive_file")?;
    let file = key("file", "total_cache")?;
    let excluded = key("file_dirty", "total_dirty")?
        .saturating_add(key("file_writeback", "total_writeback")?)
        .saturating_add(key("shmem", "total_shmem")?)
        .saturating_add(key("unevictable", "total_unevictable")?);
    let clean = inactive.min(file).min(usage).saturating_sub(excluded);
    Some((inactive, clean, clean / 2))
}

fn bounded_available(limit: u64, usage: u64, credit: u64) -> u64 {
    limit.saturating_sub(usage.saturating_sub(credit.min(usage)))
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::fs;
    use std::path::Path;

    fn limit(path: &Path) -> io::Result<Option<u64>> {
        let text = fs::read_to_string(path)?;
        if text.trim() == "max" {
            return Ok(None);
        }
        let value = text.trim().parse::<u64>().map_err(io::Error::other)?;
        Ok((value < i64::MAX as u64 - (1 << 20)).then_some(value))
    }

    fn usage(path: &Path) -> io::Result<u64> {
        fs::read_to_string(path)?
            .trim()
            .parse()
            .map_err(io::Error::other)
    }

    pub(super) fn hierarchy(
        leaf: &Path,
        root: &Path,
        v2: bool,
        host: u64,
    ) -> io::Result<AdmissionMemory> {
        let leaf = leaf.canonicalize()?;
        let root = root.canonicalize()?;
        if !leaf.starts_with(&root) {
            return Err(io::Error::other(
                "memory cgroup is outside controller mount",
            ));
        }
        let mut result = AdmissionMemory {
            host_available_bytes: host,
            available_bytes: host,
            cgroups: vec![],
        };
        for dir in leaf.ancestors().take_while(|p| p.starts_with(&root)) {
            let limit_path = dir.join(if v2 {
                "memory.max"
            } else {
                "memory.limit_in_bytes"
            });
            let usage_path = dir.join(if v2 {
                "memory.current"
            } else {
                "memory.usage_in_bytes"
            });
            // A controller absent at this level may still constrain an ancestor.
            if !limit_path.try_exists()? && !usage_path.try_exists()? {
                continue;
            }
            let hard = limit(&limit_path)?;
            let high_path = dir.join("memory.high");
            let high = if v2 && high_path.try_exists()? {
                limit(&high_path)?
            } else {
                None
            };
            let before = usage(&usage_path)?;
            let stat = fs::read_to_string(dir.join("memory.stat"));
            // Counters aren't atomic. Prefer the larger usage around the stat read.
            let current = before.max(usage(&usage_path)?);
            let credit = stat
                .as_ref()
                .ok()
                .and_then(|s| cache_credit(s, v2, current));
            let (inactive, clean, credited) = match credit {
                Some((inactive, clean, credited)) => (Some(inactive), clean, credited),
                None => (None, 0, 0),
            };
            let ceiling = match (hard, high) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            let available = ceiling.map(|cap| bounded_available(cap, current, credited));
            if let Some(available) = available {
                result.available_bytes = result.available_bytes.min(available);
            }
            result.cgroups.push(CgroupAdmission {
                path: dir.display().to_string(),
                limit_bytes: hard,
                high_bytes: high,
                usage_bytes: current,
                peak_bytes: usage(&dir.join(if v2 {
                    "memory.peak"
                } else {
                    "memory.max_usage_in_bytes"
                }))
                .ok(),
                inactive_file_bytes: inactive,
                clean_inactive_file_bytes: clean,
                credited_reclaimable_bytes: credited,
                available_bytes: available,
                memory_stat: stat.ok(),
                memory_events: fs::read_to_string(dir.join("memory.events")).ok(),
                stat_status: if credit.is_some() {
                    "complete; 50% clean-inactive credit"
                } else {
                    "missing or malformed counters; zero cache credit"
                }
                .into(),
            });
        }
        Ok(result)
    }

    pub(super) fn snapshot(host: u64) -> io::Result<AdmissionMemory> {
        let Some((leaf, v2)) =
            super::super::cgroups_mem::memory_cgroup_dir().map_err(io::Error::other)?
        else {
            return Ok(AdmissionMemory {
                host_available_bytes: host,
                available_bytes: host,
                cgroups: vec![],
            });
        };
        let root = super::super::cgroups_mem::memory_mountpoint(v2).map_err(io::Error::other)?;
        hierarchy(&leaf, &root, v2, host)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        fn frame(path: &Path, max: &str, current: &str) {
            fs::create_dir_all(path).unwrap();
            fs::write(path.join("memory.max"), max).unwrap();
            fs::write(path.join("memory.current"), current).unwrap();
            fs::write(path.join("memory.stat"),
                "inactive_file 800\nfile 900\nfile_dirty 100\nfile_writeback 100\nshmem 0\nunevictable 0\n").unwrap();
        }
        #[test]
        fn ancestor_limits_high_and_host_cap_are_respected() {
            let t = tempfile::tempdir().unwrap();
            let child = t.path().join("child");
            frame(t.path(), "1000", "950");
            frame(&child, "max", "100");
            assert_eq!(
                hierarchy(&child, t.path(), true, 9000)
                    .unwrap()
                    .available_bytes,
                350
            );
            fs::write(t.path().join("memory.high"), "900").unwrap();
            assert_eq!(
                hierarchy(&child, t.path(), true, 9000)
                    .unwrap()
                    .available_bytes,
                250
            );
            assert_eq!(
                hierarchy(&child, t.path(), true, 20)
                    .unwrap()
                    .available_bytes,
                20
            );
        }
        #[test]
        fn missing_stats_get_no_credit_unreadable_usage_fails() {
            let t = tempfile::tempdir().unwrap();
            frame(t.path(), "1000", "950");
            fs::remove_file(t.path().join("memory.stat")).unwrap();
            assert_eq!(
                hierarchy(t.path(), t.path(), true, 9000)
                    .unwrap()
                    .available_bytes,
                50
            );
            fs::write(t.path().join("memory.current"), "bad").unwrap();
            assert!(hierarchy(t.path(), t.path(), true, 9000).is_err());
        }
    }
}

pub(super) fn snapshot(host: u64) -> io::Result<AdmissionMemory> {
    #[cfg(target_os = "linux")]
    return linux::snapshot(host);
    #[cfg(not(target_os = "linux"))]
    Ok(AdmissionMemory {
        host_available_bytes: host,
        available_bytes: host,
        cgroups: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn credit_excludes_dirty_shared_and_unevictable_pages() {
        let s = "inactive_file 800\nfile 1000\nfile_dirty 100\nfile_writeback 50\nshmem 100\nunevictable 50\n";
        assert_eq!(cache_credit(s, true, 1000), Some((800, 500, 250)));
        assert_eq!(cache_credit(s, true, 200), Some((800, 0, 0)));
        assert_eq!(cache_credit("inactive_file 800", true, 1000), None);
        assert_eq!(bounded_available(100, 200, 20), 0);
    }
    #[test]
    fn v1_uses_hierarchical_counters_only() {
        let s = "inactive_file 999999\ntotal_inactive_file 800\ntotal_cache 900\ntotal_dirty 0\ntotal_writeback 0\ntotal_shmem 0\ntotal_unevictable 0\n";
        assert_eq!(cache_credit(s, false, 1000), Some((800, 800, 400)));
        assert_eq!(
            cache_credit(&s.replace("total_shmem 0", "total_shmem bad"), false, 1000),
            None
        );
    }
}
