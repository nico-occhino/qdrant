//! Explicit S.3C scalar clustering scale experiment; never runs by default.
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use serde_json::json;

use super::LmiConfig;
use crate::spaces::metric::Metric;
use crate::spaces::simple::DotProductMetric;

#[test]
#[ignore = "Explicit S.3C spherical clustering scale benchmark"]
fn phase_s3c_spherical_microbenchmark() {
    let output = std::env::var("LMI_S3C_MICRO_OUTPUT").unwrap();
    assert!(!std::path::Path::new(&output).exists());
    let sample_size: usize = std::env::var("LMI_S3C_SAMPLE").unwrap().parse().unwrap();
    let buckets: usize = std::env::var("LMI_S3C_BUCKETS").unwrap().parse().unwrap();
    let iterations: usize = std::env::var("LMI_S3C_ITERATIONS")
        .unwrap_or_else(|_| "3".into())
        .parse()
        .unwrap();
    let dim = 768usize;
    assert!(sample_size >= buckets && (1..=1_000_000).contains(&sample_size));
    assert!((1..=10_000).contains(&buckets));
    assert!((1..=20).contains(&iterations));
    let started = Instant::now();
    let sample_file = std::env::var_os("LMI_S3C_SAMPLE_FILE");
    let (data, source_kind) = if let Some(path) = &sample_file {
        use crate::types::Distance;
        use std::io::Read;
        let mut input = std::io::BufReader::new(fs_err::File::open(path).unwrap());
        let mut raw = vec![0u8; dim * std::mem::size_of::<f32>()];
        let mut data = Vec::with_capacity(sample_size.checked_mul(dim).unwrap());
        for _ in 0..sample_size {
            input.read_exact(&mut raw).unwrap();
            let row = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect::<Vec<_>>();
            data.extend(Distance::Cosine.preprocess_vector::<f32>(row));
        }
        let mut extra = [0u8; 1];
        assert_eq!(
            input.read(&mut extra).unwrap(),
            0,
            "sample file has extra rows"
        );
        (
            data,
            "LAION Lance stratified sample with Qdrant cosine preprocessing",
        )
    } else {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut data = vec![0.0f32; sample_size.checked_mul(dim).unwrap()];
        for row in data.chunks_exact_mut(dim) {
            let mut squared = 0.0f64;
            for v in row.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *v = (((state >> 40) as i32) - 8_388_608) as f32 / 8_388_608.0;
                squared += f64::from(*v).powi(2);
            }
            let reciprocal = 1.0 / squared.sqrt();
            row.iter_mut()
                .for_each(|v| *v = (f64::from(*v) * reciprocal) as f32);
        }
        (data, "synthetic normalized sample")
    };
    let generation_seconds = started.elapsed().as_secs_f64();
    let config = LmiConfig {
        n_buckets: buckets,
        sample_size,
        kmeans_iterations: iterations,
        nprobe: 1,
        seed: 42,
        ..LmiConfig::default()
    };
    if std::env::var_os("LMI_S3C_NATIVE_ONLY").is_some() {
        let started = Instant::now();
        let (labels, _, stats) = super::spherical_kmeans::cluster_with_centers_qdrant(
            &data,
            dim,
            &config,
            &AtomicBool::new(false),
        )
        .unwrap();
        let total_seconds = started.elapsed().as_secs_f64();
        let mut sizes = stats.sizes.clone();
        sizes.sort_unstable();
        let status = fs_err::read_to_string("/proc/self/status").unwrap();
        let peak_rss_kib: usize = status
            .lines()
            .find_map(|line| {
                line.strip_prefix("VmHWM:")
                    .and_then(|v| v.split_whitespace().next())
                    .and_then(|v| v.parse().ok())
            })
            .unwrap();
        let result = json!({
            "kernel":"qdrant_f32_dot", "sample_size":sample_size, "buckets":buckets,
            "dimension":dim, "configured_iterations":iterations, "actual_iterations":stats.iterations,
            "generation_seconds":generation_seconds, "total_seconds":total_seconds,
            "initialization_seconds":stats.initialization_seconds,
            "assignment_seconds_per_iteration":stats.assignment_seconds,
            "accumulation_seconds_per_iteration":stats.accumulation_seconds,
            "normalization_seconds_per_iteration":stats.normalization_seconds,
            "final_assignment_seconds":stats.final_assignment_seconds,
            "peak_rss_kib":peak_rss_kib,
            "cluster_sizes":{"min":sizes[0], "median":sizes[sizes.len()/2],
                "mean":sample_size as f64/buckets as f64,
                "p95":sizes[(sizes.len()*95/100).min(sizes.len()-1)],
                "max":sizes[sizes.len()-1], "empty":sizes.iter().filter(|&&v|v==0).count()},
            "sample_bytes":4*sample_size*dim, "centroid_bytes":8*buckets*dim,
            "accumulator_bytes":8*buckets*dim, "temporary_f32_centroid_bytes":4*buckets*dim,
            "assignment_bytes":8*sample_size, "labels":labels.len(), "workers":1,
            "scope":source_kind, "full_corpus_build":false
        });
        fs_err::write(output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
        eprintln!("S3C_NATIVE_MICRO {result}");
        return;
    }
    let started = Instant::now();
    let (labels, centers, stats) =
        super::spherical_kmeans::cluster_with_centers(&data, dim, &config, &AtomicBool::new(false))
            .unwrap();
    let total_seconds = started.elapsed().as_secs_f64();
    let simd_cluster_started = Instant::now();
    let (simd_labels, simd_centers, simd_stats) =
        super::spherical_kmeans::cluster_with_centers_qdrant(
            &data,
            dim,
            &config,
            &AtomicBool::new(false),
        )
        .unwrap();
    let simd_cluster_seconds = simd_cluster_started.elapsed().as_secs_f64();
    let full_simd_mismatches = labels
        .iter()
        .zip(&simd_labels)
        .filter(|(a, b)| a != b)
        .count();
    let max_center_difference = centers
        .iter()
        .zip(&simd_centers)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
        .fold(0.0_f64, f64::max);
    // E2 experiment only: compare the existing f32 SIMD dot kernel
    // against the f64 scalar oracle on identical final centroids.
    let centers_f32: Vec<Vec<f32>> = centers
        .iter()
        .map(|c| c.iter().map(|&v| v as f32).collect())
        .collect();
    let simd_started = Instant::now();
    let mut simd_mismatches = 0usize;
    let mut simd_examples = Vec::new();
    for (point, row) in data.chunks_exact(dim).enumerate() {
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (bucket, center) in centers_f32.iter().enumerate() {
            let score = DotProductMetric::similarity(row, center);
            if score > best_score {
                best_score = score;
                best = bucket;
            }
        }
        if best as i64 != labels[point] {
            simd_mismatches += 1;
            if simd_examples.len() < 16 {
                simd_examples.push((point, labels[point], best));
            }
        }
    }
    let simd_seconds = simd_started.elapsed().as_secs_f64();
    let mut sizes = stats.sizes.clone();
    sizes.sort_unstable();
    let assignment_seconds: f64 =
        stats.assignment_seconds.iter().sum::<f64>() + stats.final_assignment_seconds;
    let dot_products = (sample_size as u128) * (buckets as u128) * ((stats.iterations + 1) as u128);
    let status = fs_err::read_to_string("/proc/self/status").unwrap();
    let peak_kib: usize = status
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmHWM:")
                .and_then(|v| v.split_whitespace().next())
                .and_then(|v| v.parse().ok())
        })
        .unwrap();
    let result = json!({
        "sample_size":sample_size,"buckets":buckets,"dimension":dim,
        "configured_iterations":iterations,"actual_iterations":stats.iterations,
        "generation_seconds":generation_seconds,"total_seconds":total_seconds,
        "initialization_seconds":stats.initialization_seconds,
        "assignment_seconds_per_iteration":stats.assignment_seconds,
        "accumulation_seconds_per_iteration":stats.accumulation_seconds,
        "normalization_seconds_per_iteration":stats.normalization_seconds,
        "final_assignment_seconds":stats.final_assignment_seconds,
        "qdrant_simd_full_cluster_seconds":simd_cluster_seconds,
        "qdrant_simd_full_cluster_mismatches":full_simd_mismatches,
        "qdrant_simd_full_cluster_iterations":simd_stats.iterations,
        "qdrant_simd_full_cluster_assignment_seconds_per_iteration":simd_stats.assignment_seconds,
        "qdrant_simd_full_cluster_final_assignment_seconds":simd_stats.final_assignment_seconds,
        "qdrant_simd_full_cluster_max_center_difference":max_center_difference,
        "qdrant_simd_final_assignment_seconds":simd_seconds,
        "qdrant_simd_final_assignment_mismatches":simd_mismatches,
        "qdrant_simd_examples":simd_examples,
        "qdrant_simd_centroid_f32_bytes":4*buckets*dim,
        "assignments_per_second":(sample_size*(stats.iterations+1)) as f64/assignment_seconds,
        "dot_products_per_second":(dot_products as f64)/assignment_seconds,
        "coordinate_products_per_second":(dot_products as f64)*(dim as f64)/assignment_seconds,
        "sample_bytes":4*sample_size*dim,
        "centroid_bytes":8*buckets*dim,
        "accumulator_bytes":8*buckets*dim,
        "assignment_bytes":8*sample_size,
        "peak_rss_kib":peak_kib,
        "cluster_sizes":{"min":sizes[0],"median":sizes[sizes.len()/2],
            "mean":sample_size as f64/buckets as f64,"p95":sizes[(sizes.len()*95/100).min(sizes.len()-1)],
            "max":sizes[sizes.len()-1],"empty":sizes.iter().filter(|&&v|v==0).count()},
        "labels":labels.len(),"centers":centers.len(),"workers":1,
        "scope":source_kind, "full_corpus_build":false
    });
    fs_err::write(output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    eprintln!("S3C_MICRO {result}");
}

#[test]
#[ignore = "Explicit trained LAION large-bucket Torch/native tie gate"]
fn phase_s3c_trained_large_b_tie_gate() {
    use super::routing::BuildRouter;
    use crate::types::Distance;
    use std::io::Read;

    let data_path = std::env::var("LMI_S3C_SAMPLE_FILE")
        .or_else(|_| std::env::var("LMI_S3C_CORPUS"))
        .unwrap();
    let output = std::env::var("LMI_S3C_TIE_OUTPUT").unwrap();
    assert!(!std::path::Path::new(&output).exists());
    let dim = 768usize;
    let sample_count: usize = std::env::var("LMI_S3C_SAMPLE")
        .unwrap_or_else(|_| "10000".into())
        .parse()
        .unwrap();
    let validation_count: usize = std::env::var("LMI_S3C_VALIDATION_COUNT")
        .unwrap_or_else(|_| "2048".into())
        .parse()
        .unwrap();
    let bucket_count: usize = std::env::var("LMI_S3C_BUCKETS")
        .unwrap_or_else(|_| "1000".into())
        .parse()
        .unwrap();
    let kmeans_iterations: usize = std::env::var("LMI_S3C_ITERATIONS")
        .unwrap_or_else(|_| "3".into())
        .parse()
        .unwrap();
    let mut file = std::io::BufReader::new(fs_err::File::open(&data_path).unwrap());
    let mut raw = vec![0u8; dim * 4];
    let mut read_rows = |file: &mut std::io::BufReader<fs_err::File>, count: usize| -> Vec<f32> {
        let mut result = Vec::with_capacity(count * dim);
        for _ in 0..count {
            file.read_exact(&mut raw).unwrap();
            let row = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect::<Vec<_>>();
            result.extend(Distance::Cosine.preprocess_vector::<f32>(row));
        }
        result
    };
    let sample = read_rows(&mut file, sample_count);
    let validation = if let Ok(path) = std::env::var("LMI_S3C_VALIDATION_FILE") {
        let mut validation_file = std::io::BufReader::new(fs_err::File::open(path).unwrap());
        read_rows(&mut validation_file, validation_count)
    } else {
        read_rows(&mut file, validation_count)
    };
    let config = LmiConfig {
        n_buckets: bucket_count,
        sample_size: sample_count,
        hidden_dim: 512,
        epochs: 30,
        batch_size: 256,
        routing_batch_size: 256,
        kmeans_iterations,
        nprobe: 1,
        seed: 42,
    };
    let stopped = AtomicBool::new(false);
    let started = Instant::now();
    let (teacher_labels, _, teacher_stats) =
        super::spherical_kmeans::cluster_with_centers_qdrant(&sample, dim, &config, &stopped)
            .unwrap();
    let teacher_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let trained =
        super::training::train_with_tch(&sample, dim, &config, Distance::Cosine, &stopped).unwrap();
    let train_seconds = started.elapsed().as_secs_f64();
    let mut native = BuildRouter::new(&trained.native, &stopped).unwrap();
    let mut raw_mismatches = 0usize;
    let mut safe_mismatches = 0usize;
    let mut fallbacks = 0usize;
    let mut margins = Vec::with_capacity(validation_count);
    let started = Instant::now();
    for chunk in validation.chunks(256 * dim) {
        let (logits, torch_ids) = trained.logits_for_test(chunk).unwrap();
        let (safe_ids, count) = trained
            .top_buckets_tie_safe(chunk, &stopped, &mut native)
            .unwrap();
        fallbacks += count;
        for (row_id, row) in chunk.chunks_exact(dim).enumerate() {
            let expected = native.top_bucket(row, &stopped).unwrap();
            raw_mismatches += usize::from(torch_ids[row_id] != expected);
            safe_mismatches += usize::from(safe_ids[row_id] != expected);
            let scores = &logits[row_id * config.n_buckets..(row_id + 1) * config.n_buckets];
            let mut first = f32::NEG_INFINITY;
            let mut second = f32::NEG_INFINITY;
            for &score in scores {
                if score > first {
                    second = first;
                    first = score;
                } else if score > second {
                    second = score;
                }
            }
            margins.push(first - second);
        }
    }
    let routing_seconds = started.elapsed().as_secs_f64();
    margins.sort_by(f32::total_cmp);
    let sample_accuracy = sample
        .chunks_exact(dim)
        .zip(&teacher_labels)
        .filter(|(row, label)| native.top_bucket(row, &stopped).unwrap() == **label as usize)
        .count() as f64
        / sample_count as f64;
    let result = json!({
        "source":format!("sample={data_path}; validation={}", std::env::var("LMI_S3C_VALIDATION_FILE").unwrap_or_else(|_| data_path.clone())),
        "sample_size":sample_count,"validation_size":validation_count,"dimension":dim,
        "buckets":config.n_buckets,"hidden_dim":config.hidden_dim,"epochs":config.epochs,
        "teacher_seconds":teacher_seconds,"train_seconds_including_second_teacher":train_seconds,
        "teacher_active_buckets":teacher_stats.sizes.iter().filter(|&&n|n>0).count(),
        "teacher_min_size":teacher_stats.sizes.iter().min(),
        "teacher_max_size":teacher_stats.sizes.iter().max(),
        "sample_teacher_accuracy":sample_accuracy,
        "routing_diagnostic_seconds":routing_seconds,
        "raw_torch_native_mismatches":raw_mismatches,
        "tie_safe_mismatches":safe_mismatches,
        "tie_safe_fallbacks":fallbacks,
        "fallback_fraction":fallbacks as f64 / validation_count as f64,
        "margin_min":margins[0],"margin_p50":margins[validation_count/2],
        "margin_p01":margins[validation_count/100],
        "fast_path_fraction":1.0 - fallbacks as f64 / validation_count as f64,
    });
    fs_err::write(output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    assert_eq!(safe_mismatches, 0);
    eprintln!("S3C_TRAINED_TIE {result}");
}
