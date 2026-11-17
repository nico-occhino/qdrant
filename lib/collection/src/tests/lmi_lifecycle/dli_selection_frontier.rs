//! Native-replay candidate-selection frontier on a frozen DLI topology.
//!
//! Loads *copies* of the persisted continuation40 segments (never the
//! originals), then for every query/policy/budget calls the production
//! `route_pinned_traced` + `resolve_pinned_candidates` code with original
//! scoring (scorer held fixed), and records selected work, recall against the
//! full-pool exact oracle, and per-neighbor miss attribution.
//!
//! It also measures two allocation-free reference costs per query:
//! * `oracle_cost`: postings of exactly the gold-containing buckets (+ Plain if
//!   a gold neighbor lives there): the bucket-granularity lower bound.
//! * `rank_cost`: per member, the model's probability-ordered bucket prefix up
//!   to the deepest gold-containing bucket: perfect cross-level allocation with
//!   the router's own ranking.
//!
//! Optionally builds a static LMI over the same 65,536 vectors with the same
//! model configuration (native `SegmentBuilder`) as a flat reference.
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use common::flags::FeatureFlags;
use common::types::DeferredBehavior;
use segment::data_types::query_context::VectorQueryContext;
use segment::data_types::vectors::DEFAULT_VECTOR_NAME;
use segment::entry::ReadSegmentEntry;
use segment::id_tracker::IdTrackerRead;
use segment::index::VectorIndexEnum;
use segment::segment::Segment;
use segment::segment_constructor::load_segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::types::{Distance, ExtendedPointId, HnswGlobalConfig, Indexes};
use uuid::Uuid;

use crate::collection_manager::dli_bucket_plan::RoutingPolicy;
use crate::collection_manager::dli_native_route::{
    CandidateResolution, RouteTrace, resolve_pinned_candidates, route_pinned_traced,
};

const TOP: usize = 30;

struct QuerySet {
    label: String,
    queries: Vec<Vec<f32>>,
    gold: Vec<Vec<u64>>,
}

fn read_set(label: &str, queries: &Path, gold: &Path) -> QuerySet {
    let q = fs_err::read(queries).unwrap();
    let floats: Vec<f32> = q
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let queries: Vec<Vec<f32>> = floats.chunks_exact(768).map(<[f32]>::to_vec).collect();
    let g = fs_err::read(gold).unwrap();
    let ids: Vec<u64> = g
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let gold: Vec<Vec<u64>> = ids.chunks_exact(TOP).map(<[u64]>::to_vec).collect();
    assert_eq!(queries.len(), gold.len(), "{label}: queries/gold mismatch");
    QuerySet {
        label: label.to_owned(),
        queries,
        gold,
    }
}

fn num(id: ExtendedPointId) -> u64 {
    match id {
        ExtendedPointId::NumId(n) => n,
        ExtendedPointId::Uuid(_) => panic!("unexpected UUID point id"),
    }
}

/// Per learned member: offset -> bucket, bucket sizes.
struct MemberIndex {
    level: usize,
    uuid: Uuid,
    bucket_of: HashMap<u32, usize>,
    sizes: Vec<usize>,
}

fn member_index(level: usize, segment: &Segment) -> Option<MemberIndex> {
    let index = segment.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        return None;
    };
    let state = lmi.routing_state().expect("trained member");
    let postings = state.postings();
    let mut bucket_of = HashMap::new();
    let mut sizes = Vec::with_capacity(postings.len());
    for bucket in 0..postings.len() {
        let list = postings.get(bucket).unwrap();
        sizes.push(list.len());
        for &offset in list {
            assert!(
                bucket_of.insert(offset, bucket).is_none(),
                "offset {offset} appears in two buckets"
            );
        }
    }
    Some(MemberIndex {
        level,
        uuid: segment.uuid,
        bucket_of,
        sizes,
    })
}

/// Probability-descending bucket order for one learned member.
fn member_ranking(segment: &Segment, query: &[f32]) -> Vec<usize> {
    let index = segment.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        unreachable!()
    };
    let routed = Distance::Cosine.preprocess_vector::<f32>(query.to_vec());
    let logits = lmi
        .routing_state()
        .unwrap()
        .router()
        .forward(&routed)
        .unwrap();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    order
}

/// Where a gold neighbor physically lives.
#[derive(Clone, Copy)]
enum Owner {
    Plain(Uuid),
    Learned { member: usize, bucket: usize },
}

fn locate(members: &[(usize, &Segment)], index: &[Option<MemberIndex>], id: u64) -> Owner {
    let mut found = None;
    for (i, (_, segment)) in members.iter().enumerate() {
        let internal = segment
            .id_tracker
            .borrow()
            .internal_id_with_behavior(id.into(), DeferredBehavior::VisibleOnly);
        if let Some(offset) = internal {
            assert!(found.is_none(), "gold id {id} visible in two members");
            found = Some(match &index[i] {
                None => Owner::Plain(segment.uuid),
                Some(m) => Owner::Learned {
                    member: i,
                    bucket: *m.bucket_of.get(&offset).expect("live offset has a bucket"),
                },
            });
        }
    }
    found.unwrap_or_else(|| panic!("gold id {id} not present in topology"))
}

fn policies() -> Vec<(&'static str, Option<RoutingPolicy>)> {
    vec![
        ("slot", Some(RoutingPolicy::SlotWeighted)),
        ("density", Some(RoutingPolicy::PopulationDensity)),
        (
            "tdensity2",
            Some(RoutingPolicy::TemperatureDensity {
                temperature_milli: 2000,
            }),
        ),
        ("reserved", Some(RoutingPolicy::ReservedPerLevel)),
        // Offline control: learned budget split proportionally to member
        // population; Plain buffer always fully scored.
        ("proportional", None),
    ]
}

struct Evaluated {
    hits: Vec<u64>,
    selected: HashSet<(Uuid, Option<usize>)>,
    posting_entries: usize,
    unique_offsets: usize,
    original_scored: usize,
    plain_scored: usize,
    buckets_per_level: HashMap<usize, usize>,
    /// Wall time of routing + scoring + resolution (instrumented, warm-state
    /// dependent): comparable across policies within one run only.
    elapsed_ns: u128,
}

fn run_route(
    members: &[(usize, &Segment)],
    query: &[f32],
    arity: usize,
    budget: usize,
    policy: RoutingPolicy,
) -> Evaluated {
    let started = std::time::Instant::now();
    let context = VectorQueryContext::default();
    let mut trace = RouteTrace::default();
    let routed = route_pinned_traced(
        members,
        query,
        arity,
        budget,
        &context,
        policy,
        TOP,
        None,
        Some(&mut trace),
    )
    .unwrap();
    let selected = routed
        .plan
        .buckets
        .iter()
        .map(|b| (b.physical, b.bucket))
        .collect();
    let mut buckets_per_level = HashMap::new();
    for b in &routed.plan.buckets {
        *buckets_per_level.entry(b.level).or_insert(0) += 1;
    }
    let (posting_entries, unique_offsets) = (routed.posting_candidates, routed.unique_posting_offsets);
    let hits = match resolve_pinned_candidates(members, routed, TOP).unwrap() {
        CandidateResolution::Resolved(hits) => hits.into_iter().map(|h| num(h.id)).collect(),
        CandidateResolution::ExactFallback(reason) => {
            panic!("frozen static topology must resolve: {reason:?}")
        }
    };
    Evaluated {
        hits,
        selected,
        posting_entries,
        unique_offsets,
        original_scored: trace.scoring.original_scored,
        plain_scored: trace.scoring.plain_scored,
        buckets_per_level,
        elapsed_ns: started.elapsed().as_nanos(),
    }
}

fn run_proportional(
    members: &[(usize, &Segment)],
    query: &[f32],
    arity: usize,
    budget: usize,
) -> Evaluated {
    let learned_population: usize = members
        .iter()
        .filter(|(l, _)| *l > 0)
        .map(|(_, s)| s.available_point_count())
        .sum();
    let started = std::time::Instant::now();
    let mut all_hits: Vec<(f32, u64)> = Vec::new();
    let mut out = Evaluated {
        hits: Vec::new(),
        selected: HashSet::new(),
        posting_entries: 0,
        unique_offsets: 0,
        original_scored: 0,
        plain_scored: 0,
        buckets_per_level: HashMap::new(),
        elapsed_ns: 0,
    };
    for &(level, segment) in members {
        let share = if level == 0 {
            usize::MAX / 4 // Plain: always fully scored
        } else {
            (budget * segment.available_point_count() / learned_population).max(1)
        };
        let single = [(level, segment)];
        // A single learned member routes by its own probabilities; a single
        // Plain member is planned as one whole bucket. Both are valid plans.
        let context = VectorQueryContext::default();
        let mut trace = RouteTrace::default();
        let routed = route_pinned_traced(
            &single,
            query,
            arity,
            share,
            &context,
            RoutingPolicy::SlotWeighted,
            TOP,
            None,
            Some(&mut trace),
        );
        let routed = match routed {
            Ok(routed) => routed,
            // route requires at least one learned bucket only at the
            // dispatcher; here a Plain-only plan is fine. Propagate errors.
            Err(error) => panic!("proportional route failed: {error}"),
        };
        for b in &routed.plan.buckets {
            out.selected.insert((b.physical, b.bucket));
            *out.buckets_per_level.entry(b.level).or_insert(0) += 1;
        }
        out.posting_entries += routed.posting_candidates;
        out.unique_offsets += routed.unique_posting_offsets;
        out.original_scored += trace.scoring.original_scored;
        out.plain_scored += trace.scoring.plain_scored;
        for (_, hits) in routed.per_segment {
            all_hits.extend(hits.into_iter().map(|h| (h.score, num(h.id))));
        }
    }
    // Disjoint frozen members (no duplicates), so a score merge is exact.
    all_hits.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    out.hits = all_hits.into_iter().take(TOP).map(|(_, id)| id).collect();
    out.elapsed_ns = started.elapsed().as_nanos();
    out
}

fn build_static_reference(
    root: &Path,
    sources: &[(usize, &Segment)],
    template: &Segment,
) -> Segment {
    let staging = tempfile::tempdir_in(root).unwrap();
    let mut config = template.config().clone();
    let vector = config.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap();
    assert!(matches!(vector.index, Indexes::LmiTrained(_)));
    vector.quantization_config = None;
    let mut builder = SegmentBuilder::new(
        staging.path(),
        &config,
        &HnswGlobalConfig::default(),
        FeatureFlags::default(),
    )
    .unwrap();
    let refs: Vec<&Segment> = sources.iter().map(|(_, s)| *s).collect();
    builder.update(&refs, &AtomicBool::new(false)).unwrap();
    builder.build_for_test(root)
}

#[test]
#[ignore = "native-replay selection frontier; requires DLI_FRONTIER_SEGMENTS, DLI_FRONTIER_SETS, DLI_FRONTIER_OUTPUT"]
fn dli_65k_selection_frontier() {
    let _scope = common::ambient::test_guard();
    let segments_dir = PathBuf::from(std::env::var("DLI_FRONTIER_SEGMENTS").unwrap());
    let output = PathBuf::from(std::env::var("DLI_FRONTIER_OUTPUT").unwrap());
    assert!(!output.join("rows.jsonl").exists(), "refusing to overwrite");
    fs_err::create_dir_all(&output).unwrap();
    // label:queries.f32:gold_ids.u64 entries separated by ';'
    let sets: Vec<QuerySet> = std::env::var("DLI_FRONTIER_SETS")
        .unwrap()
        .split(';')
        .map(|spec| {
            let parts: Vec<&str> = spec.split(':').collect();
            read_set(parts[0], Path::new(parts[1]), Path::new(parts[2]))
        })
        .collect();
    let budgets: Vec<usize> = std::env::var("DLI_FRONTIER_BUDGETS")
        .unwrap_or_else(|_| "500,1000,2000,3000,4000,6000,8000,10000,12000,16000,24000".into())
        .split(',')
        .map(|b| b.parse().unwrap())
        .collect();
    // Optional comma-separated policy subset (names as in `policies()` plus
    // `static_flat`); all policies when unset.
    let policy_filter: Option<HashSet<String>> = std::env::var("DLI_FRONTIER_POLICIES")
        .ok()
        .map(|v| v.split(',').map(str::to_owned).collect());
    // Optional cap on queries per set (validation of the harness at scale).
    let query_cap: usize = std::env::var("DLI_FRONTIER_MAX_QUERIES")
        .map_or(usize::MAX, |v| v.parse().unwrap());
    let descriptor: serde_json::Value =
        serde_json::from_slice(&fs_err::read(segments_dir.join("dli_hierarchy.json")).unwrap())
            .unwrap();
    let arity = descriptor["arity"].as_u64().unwrap() as usize;
    let mut loaded: Vec<(usize, Segment)> = descriptor["members"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            let uuid: Uuid = m[0].as_str().unwrap().parse().unwrap();
            let level = m[1].as_u64().unwrap() as usize;
            let path = segments_dir.join("segments").join(uuid.to_string());
            let segment = load_segment(&path, uuid, None, &AtomicBool::new(false), true)
                .unwrap_or_else(|e| panic!("segment {uuid} did not load: {e}"));
            (level, segment)
        })
        .collect();
    loaded.sort_by_key(|(level, _)| std::cmp::Reverse(*level));
    let members: Vec<(usize, &Segment)> = loaded.iter().map(|(l, s)| (*l, s)).collect();
    let index: Vec<Option<MemberIndex>> = members.iter().map(|(l, s)| member_index(*l, s)).collect();
    let total: usize = members.iter().map(|(_, s)| s.available_point_count()).sum();
    let plain_size: usize = members
        .iter()
        .filter(|(l, _)| *l == 0)
        .map(|(_, s)| s.available_point_count())
        .sum();

    let static_ref = std::env::var_os("DLI_FRONTIER_STATIC").map(|_| {
        let started = std::time::Instant::now();
        let template = members.iter().find(|(l, _)| *l > 0).unwrap().1;
        let built = build_static_reference(&output, &members, template);
        let secs = started.elapsed().as_secs_f64();
        println!("STATIC_REFERENCE_BUILT seconds={secs:.1} points={}", built.available_point_count());
        (built, secs)
    });
    let static_index = static_ref.as_ref().map(|(s, _)| member_index(1, s).unwrap());

    let mut rows = fs_err::File::create(output.join("rows.jsonl")).unwrap();
    let mut refs = fs_err::File::create(output.join("reference_costs.jsonl")).unwrap();
    let topology = serde_json::json!({
        "members": members.iter().zip(&index).map(|((l, s), m)| serde_json::json!({
            "uuid": s.uuid, "level": l, "live": s.available_point_count(),
            "buckets": m.as_ref().map(|m| m.sizes.len()),
            "bucket_size_max": m.as_ref().map(|m| m.sizes.iter().max().copied()),
            "bucket_size_nonempty": m.as_ref().map(|m| m.sizes.iter().filter(|s| **s > 0).count()),
            "bucket_sizes": m.as_ref().map(|m| m.sizes.clone()),
        })).collect::<Vec<_>>(),
        "total_live": total, "arity": arity, "budgets": budgets,
        "static_reference": static_ref.as_ref().map(|(s, secs)| serde_json::json!({
            "uuid": s.uuid, "build_seconds": secs,
            "bucket_sizes": static_index.as_ref().map(|m| m.sizes.clone()),
        })),
    });
    fs_err::write(output.join("topology.json"), serde_json::to_vec_pretty(&topology).unwrap()).unwrap();

    for set in &sets {
        for (qi, (query, gold)) in set.queries.iter().zip(&set.gold).enumerate().take(query_cap) {
            let owners: Vec<Owner> = gold.iter().map(|id| locate(&members, &index, *id)).collect();
            let rankings: Vec<Option<Vec<usize>>> = members
                .iter()
                .zip(&index)
                .map(|((_, s), m)| m.as_ref().map(|_| member_ranking(s, query)))
                .collect();
            // Reference costs for top-10 and top-30 coverage.
            let mut ref_row = serde_json::json!({"set": set.label, "query": qi});
            for k in [10usize, 30] {
                let mut oracle_buckets: HashSet<(usize, usize)> = HashSet::new();
                let mut plain_needed = false;
                let mut depth: HashMap<usize, usize> = HashMap::new();
                for owner in &owners[..k] {
                    match *owner {
                        Owner::Plain(_) => plain_needed = true,
                        Owner::Learned { member, bucket } => {
                            oracle_buckets.insert((member, bucket));
                            let rank = rankings[member]
                                .as_ref()
                                .unwrap()
                                .iter()
                                .position(|b| *b == bucket)
                                .unwrap();
                            let d = depth.entry(member).or_insert(0);
                            *d = (*d).max(rank + 1);
                        }
                    }
                }
                let plain_cost = if plain_needed { plain_size } else { 0 };
                let oracle_cost: usize = oracle_buckets
                    .iter()
                    .map(|(m, b)| index[*m].as_ref().unwrap().sizes[*b])
                    .sum::<usize>()
                    + plain_cost;
                let rank_cost: usize = depth
                    .iter()
                    .map(|(m, d)| {
                        let sizes = &index[*m].as_ref().unwrap().sizes;
                        rankings[*m].as_ref().unwrap()[..*d]
                            .iter()
                            .map(|b| sizes[*b])
                            .sum::<usize>()
                    })
                    .sum::<usize>()
                    + plain_cost;
                ref_row[format!("oracle_cost_top{k}")] = oracle_cost.into();
                ref_row[format!("rank_cost_top{k}")] = rank_cost.into();
                ref_row[format!("oracle_buckets_top{k}")] = oracle_buckets.len().into();
                ref_row[format!("rank_depth_top{k}")] =
                    serde_json::json!(depth.iter().map(|(m, d)| (index[*m].as_ref().unwrap().level, *d)).collect::<Vec<_>>());
            }
            // Static reference: same costs in the flat model.
            if let (Some((static_segment, _)), Some(sidx)) = (&static_ref, &static_index) {
                let ranking = member_ranking(static_segment, query);
                for k in [10usize, 30] {
                    let mut buckets = HashSet::new();
                    let mut d = 0;
                    for id in &gold[..k] {
                        let off = static_segment
                            .id_tracker
                            .borrow()
                            .internal_id_with_behavior((*id).into(), DeferredBehavior::VisibleOnly)
                            .unwrap();
                        let b = sidx.bucket_of[&off];
                        buckets.insert(b);
                        d = d.max(ranking.iter().position(|x| *x == b).unwrap() + 1);
                    }
                    ref_row[format!("static_oracle_cost_top{k}")] =
                        buckets.iter().map(|b| sidx.sizes[*b]).sum::<usize>().into();
                    ref_row[format!("static_rank_cost_top{k}")] =
                        ranking[..d].iter().map(|b| sidx.sizes[*b]).sum::<usize>().into();
                }
            }
            writeln!(refs, "{ref_row}").unwrap();

            let mut configs: Vec<(String, usize, Evaluated)> = Vec::new();
            for &budget in &budgets {
                for (name, policy) in policies()
                    .into_iter()
                    .filter(|(name, _)| policy_filter.as_ref().is_none_or(|f| f.contains(*name)))
                {
                    let evaluated = match policy {
                        Some(policy) => run_route(&members, query, arity, budget, policy),
                        None => run_proportional(&members, query, arity, budget),
                    };
                    configs.push((name.to_owned(), budget, evaluated));
                }
                if let Some((static_segment, _)) = static_ref
                    .as_ref()
                    .filter(|_| policy_filter.as_ref().is_none_or(|f| f.contains("static_flat")))
                {
                    let single = [(1usize, static_segment)];
                    let evaluated = run_route(&single, query, arity, budget, RoutingPolicy::SlotWeighted);
                    configs.push(("static_flat".to_owned(), budget, evaluated));
                }
            }
            for (name, budget, e) in configs {
                let flat = name == "static_flat";
                let hit_set: HashSet<u64> = e.hits.iter().copied().collect();
                let recall = |k: usize| gold[..k].iter().filter(|id| e.hits[..k.min(e.hits.len())].contains(id)).count();
                // Miss attribution for top-30 gold neighbours (DLI topology only).
                let (mut miss_plain_unselected, mut miss_bucket_unselected, mut miss_lost) = (0, 0, 0);
                let mut unselected_ranks: Vec<(usize, usize)> = Vec::new();
                if !flat {
                    for (id, owner) in gold.iter().zip(&owners) {
                        if hit_set.contains(id) {
                            continue;
                        }
                        match *owner {
                            Owner::Plain(uuid) => {
                                if e.selected.contains(&(uuid, None)) {
                                    miss_lost += 1;
                                } else {
                                    miss_plain_unselected += 1;
                                }
                            }
                            Owner::Learned { member, bucket } => {
                                let m = index[member].as_ref().unwrap();
                                if e.selected.contains(&(m.uuid, Some(bucket))) {
                                    miss_lost += 1;
                                } else {
                                    miss_bucket_unselected += 1;
                                    let rank = rankings[member].as_ref().unwrap().iter().position(|b| *b == bucket).unwrap();
                                    unselected_ranks.push((m.level, rank));
                                }
                            }
                        }
                    }
                }
                let row = serde_json::json!({
                    "set": set.label, "query": qi, "policy": name, "budget": budget,
                    "recall10": recall(10), "recall30": recall(30),
                    "posting_entries": e.posting_entries, "unique_offsets": e.unique_offsets,
                    "original_scored": e.original_scored, "plain_scored": e.plain_scored,
                    "scored_total": e.original_scored + e.plain_scored,
                    "buckets_per_level": e.buckets_per_level,
                    "miss_plain_unselected": miss_plain_unselected,
                    "miss_bucket_unselected": miss_bucket_unselected,
                    "miss_selected_but_lost": miss_lost,
                    "unselected_level_rank": unselected_ranks,
                    "elapsed_ns": e.elapsed_ns,
                });
                writeln!(rows, "{row}").unwrap();
            }
        }
        println!("FRONTIER_SET_DONE set={} queries={}", set.label, set.queries.len());
    }
}
