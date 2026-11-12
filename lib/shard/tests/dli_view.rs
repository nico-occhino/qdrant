#![cfg(feature = "testing")]

//! Research-only independent segment probing, not a DLI coordinator or publication protocol.
use common::flags::FeatureFlags;
use segment::data_types::query_context::VectorQueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use segment::entry::entry_point::{ReadSegmentEntry, SegmentEntry};
use segment::id_tracker::IdTrackerRead;
use segment::index::VectorIndexEnum;
use segment::index::lmi_index::LmiConfig;
use segment::segment::Segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{Distance, HnswGlobalConfig, Indexes};
use std::sync::atomic::AtomicBool;

const DATA: [(u64, [f32; 2]); 9] = [
    (11, [1.0, 0.0]),
    (12, [0.8, 0.6]),
    (13, [0.0, 1.0]),
    (14, [-0.6, 0.8]),
    (21, [-1.0, 0.0]),
    (22, [-0.8, -0.6]),
    (23, [0.0, -1.0]),
    (24, [0.6, -0.8]),
    (31, [0.6, 0.8]),
];

fn lmi(root: &std::path::Path, rows: &[(u64, [f32; 2])], seed: u64) -> Segment {
    let staging = tempfile::tempdir().unwrap();
    let mut src = build_simple_segment(root, 2, Distance::Cosine).unwrap();
    for (id, vector) in rows {
        src.upsert_point(*id, (*id).into(), only_default_vector(vector))
            .unwrap();
    }
    let mut cfg = src.config().clone();
    cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().index = Indexes::LmiTrained(LmiConfig {
        n_buckets: 2,
        sample_size: 4,
        hidden_dim: 4,
        epochs: 1,
        batch_size: 4,
        routing_batch_size: 4,
        kmeans_iterations: 2,
        nprobe: 2,
        seed,
    });
    let mut builder = SegmentBuilder::new(
        staging.path(),
        &cfg,
        &HnswGlobalConfig::default(),
        FeatureFlags::default(),
    )
    .unwrap();
    builder.update(&[&src], &AtomicBool::new(false)).unwrap();
    builder.build_for_test(root)
}

use parking_lot::{Mutex, RwLock};
use shard::locked_segment::LockedSegment;
use shard::operations::point_ops::{PointStructPersisted, VectorStructPersisted};
use shard::segment_holder::locked::LockedSegmentHolder;
use shard::segment_holder::{SegmentHolder, SegmentId};
use shard::update::{delete_points, upsert_points};
use std::sync::{Arc, Barrier};
use uuid::Uuid;

#[derive(Clone)]
struct Member {
    level: u64,
    level_generation: u64,
    holder: SegmentId,
    physical: Uuid,
}
struct Registry {
    generation: u64,
    members: Vec<Member>,
}
struct Controller {
    holder: LockedSegmentHolder,
    registry: Mutex<Registry>,
}
struct Pinned {
    descriptor: Member,
    segment: Arc<RwLock<Segment>>,
}
struct View {
    generation: u64,
    members: Vec<Pinned>,
}
#[derive(Clone)]
struct Route {
    view: u64,
    physical: Uuid,
    member: usize,
    bucket: usize,
    priority: f32,
    offsets: Vec<u32>,
}

impl Controller {
    fn capture(&self) -> Result<View, &'static str> {
        let registry = self.registry.lock();
        let holder = self.holder.read();
        let mut members = Vec::new();
        for d in &registry.members {
            let Some(LockedSegment::Original(segment)) = holder.get(d.holder) else {
                return Err("missing or proxied member");
            };
            let s = segment.read();
            if s.uuid != d.physical {
                return Err("physical generation mismatch");
            }
            let config = &s.config().vector_data[DEFAULT_VECTOR_NAME];
            if config.size != 2 || config.distance != Distance::Cosine {
                return Err("incompatible vectors");
            }
            if d.level > 0 {
                let index = s.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
                let VectorIndexEnum::Lmi(index) = &*index else {
                    return Err("missing LMI");
                };
                let state = index.routing_state().ok_or("missing routing state")?;
                if state.router().output_dim().unwrap() != state.postings().len() {
                    return Err("model/posting mismatch");
                }
            }
            drop(s);
            members.push(Pinned {
                descriptor: d.clone(),
                segment: segment.clone(),
            });
        }
        Ok(View {
            generation: registry.generation,
            members,
        })
    }
    fn replace(&self, member: usize, replacement: Segment) {
        let mut registry = self.registry.lock();
        let physical = replacement.uuid;
        let old = registry.members[member].holder;
        let (id, retired) = self.holder.write().swap_new(replacement, &[old]);
        assert_eq!(retired.len(), 1);
        assert_ne!(old, id);
        registry.generation += 1;
        registry.members[member].holder = id;
        registry.members[member].physical = physical;
        registry.members[member].level_generation += 1;
        // No destructive cleanup: views still pin native segment Arcs.
    }
}
impl View {
    fn route(&self) -> Vec<Route> {
        let mut routes = Vec::new();
        for (member, pin) in self.members.iter().enumerate() {
            let s = pin.segment.read();
            let offsets = if pin.descriptor.level == 0 {
                use segment::vector_storage::VectorStorageRead;
                vec![(
                    1.0,
                    (0..s.vector_data[DEFAULT_VECTOR_NAME]
                        .vector_storage
                        .borrow()
                        .total_vector_count() as u32)
                        .collect(),
                )]
            } else {
                let idx = s.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
                let VectorIndexEnum::Lmi(index) = &*idx else {
                    unreachable!()
                };
                let state = index.routing_state().unwrap();
                let logits = state.router().forward(&[1.0, 0.0]).unwrap();
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let z: f32 = logits.iter().map(|x| (x - max).exp()).sum();
                logits
                    .iter()
                    .enumerate()
                    .map(|(b, x)| {
                        (
                            (x - max).exp() / z * 2_f32.powi(pin.descriptor.level as i32),
                            state.postings().get(b).unwrap().to_vec(),
                        )
                    })
                    .collect()
            };
            for (bucket, (priority, offsets)) in offsets.into_iter().enumerate() {
                if !offsets.is_empty() {
                    routes.push(Route {
                        view: self.generation,
                        physical: s.uuid,
                        member,
                        bucket,
                        priority,
                        offsets,
                    });
                }
            }
        }
        routes.sort_by(|a, b| {
            b.priority
                .total_cmp(&a.priority)
                .then(a.member.cmp(&b.member))
                .then(a.bucket.cmp(&b.bucket))
        });
        routes
    }
    fn score(&self, routes: &[Route], budget: usize) -> Result<Vec<(u64, f32)>, &'static str> {
        let mut results = Vec::new();
        let mut candidates = 0;
        for r in routes {
            // Validate even routes beyond the budget, fail closed on invalid input.
            let pin = self.members.get(r.member).ok_or("invalid member")?;
            if r.view != self.generation || r.physical != pin.descriptor.physical {
                return Err("stale route");
            }
        }
        for r in routes {
            if candidates >= budget {
                break;
            }
            let s = self.members[r.member].segment.read();
            if s.uuid != r.physical {
                return Err("replaced pinned segment");
            }
            let tracker = s.id_tracker.borrow();
            let storage = s.vector_data[DEFAULT_VECTOR_NAME].vector_storage.borrow();
            let q: QueryVector = [1.0_f32, 0.0].into();
            let scores = segment::index::candidate_scoring::score_candidates(
                &q,
                &r.offsets,
                20,
                &tracker,
                &storage,
                &VectorQueryContext::default(),
            )
            .unwrap();
            candidates += r.offsets.len();
            for p in scores {
                let segment::types::PointIdType::NumId(id) = tracker.external_id(p.idx).unwrap()
                else {
                    unreachable!()
                };
                results.push((id, p.score));
            }
        }
        results.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        Ok(results)
    }
}
fn fixture(root: &std::path::Path) -> Controller {
    let mut holder = SegmentHolder::default();
    let mut members = Vec::new();
    let mut buffer = build_simple_segment(root, 2, Distance::Cosine).unwrap();
    buffer
        .upsert_point(31, 31_u64.into(), only_default_vector(&DATA[8].1))
        .unwrap();
    for (level, segment) in [
        buffer,
        lmi(root, &DATA[..4], 17),
        lmi(root, &DATA[4..8], 29),
    ]
    .into_iter()
    .enumerate()
    {
        let physical = segment.uuid;
        let id = holder.add_new(segment);
        members.push(Member {
            level: level as u64,
            level_generation: 1,
            holder: id,
            physical,
        });
    }
    Controller {
        holder: LockedSegmentHolder::new(holder),
        registry: Mutex::new(Registry {
            generation: 1,
            members,
        }),
    }
}
#[test]
fn pinned_view_replacement_interleaving_and_stale_rejection() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("DLI view test"));
    let root = tempfile::tempdir().unwrap();
    let c = Arc::new(fixture(root.path()));
    let old = c.capture().unwrap();
    let routes = old.route();
    let expected = old.score(&routes, usize::MAX).unwrap();
    assert_eq!(expected.len(), 9);
    let small = old.score(&routes, 2).unwrap();
    assert!(!small.is_empty());
    for (id, score) in &small {
        let v = DATA.iter().find(|r| r.0 == *id).unwrap().1;
        assert!((score - v[0]).abs() < 1e-5);
    }
    println!(
        "view={} budget=2 returned={} ids={:?}",
        old.generation,
        small.len(),
        small.iter().map(|p| p.0).collect::<Vec<_>>()
    );
    for (id, score) in &expected {
        let v = DATA.iter().find(|r| r.0 == *id).unwrap().1;
        assert!((score - v[0]).abs() < 1e-5);
    }
    let ready = Arc::new(Barrier::new(2));
    let resume = Arc::new(Barrier::new(2));
    let query = {
        let ready = ready.clone();
        let resume = resume.clone();
        std::thread::spawn(move || {
            let _scope = common::ambient::unmeasured_guard(common::reason::reason("DLI query thread"));
            ready.wait();
            resume.wait();
            old.score(&routes, usize::MAX).unwrap()
        })
    };
    let replacement = lmi(root.path(), &DATA[..4], 43);
    ready.wait();
    c.replace(1, replacement);
    resume.wait();
    assert_eq!(query.join().unwrap(), expected);
    let current = c.capture().unwrap();
    assert_eq!(current.generation, 2);
    assert_eq!(
        current.score(&current.route(), usize::MAX).unwrap(),
        expected
    );
    let mut bad = current.route();
    bad[0].view = 1;
    assert!(current.score(&bad, 2).is_err());
    let mut bad = current.route();
    bad[0].physical = Uuid::nil();
    assert!(current.score(&bad, 2).is_err());
    let stale = c.registry.lock().members[1].holder;
    c.holder
        .write()
        .swap_new(lmi(root.path(), &DATA[..4], 53), &[stale]);
    assert!(
        c.capture().is_err(),
        "uncoordinated replacement must fail closed"
    );
}
#[test]
fn native_updates_are_visible_in_pinned_handles() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("DLI update test"));
    let root = tempfile::tempdir().unwrap();
    let c = fixture(root.path());
    let view = c.capture().unwrap();
    let points = vec![
        PointStructPersisted {
            id: 11_u64.into(),
            vector: VectorStructPersisted::Single(vec![0.0, 1.0]),
            payload: None,
        },
        PointStructPersisted {
            id: 99_u64.into(),
            vector: VectorStructPersisted::Single(vec![1.0, 0.0]),
            payload: None,
        },
    ];
    {
        let _updates = c.holder.acquire_updates_lock();
        let holder = c.holder.read();
        upsert_points(&holder, 100, &points, None).unwrap();
        delete_points(&holder, 101, &[21_u64.into()]).unwrap();
    }
    let results = view.score(&view.route(), usize::MAX).unwrap();
    assert_eq!(results.iter().filter(|p| p.0 == 11).count(), 1);
    assert!(results.iter().find(|p| p.0 == 11).unwrap().1.abs() < 1e-5);
    assert!(results.iter().any(|p| p.0 == 99));
    assert!(!results.iter().any(|p| p.0 == 21));
}
