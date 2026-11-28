//! CLI Stage A (research/dli/cli/CLI_QDRANT_DESIGN.md §6b): the whole algorithm of CLI_SHARED_SPEC.md v2 on an
//! in-memory, frozen corpus. No segment, persistence or concurrency; maintenance rewrites this core's state
//! wholesale, which is never reported as an incremental-maintenance cost.
//!
//! Two routings share every maintenance rule:
//! * `Routing::Mlp`: the CLI router (d → hidden → |B|), fine-tuned with stored-label replay (spec §4.2–§4.3);
//! * `Routing::Centroid`: the diagnostic control of spec §8. Buckets route by cosine to their spherical k-means
//!   centroid; a split or recluster replaces centroids instead of fine-tuning. Entropy is not defined for this
//!   router, so its reclustering trigger is growth only (declared).

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use super::cli::{
    DriftSignal, Trigger, dedrift_select, entropy_bits, grow_output, initial_buckets, recluster_k, routing_accuracy_loss,
    shrink_output, split_k, split_sample, training_sample, trigger,
};
use super::{LmiConfig, MlpRouter};
use crate::common::operation_error::{OperationError, OperationResult};

const KMEANS_ITERATIONS: usize = 20;
const LROUTE_SAMPLE: usize = 20_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingKind {
    Mlp,
    Centroid,
}

#[derive(Debug, Clone)]
pub struct CliParams {
    pub b_max: usize,
    pub hidden: usize,
    pub build_epochs: usize,
    pub finetune_epochs: usize,
    pub batch_size: usize,
    pub seed: u64,
    pub signal: DriftSignal,
    pub routing: RoutingKind,
    /// k-means assignment threads (labels are bit-identical for every thread count).
    pub threads: usize,
    /// Replay size (spec §0): `Some(20)` = v3's ⌊0.02 · live_t⌋; `None` = v2's ⌊n_init / 20⌋.
    pub replay_permille: Option<usize>,
}

impl CliParams {
    /// Spec §2 and Table 3.2 defaults at a given b_max.
    pub fn thesis(b_max: usize, routing: RoutingKind, signal: DriftSignal) -> Self {
        Self { b_max, hidden: 128, build_epochs: 30, finetune_epochs: 20, batch_size: 256, seed: 42, signal, routing, threads: 8, replay_permille: Some(20) }
    }
}

enum Routing {
    Mlp(MlpRouter),
    /// One unit centroid per bucket.
    Centroid(Vec<Vec<f32>>),
}

/// What one maintenance step did (spec §7).
#[derive(Debug, Clone, serde::Serialize)]
pub struct MaintenanceRecord {
    pub buckets_before: usize,
    pub buckets_after: usize,
    pub live: usize,
    pub max_before: usize,
    pub max_after: usize,
    pub max_over_mean_before: f64,
    pub trigger: String,
    pub path: String,
    pub h_batch: Option<f64>,
    pub h_init: f64,
    pub split_buckets: usize,
    pub reclustered_buckets: usize,
    pub created: usize,
    pub removed: usize,
    pub reassigned: usize,
    pub replay: usize,
    pub train_rows: usize,
    pub train_seconds: f64,
    pub maintenance_seconds: f64,
    pub l_route: Option<f64>,
    pub residual_overflow: usize,
}

pub struct CliCore<'a> {
    params: CliParams,
    dim: usize,
    /// Row-major normalized vectors of every point id that can ever be inserted (id = row).
    vectors: &'a [f32],
    routing: Routing,
    /// Bucket of each id (`u32::MAX` = not indexed).
    assign: Vec<u32>,
    live: Vec<bool>,
    /// Member ids per bucket; may hold dead ids until maintenance (lazy removal, spec §3).
    buckets: Vec<Vec<u32>>,
    h_init: f64,
    buckets_last: usize,
    /// v2 replay size, ⌊n_init / 20⌋, used when `params.replay_permille` is `None`.
    replay_fixed: usize,
    rng: StdRng,
    /// Entropies of the current batch's inserts (MLP only).
    batch_entropy: Vec<f64>,
    stopped: AtomicBool,
}

fn row<'v>(vectors: &'v [f32], dim: usize, id: u32) -> &'v [f32] {
    &vectors[id as usize * dim..(id as usize + 1) * dim]
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (i, &z) in logits.iter().enumerate() {
        if z > logits[best] {
            best = i;
        }
    }
    best
}

impl<'a> CliCore<'a> {
    fn logits(&self, x: &[f32]) -> Vec<f32> {
        match &self.routing {
            Routing::Mlp(r) => r.forward(x).expect("router forward"),
            Routing::Centroid(c) => c.iter().map(|c| dot(x, c)).collect(),
        }
    }

    fn n_buckets(&self) -> usize {
        self.buckets.len()
    }

    fn kmeans(&self, ids: &[u32], k: usize) -> OperationResult<(Vec<i64>, Vec<Vec<f32>>)> {
        let mut data = Vec::with_capacity(ids.len() * self.dim);
        for &id in ids {
            data.extend_from_slice(row(self.vectors, self.dim, id));
        }
        let config = LmiConfig {
            n_buckets: k,
            sample_size: ids.len().max(k),
            kmeans_iterations: KMEANS_ITERATIONS,
            seed: self.params.seed,
            ..Default::default()
        };
        let (labels, centers, _) =
            super::spherical_kmeans::cluster_with_centers_qdrant(&data, self.dim, &config, self.params.threads, &self.stopped)?;
        let centers = centers.into_iter().map(|c| c.into_iter().map(|v| v as f32).collect()).collect();
        Ok((labels, centers))
    }

    fn train(&self, ids: &[u32], labels: &[i64], buckets: usize, epochs: usize, init: Option<&MlpRouter>) -> OperationResult<MlpRouter> {
        let mut data = Vec::with_capacity(ids.len() * self.dim);
        for &id in ids {
            data.extend_from_slice(row(self.vectors, self.dim, id));
        }
        let config = LmiConfig {
            n_buckets: buckets,
            sample_size: ids.len().max(buckets),
            hidden_dim: self.params.hidden,
            epochs,
            batch_size: self.params.batch_size,
            seed: self.params.seed,
            ..Default::default()
        };
        let trained = super::training::train_labeled_from(&data, labels, self.dim, &config, init, &self.stopped, None)?;
        Ok(trained.native)
    }

    /// Spec §2 / v2 §0: build on `init_ids` (the first plan batch). Returns the build record.
    pub fn build(params: CliParams, dim: usize, vectors: &'a [f32], capacity: usize, init_ids: &[u32]) -> OperationResult<(Self, MaintenanceRecord)> {
        let started = Instant::now();
        let n = init_ids.len();
        let b = initial_buckets(n, params.b_max);
        let s = training_sample(b, params.b_max, n);
        if s < b || s == n {
            return Err(OperationError::service_error(format!("CLI build rejected: S = {s}, B = {b}, n = {n} (v2 §0)")));
        }
        let mut rng = StdRng::seed_from_u64(params.seed);
        let mut order: Vec<u32> = init_ids.to_vec();
        order.sort_unstable();
        order.shuffle(&mut rng);
        let (sample, holdout) = order.split_at(s);
        let mut core = Self {
            dim,
            vectors,
            routing: Routing::Centroid(Vec::new()),
            assign: vec![u32::MAX; capacity],
            live: vec![false; capacity],
            buckets: vec![Vec::new(); b],
            h_init: 0.0,
            buckets_last: b,
            replay_fixed: n / 20,
            rng,
            batch_entropy: Vec::new(),
            stopped: AtomicBool::new(false),
            params,
        };
        let (labels, centers) = core.kmeans(sample, b)?;
        let train_started = Instant::now();
        core.routing = match core.params.routing {
            RoutingKind::Mlp => Routing::Mlp(core.train(sample, &labels, b, core.params.build_epochs, None)?),
            RoutingKind::Centroid => Routing::Centroid(centers),
        };
        let train_seconds = train_started.elapsed().as_secs_f64();
        if core.params.routing == RoutingKind::Mlp {
            let h: f64 = holdout.iter().map(|&id| entropy_bits(&core.logits(row(vectors, dim, id)))).sum();
            core.h_init = h / holdout.len() as f64;
        }
        for &id in init_ids {
            core.place(id);
        }
        let sizes = core.census();
        let rec = MaintenanceRecord {
            buckets_before: 0,
            buckets_after: b,
            live: n,
            max_before: 0,
            max_after: sizes.iter().copied().max().unwrap_or(0),
            max_over_mean_before: 0.0,
            trigger: "build".into(),
            path: "build".into(),
            h_batch: None,
            h_init: core.h_init,
            split_buckets: 0,
            reclustered_buckets: 0,
            created: b,
            removed: 0,
            reassigned: n,
            replay: 0,
            train_rows: s,
            train_seconds,
            maintenance_seconds: started.elapsed().as_secs_f64(),
            l_route: None,
            residual_overflow: sizes.iter().filter(|&&x| x > core.params.b_max).count(),
        };
        Ok((core, rec))
    }

    fn place(&mut self, id: u32) {
        let bucket = argmax(&self.logits(row(self.vectors, self.dim, id)));
        self.assign[id as usize] = bucket as u32;
        self.live[id as usize] = true;
        self.buckets[bucket].push(id);
    }

    /// Spec §3: insert by argmax; records the entropy under the current (pre-maintenance) router.
    pub fn insert(&mut self, ids: &[u32]) {
        for &id in ids {
            assert!(!self.live[id as usize], "id {id} inserted twice");
            if self.params.routing == RoutingKind::Mlp {
                let z = self.logits(row(self.vectors, self.dim, id));
                self.batch_entropy.push(entropy_bits(&z));
            }
            self.place(id);
        }
    }

    /// Spec §3: delete marks dead; bucket lists keep the id until maintenance (lazy removal).
    pub fn delete(&mut self, ids: &[u32]) {
        for &id in ids {
            assert!(self.live[id as usize], "id {id} deleted while not live");
            self.live[id as usize] = false;
            self.assign[id as usize] = u32::MAX;
        }
    }

    fn live_members(&self, bucket: usize) -> Vec<u32> {
        self.buckets[bucket].iter().copied().filter(|&id| self.live[id as usize] && self.assign[id as usize] == bucket as u32).collect()
    }

    /// Live size of every allocated bucket.
    pub fn census(&self) -> Vec<usize> {
        (0..self.n_buckets()).map(|b| self.live_members(b).len()).collect()
    }

    fn sample_without_replacement(&mut self, ids: &[u32], n: usize) -> Vec<u32> {
        let mut v = ids.to_vec();
        v.shuffle(&mut self.rng);
        v.truncate(n);
        v.sort_unstable();
        v
    }

    /// v2 §0, §4.2: replay quotas over the eligible (non-affected, non-empty) buckets, ascending; labels = stored.
    fn replay(&mut self, affected: &[usize]) -> (Vec<u32>, Vec<i64>) {
        let sizes = self.census();
        let eligible: Vec<usize> = (0..sizes.len()).filter(|b| !affected.contains(b) && sizes[*b] > 0).collect();
        let live: usize = sizes.iter().sum();
        let want = match self.params.replay_permille {
            Some(permille) => live * permille / 1000,
            None => self.replay_fixed,
        };
        if eligible.is_empty() || want == 0 {
            return (Vec::new(), Vec::new());
        }
        let target = want.min(eligible.iter().map(|&b| sizes[b]).sum());
        let mut quota = vec![0usize; eligible.len()];
        let mut left = target;
        while left > 0 {
            let open: Vec<usize> = (0..eligible.len()).filter(|&i| quota[i] < sizes[eligible[i]]).collect();
            if open.is_empty() {
                break;
            }
            let each = (left / open.len()).max(1);
            for &i in &open {
                if left == 0 {
                    break;
                }
                let give = each.min(sizes[eligible[i]] - quota[i]).min(left);
                quota[i] += give;
                left -= give;
            }
        }
        let (mut ids, mut labels) = (Vec::new(), Vec::new());
        for (i, &b) in eligible.iter().enumerate() {
            let members = self.live_members(b);
            for id in self.sample_without_replacement(&members, quota[i]) {
                ids.push(id);
                labels.push(b as i64);
            }
        }
        (ids, labels)
    }

    /// Spec §4 / v2 §0: one maintenance step after a committed plan batch.
    pub fn maintain(&mut self) -> OperationResult<MaintenanceRecord> {
        let started = Instant::now();
        let sizes = self.census();
        let live: usize = sizes.iter().sum();
        let b0 = self.n_buckets();
        let max_before = sizes.iter().copied().max().unwrap_or(0);
        let h_batch = (!self.batch_entropy.is_empty()).then(|| self.batch_entropy.iter().sum::<f64>() / self.batch_entropy.len() as f64);
        self.batch_entropy.clear();
        // The centroid control has no entropy: its drift signal is growth only (spec §8, declared).
        let fired = if live == 0 {
            Trigger::None
        } else {
            match (self.params.routing, self.params.signal, h_batch) {
                (RoutingKind::Centroid, _, _) => trigger(DriftSignal::Entropy, f64::NEG_INFINITY, 0.0, &sizes, self.buckets_last),
                (_, DriftSignal::Entropy, None) => trigger(DriftSignal::Entropy, f64::NEG_INFINITY, 0.0, &sizes, self.buckets_last),
                (_, signal, h) => trigger(signal, h.unwrap_or(f64::NEG_INFINITY), self.h_init, &sizes, self.buckets_last),
            }
        };
        let mut rec = MaintenanceRecord {
            buckets_before: b0,
            buckets_after: b0,
            live,
            max_before,
            max_after: max_before,
            max_over_mean_before: if live > 0 { max_before as f64 * b0 as f64 / live as f64 } else { 0.0 },
            trigger: format!("{fired:?}"),
            path: "none".into(),
            h_batch,
            h_init: self.h_init,
            split_buckets: 0,
            reclustered_buckets: 0,
            created: 0,
            removed: 0,
            reassigned: 0,
            replay: 0,
            train_rows: 0,
            train_seconds: 0.0,
            maintenance_seconds: 0.0,
            l_route: None,
            residual_overflow: 0,
        };
        // Selected buckets and, per selected group, (sample ids, k-means labels local to the group, centers).
        let recluster = fired != Trigger::None;
        let (affected, groups): (Vec<usize>, Vec<(Vec<usize>, Vec<u32>, Vec<i64>, Vec<Vec<f32>>)>) = if recluster {
            let sel = dedrift_select(&sizes, self.params.b_max);
            if sel.is_empty() {
                (Vec::new(), Vec::new())
            } else {
                let pooled: Vec<u32> = sel.iter().flat_map(|&b| self.live_members(b)).collect();
                let (k_new, samp) = recluster_k(pooled.len(), self.params.b_max);
                if k_new > samp {
                    return Err(OperationError::service_error("CLI recluster rejected: more clusters than samples (v2 §0)"));
                }
                let sample = self.sample_without_replacement(&pooled, samp);
                let (labels, centers) = self.kmeans(&sample, k_new)?;
                (sel.clone(), vec![(sel, sample, labels, centers)])
            }
        } else {
            let overflow: Vec<usize> = (0..b0).filter(|&b| sizes[b] > self.params.b_max).collect();
            let mut groups = Vec::new();
            for &b in &overflow {
                let k = split_k(sizes[b], self.params.b_max).unwrap();
                let members = self.live_members(b);
                let sample = self.sample_without_replacement(&members, split_sample(members.len(), k));
                let (labels, centers) = self.kmeans(&sample, k)?;
                groups.push((vec![b], sample, labels, centers));
            }
            (overflow, groups)
        };
        if affected.is_empty() {
            rec.path = "none".into();
            rec.maintenance_seconds = started.elapsed().as_secs_f64();
            return Ok(rec);
        }
        rec.path = if recluster { "recluster".into() } else { "split".into() };
        if recluster {
            rec.reclustered_buckets = affected.len();
        } else {
            rec.split_buckets = affected.len();
        }
        // Map each group's local clusters to stable bucket ids: the group's own ids first (ascending), then new ids;
        // new output rows copy their donor (spec §4.2 parent; §4.3 pre-maintenance argmax of the centroid).
        let mut train_ids = Vec::new();
        let mut train_labels = Vec::new();
        let mut new_router = match &self.routing {
            Routing::Mlp(r) => Some(r.clone()),
            Routing::Centroid(_) => None,
        };
        let mut new_centers = match &self.routing {
            Routing::Centroid(c) => Some(c.clone()),
            Routing::Mlp(_) => None,
        };
        let mut next = b0;
        for (own, sample, labels, centers) in &groups {
            let mut target = Vec::with_capacity(centers.len());
            for (c, center) in centers.iter().enumerate() {
                if c < own.len() {
                    target.push(own[c]);
                    if let Some(cs) = new_centers.as_mut() {
                        cs[own[c]] = center.clone();
                    }
                } else {
                    let donor = if recluster { argmax(&self.logits(center)) } else { own[0] };
                    if let Some(r) = new_router.as_mut() {
                        grow_output(r, donor, 1)?;
                    }
                    if let Some(cs) = new_centers.as_mut() {
                        cs.push(center.clone());
                    }
                    target.push(next);
                    next += 1;
                }
            }
            for (&id, &l) in sample.iter().zip(labels) {
                train_ids.push(id);
                train_labels.push(target[l as usize] as i64);
            }
        }
        rec.created = next - b0;
        self.buckets.resize(next, Vec::new());
        // Fine-tune the MLP from its current (grown) weights on the group samples plus stored-label replay.
        let train_started = Instant::now();
        if let Some(r) = new_router.take() {
            let (rids, rlabels) = self.replay(&affected);
            rec.replay = rids.len();
            train_ids.extend(rids);
            train_labels.extend(rlabels);
            rec.train_rows = train_ids.len();
            let tuned = self.train(&train_ids, &train_labels, next, self.params.finetune_epochs, Some(&r))?;
            self.routing = Routing::Mlp(tuned);
        } else if let Some(cs) = new_centers.take() {
            self.routing = Routing::Centroid(cs);
        }
        rec.train_seconds = train_started.elapsed().as_secs_f64();
        // Reinsert the affected buckets' live points by argmax over the full new router; untouched points stay.
        // The L_route probe is drawn from the untouched points before any mover can enter an untouched bucket.
        let untouched: Vec<u32> =
            (0..b0).filter(|b| !affected.contains(b)).flat_map(|b| self.live_members(b)).collect();
        let probe = self.sample_without_replacement(&untouched, LROUTE_SAMPLE);
        let stored: Vec<usize> = probe.iter().map(|&id| self.assign[id as usize] as usize).collect();
        let movers: Vec<u32> = affected.iter().flat_map(|&b| self.live_members(b)).collect();
        for &b in &affected {
            self.buckets[b].clear();
        }
        for &id in &movers {
            self.place(id);
        }
        rec.reassigned = movers.len();
        let now: Vec<usize> = probe.iter().map(|&id| argmax(&self.logits(row(self.vectors, self.dim, id)))).collect();
        rec.l_route = (!probe.is_empty() && self.params.routing == RoutingKind::Mlp).then(|| routing_accuracy_loss(&stored, &now));
        // Lazy removal: compact lists, then drop empty buckets and their neurons (never all of them).
        for b in 0..self.n_buckets() {
            let keep = self.live_members(b);
            self.buckets[b] = keep;
        }
        let empty: Vec<usize> = (0..self.n_buckets()).filter(|&b| self.buckets[b].is_empty()).collect();
        if !empty.is_empty() && empty.len() < self.n_buckets() {
            let map: Vec<Option<usize>> = match &mut self.routing {
                Routing::Mlp(r) => shrink_output(r, &empty)?,
                Routing::Centroid(cs) => {
                    let mut map = vec![None; cs.len()];
                    let mut kept = Vec::new();
                    for (b, c) in cs.drain(..).enumerate() {
                        if !empty.contains(&b) {
                            map[b] = Some(kept.len());
                            kept.push(c);
                        }
                    }
                    *cs = kept;
                    map
                }
            };
            let old = std::mem::take(&mut self.buckets);
            self.buckets = vec![Vec::new(); old.len() - empty.len()];
            for (b, members) in old.into_iter().enumerate() {
                if let Some(nb) = map[b] {
                    for &id in &members {
                        self.assign[id as usize] = nb as u32;
                    }
                    self.buckets[nb] = members;
                }
            }
            rec.removed = empty.len();
        }
        if recluster {
            self.buckets_last = self.n_buckets();
        }
        let after = self.census();
        rec.buckets_after = after.len();
        rec.max_after = after.iter().copied().max().unwrap_or(0);
        rec.residual_overflow = after.iter().filter(|&&x| x > self.params.b_max).count();
        self.check_membership()?;
        rec.maintenance_seconds = started.elapsed().as_secs_f64();
        Ok(rec)
    }

    /// G2: every live id is in exactly one bucket (its assigned one) and the sizes sum to the live count.
    pub fn check_membership(&self) -> OperationResult<()> {
        let mut seen = vec![false; self.live.len()];
        let mut total = 0usize;
        for (b, members) in self.buckets.iter().enumerate() {
            for &id in members {
                if !self.live[id as usize] || self.assign[id as usize] != b as u32 {
                    continue;
                }
                if seen[id as usize] {
                    return Err(OperationError::service_error(format!("G2: id {id} in two buckets")));
                }
                seen[id as usize] = true;
                total += 1;
            }
        }
        let live = self.live.iter().filter(|&&l| l).count();
        if total != live || self.live.iter().zip(&seen).any(|(&l, &s)| l != s) {
            return Err(OperationError::service_error(format!("G2: {total} placed vs {live} live")));
        }
        if let Routing::Mlp(r) = &self.routing
            && r.output_dim()? != self.n_buckets()
        {
            return Err(OperationError::service_error("G2: router outputs differ from the bucket directory"));
        }
        if let Routing::Centroid(c) = &self.routing
            && c.len() != self.n_buckets()
        {
            return Err(OperationError::service_error("G2: centroids differ from the bucket directory"));
        }
        Ok(())
    }

    /// Spec §1: whole buckets by logit (ties by lower id) until the live candidates reach each budget. Returns, per
    /// budget (ascending), the vectors scored and the top `k` ids by cosine (ties by lower id).
    pub fn search(&self, query: &[f32], budgets: &[usize], k: usize) -> Vec<(usize, Vec<u32>)> {
        let z = self.logits(query);
        let mut order: Vec<usize> = (0..z.len()).collect();
        order.sort_by(|&a, &b| z[b].total_cmp(&z[a]).then(a.cmp(&b)));
        let mut scored: Vec<(f32, u32)> = Vec::new();
        let mut out = Vec::with_capacity(budgets.len());
        let mut bi = 0;
        let top = |scored: &[(f32, u32)]| {
            let mut v = scored.to_vec();
            v.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            v.truncate(k);
            v.into_iter().map(|(_, id)| id).collect::<Vec<_>>()
        };
        for b in order {
            for &id in &self.buckets[b] {
                if self.live[id as usize] && self.assign[id as usize] == b as u32 {
                    scored.push((dot(query, row(self.vectors, self.dim, id)), id));
                }
            }
            while bi < budgets.len() && scored.len() >= budgets[bi] {
                out.push((scored.len(), top(&scored)));
                bi += 1;
            }
            if bi == budgets.len() {
                break;
            }
        }
        while bi < budgets.len() {
            out.push((scored.len(), top(&scored)));
            bi += 1;
        }
        out
    }

    pub fn buckets(&self) -> usize {
        self.n_buckets()
    }
}
