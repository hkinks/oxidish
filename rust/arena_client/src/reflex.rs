//! Hybrid reflex layer for the vision agents (docs/vision-bot.md, M5). The net keeps the movement; on each decision
//! this layer looks at the agent's Snapshot and, per enabled skill, takes over the aim / split / eject with the rule
//! bots' own code:
//!   * split-kill  (`reflex_split`):   `heuristic_plan` picks the Hunt target, `steer::split_kill_ok` is the exact geometry
//!                                      and safety check of `steer_to` (reach, size, no threat that eats the halves);
//!   * harvest     (`reflex_harvest`): `heuristic_plan` with the harvest gene on; if the chosen target belongs to a player
//!                                      shown in >= 3 pieces (edible: `classify` prey) the aim goes to that cloud;
//!   * virus-split (`reflex_virus`):   `Tactics::act` with only the shooting gene on (no staging walk): a big player
//!                                      lined up behind a virus within shooting range and the shot is safe -> eject at the virus.
//! Every skill is vetoed by the rule bots' own Flee check (inside `heuristic_plan`) or the shooting safety check.
//! Spec keys (all off by default): `reflex_split`, `reflex_harvest`, `reflex_virus` (bool), `reflex_genes` (genome file
//! whose `farm`/`params` give the rule genes, default `bot/c3_4.json`), `reflex_params` (object merged over those genes, e.g.
//! {"flee_margin": 1.0} for a wider berth than the genome's), `reflex_harvest_range` (gene `harvest_range`, default 1),
//! `reflex_split_clear` (split-kill only when no other player cell above half of our mass is within this many world units;
//! default 0 = off), `reflex_guard` (bool): recombine guard, see `guard`, `reflex_hunter` (bool, M6): the lurk-and-harvest hunter
//! (hunter.rs): while its `commit` phase is active it overrides aim / split / eject, otherwise the net drives; genes from
//! `reflex_hunter_genes` (genome file, default `bot/drill/hv_commit_only.json`) + `reflex_hunter_params` (merged over its `params`).

use std::collections::HashMap;

use crate::game::{Kind, Snapshot};
use crate::hunter::{Hunter, Phase};
use crate::params::Params;
use crate::policy::{heuristic_plan, Counters, HeuristicPolicy, Out, Tactics};
use crate::steer::{split_kill_ok, Mode, Plan};

pub struct Reflex {
    pub split: bool,
    pub harvest: bool,
    pub virus: bool,
    harvest_range: f64,
    split_clear: f64,
    guard: bool,
    /// M6 hunter: per-agent state (piece memory, velocities) and its gene tiers
    hunter: Option<(Hunter, HeuristicPolicy)>,
    genes: HeuristicPolicy,
    tactics: Tactics,
    pub counters: Counters,
}

impl Reflex {
    /// None when no skill is enabled in the (vision) spec.
    pub fn from_spec(spec: &serde_json::Value) -> Option<Reflex> {
        let on = |k: &str| spec.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
        let (split, harvest, virus) = (on("reflex_split"), on("reflex_harvest"), on("reflex_virus"));
        if !(split || harvest || virus || on("reflex_guard") || on("reflex_hunter")) {
            return None;
        }
        let path = spec.get("reflex_genes").and_then(|v| v.as_str()).unwrap_or("bot/c3_4.json");
        let mut genome = std::fs::read_to_string(path)
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .unwrap_or_else(|| {
                eprintln!("reflex: genome {path} not readable, using default genes");
                serde_json::json!({"farm": "field"})
            });
        if let Some(over) = spec.get("reflex_params").and_then(|v| v.as_object()) {
            if !genome["params"].is_object() {
                genome["params"] = serde_json::json!({});
            }
            for (k, v) in over {
                genome["params"][k] = v.clone();
            }
        }
        let hr = spec.get("reflex_harvest_range").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let mut r = Reflex::new(split, harvest, virus, hr, &genome);
        r.guard = spec.get("reflex_guard").and_then(|v| v.as_bool()).unwrap_or(false);
        r.split_clear = spec.get("reflex_split_clear").and_then(|v| v.as_f64()).unwrap_or(0.0);
        if on("reflex_hunter") {
            let hp = spec.get("reflex_hunter_genes").and_then(|v| v.as_str()).unwrap_or("bot/drill/hv_commit_only.json");
            let mut hg = std::fs::read_to_string(hp).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()).unwrap_or_else(|| {
                eprintln!("reflex: hunter genome {hp} not readable, using the built-in commit set");
                serde_json::json!({"farm": "field", "params": {"commit": 1, "commit_kite": 1, "split_harvest": 1, "race": 1, "hv_memory": 3, "hv_still": 100}})
            });
            if let Some(over) = spec.get("reflex_hunter_params").and_then(|v| v.as_object()) {
                if !hg["params"].is_object() {
                    hg["params"] = serde_json::json!({});
                }
                for (k, v) in over {
                    hg["params"][k] = v.clone();
                }
            }
            r.set_hunter(&hg);
        }
        Some(r)
    }

    pub fn new(split: bool, harvest: bool, virus: bool, harvest_range: f64, genome: &serde_json::Value) -> Reflex {
        let mut counters = Counters::default();
        for k in ["reflex_split", "reflex_harvest", "reflex_shot", "reflex_hunter"] {
            counters.0.push((k.to_string(), 0));
        }
        Reflex { split, harvest, virus, harvest_range, split_clear: 0.0, guard: false, hunter: None, genes: HeuristicPolicy::from_spec(genome), tactics: Tactics::new(), counters }
    }

    /// Enable the hunter reflex with this genome (`params` hold the commit gene set).
    pub fn set_hunter(&mut self, genome: &serde_json::Value) {
        self.hunter = Some((Hunter::new(), HeuristicPolicy::from_spec(genome)));
    }

    pub fn has_hunter(&self) -> bool {
        self.hunter.is_some()
    }

    pub fn reset(&mut self) {
        self.tactics.reset();
        if let Some((h, _)) = self.hunter.as_mut() {
            h.reset();
        }
    }

    /// The split-kill skill alone, without side effects: the aim point if the rule bots' split-kill check passes now
    /// (also the teacher label of the split head in the PPO env, rl/vision/train_ppo3.py).
    pub fn split_kill(&self, s: &Snapshot) -> Option<(f64, f64)> {
        let me = s.main().copied()?;
        let p = self.genes.params_for(s.total_mass());
        if s.mine.len() as f64 > p.split_max_cells.round_ties_even() {
            return None;
        }
        let plan = heuristic_plan(s, p);
        if plan.mode != Mode::Hunt || !plan.split_ok {
            return None;
        }
        let t = plan.target_id.and_then(|id| s.others.iter().find(|c| c.id == id))?;
        if !split_kill_ok(s, &me, t, p) {
            return None;
        }
        if self.split_clear > 0.0 {
            let total = s.total_mass();
            let crowded = s.others.iter().any(|c| c.kind == Kind::Player && c.id != t.id && c.mass() > total * 0.5 && me.dist(c) < self.split_clear);
            if crowded {
                return None;
            }
        }
        Some((t.x, t.y))
    }

    /// Recombine guard: while we own >= 2 cells, a player cell that outweighs our biggest piece by the eat ratio and is within
    /// reach (its radius + ours + 400) makes us run, as a group, straight away from it (weighted by how close it is) until the
    /// pieces have merged or the danger has passed. None = no danger, the net keeps control.
    pub fn guard_aim(&self, s: &Snapshot) -> Option<(f64, f64)> {
        let me = s.main().copied()?;
        let (mut fx, mut fy, mut any) = (0.0, 0.0, false);
        let (cx, cy) = {
            let n = s.mine.len() as f64;
            (s.mine.iter().map(|c| c.x).sum::<f64>() / n, s.mine.iter().map(|c| c.y).sum::<f64>() / n)
        };
        for t in s.others.iter().filter(|c| c.kind == Kind::Player && c.mass() > me.mass() * 1.3) {
            let gap = (t.x - cx).hypot(t.y - cy) - t.size - me.size;
            if gap < 400.0 {
                let d = (t.x - cx).hypot(t.y - cy).max(1.0);
                let w = 1.0 / (gap.max(0.0) + 100.0);
                fx -= w * (t.x - cx) / d;
                fy -= w * (t.y - cy) / d;
                any = true;
            }
        }
        if !any {
            return None;
        }
        let n = fx.hypot(fy).max(1e-9);
        Some((cx + fx / n * 1500.0, cy + fy / n * 1500.0))
    }

    /// The net's action `a` for this decision, possibly replaced by a reflex.
    pub fn apply(&mut self, s: &Snapshot, a: Out) -> Out {
        let Some(me) = s.main().copied() else { return a };
        let p = self.genes.params_for(s.total_mass()).clone();
        // the hunter must see every snapshot (velocities, piece memory), so it steps first, even when another reflex then wins
        if let Some((h, hg)) = self.hunter.as_mut() {
            let hp = hg.params_for(s.total_mass());
            let out = h.step(s, hp);
            if h.phase == Phase::Commit {
                if let Some(o) = out {
                    self.counters.bump("reflex_hunter", 1);
                    return Out { x: o.x, y: o.y, split: o.split, eject: o.eject };
                }
            }
        }
        if self.guard && s.mine.len() >= 2 {
            if let Some((x, y)) = self.guard_aim(s) {
                self.counters.bump("reflex_guard", 1);
                return Out { x, y, split: false, eject: false };
            }
        }
        if self.split {
            if let Some((x, y)) = self.split_kill(s) {
                self.counters.bump("reflex_split", 1);
                return Out { x, y, split: true, eject: false };
            }
        }
        if self.virus {
            let pv = Params { shoot_enable: 1.0, shoot_stage: 0.0, selffeed_enable: 0.0, double_split_enable: 0.0, hide_enable: 0.0, ..p.clone() };
            let mut scratch = Counters::default();
            if let Some(o) = self.tactics.act(s, &Plan::new(Mode::Farm), &pv, &mut scratch, None) {
                if o.eject {
                    self.counters.bump("reflex_shot", 1);
                    return Out { x: o.x, y: o.y, split: false, eject: true };
                }
            }
        }
        if self.harvest {
            let ph = Params { harvest: 1.0, harvest_range: self.harvest_range, ..p };
            let plan = heuristic_plan(s, &ph);
            if plan.mode == Mode::Hunt {
                if let Some(t) = plan.target_id.and_then(|id| s.others.iter().find(|c| c.id == id)) {
                    let mut pieces: HashMap<u32, usize> = HashMap::new();
                    for c in s.others.iter().filter(|c| c.kind == Kind::Player && c.color != 0) {
                        *pieces.entry(c.color).or_insert(0) += 1;
                    }
                    if t.color != 0 && pieces.get(&t.color).copied().unwrap_or(0) >= 3 {
                        self.counters.bump("reflex_harvest", 1);
                        return Out { x: t.x, y: t.y, split: a.split, eject: a.eject };
                    }
                }
            }
        }
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Border, Cell};

    fn snap(others: Vec<Cell>, mass: f64) -> Snapshot {
        let me = Cell::new(1, 0.0, 0.0, (mass * 100.0).sqrt(), Kind::Mine);
        Snapshot { mine: vec![me], others, border: Border { left: -5000.0, right: 5000.0, top: -5000.0, bottom: 5000.0 }, dead: false }
    }
    fn player(id: u32, x: f64, y: f64, mass: f64, color: u32) -> Cell {
        let mut c = Cell::new(id, x, y, (mass * 100.0).sqrt(), Kind::Player);
        c.color = color;
        c
    }
    fn genome() -> serde_json::Value {
        serde_json::json!({"farm": "field", "params": {"flee_margin": 1.0}})
    }
    /// the net wants to go west
    fn net() -> Out {
        Out { x: -2000.0, y: 0.0, split: false, eject: false }
    }
    fn reflex(split: bool, harvest: bool, virus: bool) -> Reflex {
        Reflex::new(split, harvest, virus, 1.0, &genome())
    }

    #[test]
    fn split_kill_fires_on_a_reachable_small_target() {
        let s = snap(vec![player(10, 800.0, 0.0, 400.0, 0xff0000)], 3000.0);
        let o = reflex(true, false, false).apply(&s, net());
        assert!(o.split && o.x > 0.0, "expected a split at the prey, got split={} x={}", o.split, o.x);
        // flag off: the net keeps its action
        let o = reflex(false, true, true).apply(&s, net());
        assert!(!o.split && o.x < 0.0);
    }

    #[test]
    fn split_kill_refuses_when_the_target_is_too_far_too_big_or_a_threat_would_eat_the_halves() {
        let mut r = reflex(true, false, false);
        // out of the split reach (gap 1300 - 173 - 547 > 520 and hunt range)
        assert!(!r.apply(&snap(vec![player(10, 2000.0, 0.0, 400.0, 1)], 3000.0), net()).split);
        // too big to eat with the half (1500 / 1.3 = 1154 < 1300)
        assert!(!r.apply(&snap(vec![player(10, 900.0, 0.0, 1300.0, 1)], 3000.0), net()).split);
        // a 9000 giant near the prey: the rule bots flee instead, no split
        let s = snap(vec![player(10, 800.0, 0.0, 400.0, 1), player(11, 0.0, 1500.0, 9000.0, 2)], 3000.0);
        let o = r.apply(&s, net());
        assert!(!o.split && o.x < 0.0, "a threat in reach must veto the split");
    }

    #[test]
    fn harvest_aims_at_the_cloud_of_pieces() {
        let mut others = vec![player(10, 900.0, 0.0, 700.0, 0xff0000)];
        for i in 0..5 {
            others.push(player(20 + i, -900.0 - 40.0 * i as f64, 30.0 * i as f64, 300.0, 0x00ff00));
        }
        let s = snap(others, 3000.0);
        // the net goes east, towards the lone prey: the reflex turns the aim to the pieces, and does not split by itself
        let east = Out { x: 2000.0, y: 0.0, split: false, eject: false };
        let o = reflex(false, true, false).apply(&s, east);
        assert!(o.x < 0.0 && !o.split, "aim {} should be west, at the pieces", o.x);
        // with the flag off the aim is untouched
        let east = Out { x: 2000.0, y: 0.0, split: false, eject: false };
        assert!(reflex(true, false, true).apply(&s, east).x > 0.0);
    }

    #[test]
    fn harvest_ignores_a_player_in_fewer_than_three_pieces() {
        let others = vec![player(10, 900.0, 0.0, 700.0, 0xff0000), player(20, -900.0, 0.0, 300.0, 0x00ff00), player(21, -940.0, 0.0, 300.0, 0x00ff00)];
        let s = snap(others, 3000.0);
        let o = reflex(false, true, false).apply(&s, Out { x: 2000.0, y: 0.0, split: false, eject: false });
        assert!(o.x > 0.0, "two pieces are not a cloud");
    }

    #[test]
    fn harvest_is_vetoed_by_a_threat() {
        let mut others = vec![player(30, 0.0, 700.0, 9000.0, 0x0000ff)];
        for i in 0..4 {
            others.push(player(20 + i, -900.0 - 40.0 * i as f64, 0.0, 300.0, 0x00ff00));
        }
        let s = snap(others, 3000.0);
        let o = reflex(false, true, false).apply(&s, Out { x: 5.0, y: -2000.0, split: false, eject: false });
        assert_eq!(o.y, -2000.0, "a giant next to us: the net keeps control");
    }

    #[test]
    fn virus_shot_when_a_big_player_is_lined_up_behind_a_virus() {
        let virus = Cell::new(30, 1000.0, 0.0, 100.0, Kind::Virus);
        let s = snap(vec![virus, player(10, 1800.0, 0.0, 2000.0, 0xff0000)], 1000.0);
        let o = reflex(false, false, true).apply(&s, net());
        assert!(o.eject && o.x > 0.0, "expected a shot at the virus");
        // not lined up: the target is far to the side
        let s = snap(vec![virus, player(10, 1000.0, 800.0, 2000.0, 0xff0000)], 1000.0);
        assert!(!reflex(false, false, true).apply(&s, net()).eject);
        // flag off
        let s = snap(vec![virus, player(10, 1800.0, 0.0, 2000.0, 0xff0000)], 1000.0);
        assert!(!reflex(true, true, false).apply(&s, net()).eject);
    }

    #[test]
    fn virus_shot_is_vetoed_by_a_threat_next_to_us() {
        let virus = Cell::new(30, 1000.0, 0.0, 100.0, Kind::Virus);
        let s = snap(vec![virus, player(10, 1800.0, 0.0, 2000.0, 0xff0000), player(11, -500.0, 0.0, 9000.0, 0x0000ff)], 1000.0);
        assert!(!reflex(false, false, true).apply(&s, net()).eject);
    }

    #[test]
    fn split_clear_vetoes_a_crowded_split_and_params_override_genes() {
        // a near-equal (neither prey nor threat) bystander of 2700 within 1500 of us, the prey 400 at 800
        let s = snap(vec![player(10, 800.0, 0.0, 400.0, 1), player(12, -1200.0, 0.0, 2700.0, 2)], 3000.0);
        let mut r = Reflex::from_spec(&serde_json::json!({"reflex_split": true, "reflex_genes": "no/such.json"})).unwrap();
        assert!(r.apply(&s, net()).split);
        let mut r2 = Reflex::from_spec(&serde_json::json!({"reflex_split": true, "reflex_genes": "no/such.json", "reflex_split_clear": 1500.0})).unwrap();
        assert!(!r2.apply(&s, net()).split);
        // flee_margin override: a threat at the edge of the danger radius makes the default genes flee (no split) with 1.0
        let r3 = Reflex::from_spec(&serde_json::json!({"reflex_split": true, "reflex_genes": "no/such.json", "reflex_params": {"split_reach": 0.1}})).unwrap();
        let _ = r;
        let mut r3 = r3;
        assert!(!r3.apply(&snap(vec![player(10, 800.0, 0.0, 400.0, 1)], 3000.0), net()).split, "split_reach 0.1 must shrink the split distance");
    }

    #[test]
    fn guard_runs_from_a_bigger_player_while_split_and_is_silent_otherwise() {
        let mk = |others: Vec<Cell>, two: bool| {
            let mut s = snap(others, 1000.0);
            if two {
                s.mine.push(Cell::new(2, 60.0, 0.0, (400.0f64 * 100.0).sqrt(), Kind::Mine));
            }
            s
        };
        let mut r = Reflex::from_spec(&serde_json::json!({"reflex_split": true, "reflex_guard": true, "reflex_genes": "no/such.json"})).unwrap();
        // east of us a 2500 player at ~700: two pieces -> aim west, away from it
        let s = mk(vec![player(10, 700.0, 0.0, 2500.0, 1)], true);
        let o = r.apply(&s, Out { x: 2000.0, y: 0.0, split: false, eject: false });
        assert!(o.x < 0.0 && !o.split, "guard must turn the aim away, got x={}", o.x);
        // one cell only: the net keeps control
        let s = mk(vec![player(10, 700.0, 0.0, 2500.0, 1)], false);
        assert!(r.apply(&s, Out { x: 2000.0, y: 0.0, split: false, eject: false }).x > 0.0);
        // two pieces, nothing bigger around: the net keeps control
        let s = mk(vec![player(10, 700.0, 0.0, 300.0, 1)], true);
        assert!(r.apply(&s, Out { x: 2000.0, y: 0.0, split: false, eject: false }).x > 0.0);
    }

    /// the drill's popped giant (500 vs 5k): 12 small pieces and four big ones, 1100+ east of our 500-mass cell
    fn burst(extra: Vec<Cell>) -> Snapshot {
        let mut others = Vec::new();
        for i in 0..12 {
            let a = i as f64 * 0.52;
            others.push(player(20 + i, 1100.0 + 300.0 * a.cos(), 300.0 * a.sin(), 28.0, 9));
        }
        others.push(player(40, 1700.0, 500.0, 318.0, 9));
        others.push(player(41, 1800.0, -500.0, 637.0, 9));
        others.push(player(42, 2300.0, 0.0, 1275.0, 9));
        others.push(player(43, 3600.0, 0.0, 2521.0, 9));
        others.extend(extra);
        snap(others, 500.0)
    }
    fn hunter_reflex() -> Reflex {
        Reflex::from_spec(&serde_json::json!({"reflex_hunter": true, "reflex_hunter_genes": "no/such.json", "reflex_genes": "no/such.json"})).unwrap()
    }

    #[test]
    fn hunter_overrides_the_net_while_committing_to_a_burst() {
        let mut r = hunter_reflex();
        assert!(r.has_hunter());
        // the net goes west, away from the pieces: the hunter turns the aim east
        let o = r.apply(&burst(vec![]), net());
        assert!(o.x > 0.0 && !o.split, "hunter should head for the pieces in the east, got {o:?}");
        assert!(r.counters.0.iter().any(|(k, v)| k == "reflex_hunter" && *v >= 1));
    }

    #[test]
    fn hunter_leaves_the_net_alone_without_a_burst_or_when_off() {
        let mut r = hunter_reflex();
        // one lone prey: not a burst
        let s = snap(vec![player(10, 900.0, 0.0, 300.0, 1)], 3000.0);
        let o = r.apply(&s, net());
        assert_eq!((o.x, o.split), (net().x, false));
        // flag off: same burst, the net keeps its action
        let mut plain = Reflex::from_spec(&serde_json::json!({"reflex_guard": true, "reflex_genes": "no/such.json"})).unwrap();
        assert!(plain.apply(&burst(vec![]), net()).x < 0.0, "without reflex_hunter the net keeps the aim");
        assert!(!plain.has_hunter());
    }

    #[test]
    fn hunter_refuses_to_walk_into_a_giant_next_to_the_burst() {
        let mut r = hunter_reflex();
        // the 2521 core 900 away: inside its lunge, the hunter leaves along the clear side instead of walking at the pieces
        let mut s = burst(vec![]);
        let core = s.others.iter_mut().find(|c| c.id == 43).unwrap();
        core.x = 900.0;
        core.y = 0.0;
        let east = Out { x: 2000.0, y: 0.0, split: false, eject: false };
        let o = r.apply(&s, east);
        assert!(o.x < 100.0 && o.y.abs() > 300.0 && !o.split, "must flee the core sideways, got {o:?}");
    }

    #[test]
    fn hunter_state_is_cleared_by_reset_and_spec_default_is_off() {
        let mut r = hunter_reflex();
        let _ = r.apply(&burst(vec![]), net());
        r.reset();
        assert!(r.has_hunter());
        assert!(Reflex::from_spec(&serde_json::json!({"kind": "vision", "reflex_guard": true})).unwrap().hunter.is_none());
    }

    #[test]
    fn spec_flags_default_off() {
        assert!(Reflex::from_spec(&serde_json::json!({"kind": "vision"})).is_none());
        let r = Reflex::from_spec(&serde_json::json!({"kind": "vision", "reflex_split": true, "reflex_genes": "no/such/file.json"})).unwrap();
        assert!(r.split && !r.harvest && !r.virus);
    }
}
