//! Commander orders for a rule bot (docs/commander.md): an order overrides the mode / target the bot's
//! planner chose, but never the reflexes: when the bot's own plan is Flee (a bigger cell inside its danger
//! radius) the order is ignored for that step, and steering keeps its threat repulsion for every verb.

use crate::game::{Cell, Snapshot};
use crate::params::Params;
use crate::steer::{classify, danger_radius, half_of, split_threats, Mode, Plan, EAT_RATIO};

const HARVEST_R: f64 = 250.0;

#[derive(Clone, Debug, PartialEq)]
pub enum Verb {
    /// only the split flag of the order is active
    Keep,
    Hunt,
    Goto(f64, f64),
    Farm,
    Flee,
    Hide,
    /// shoot viruses at the target (bigger than the viruses) so they pop it
    VirusSplit,
    /// eat the pieces of the target's popped cells
    Harvest,
}

#[derive(Clone, Debug)]
pub struct Order {
    pub verb: Verb,
    /// visible cell ids of the target player (Hunt, VirusSplit, Harvest)
    pub ids: Vec<u32>,
    /// Some(false) = no split attacks; Some(true) = split attacks allowed
    pub split: Option<bool>,
}

/// What an order changed in one decision.
pub struct Applied {
    pub plan: Plan,
    pub goto: Option<(f64, f64)>,
    /// the order's verb was acted on (false: vetoed by a reflex, or its target is not visible / not edible)
    pub honoured: bool,
}

pub fn apply_plan(base: Plan, s: &Snapshot, p: &Params, o: &Order) -> Applied {
    let mut a = Applied { plan: base, goto: None, honoured: false };
    if base.mode == Mode::Flee {
        return a;
    }
    let me = *s.main().unwrap();
    let cl = classify(s, Some(p));
    let gap = |c: &Cell| me.dist(c) - c.size - me.size;
    match &o.verb {
        Verb::Keep => {}
        Verb::Farm => {
            a.plan = Plan::new(Mode::Farm);
            a.honoured = true;
        }
        Verb::Flee => {
            a.plan = Plan::new(Mode::Flee);
            a.honoured = true;
        }
        Verb::Hide => {
            let cover = cl.viruses.iter().any(|v| v.mass() > me.mass() * EAT_RATIO);
            a.plan = Plan::new(if cover { Mode::Hide } else if !cl.threats.is_empty() { Mode::Flee } else { Mode::Farm });
            a.honoured = cover;
        }
        Verb::Goto(x, y) => {
            a.plan = Plan::new(Mode::Farm);
            a.goto = Some((*x, *y));
            a.honoured = true;
        }
        Verb::VirusSplit => {
            // the virus shooting itself is a tactic (Tactics::act); between shots the bot keeps farming
            a.plan = Plan::new(Mode::Farm);
            a.honoured = s.others.iter().any(|c| o.ids.contains(&c.id));
        }
        Verb::Hunt | Verb::Harvest => {
            let cands: Vec<&Cell> = cl.prey.iter().filter(|c| o.ids.contains(&c.id)).collect();
            let mut best: Option<(&Cell, f64)> = None;
            for c in &cands {
                let k = if o.verb == Verb::Harvest {
                    cands.iter().filter(|q| q.dist(c) < HARVEST_R).map(|q| q.mass()).sum::<f64>() / (gap(c).max(0.0) + 100.0)
                } else {
                    c.mass() / (gap(c).max(0.0) + 100.0)
                };
                if best.map_or(true, |(_, bk)| k > bk) {
                    best = Some((c, k));
                }
            }
            if let Some((t, _)) = best {
                let side = if p.half_split_safety > 0.5 { half_of(&me) } else { me };
                let safe = !split_threats(s, &me, p).iter().any(|th| gap(th) < danger_radius(th, &side, p) * p.split_safety);
                let can = me.mass() / 2.0 > t.mass() * EAT_RATIO;
                a.plan = Plan { mode: Mode::Hunt, target_id: Some(t.id), split_ok: can && (safe || p.split_ev > 0.0) };
                a.honoured = true;
            }
        }
    }
    if o.split == Some(false) {
        a.plan.split_ok = false;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Border, Kind};
    use crate::policy::heuristic_plan;

    fn snap(others: Vec<Cell>) -> Snapshot {
        let me = Cell::new(1, 0.0, 0.0, (3000.0f64 * 100.0).sqrt(), Kind::Mine);
        Snapshot { mine: vec![me], others, border: Border { left: -5000.0, right: 5000.0, top: -5000.0, bottom: 5000.0 }, dead: false }
    }
    fn player(id: u32, x: f64, y: f64, mass: f64) -> Cell {
        Cell::new(id, x, y, (mass * 100.0).sqrt(), Kind::Player)
    }
    fn order(verb: Verb, ids: &[u32]) -> Order {
        Order { verb, ids: ids.to_vec(), split: None }
    }

    #[test]
    fn hunt_order_changes_target() {
        // two prey: the bot picks the closer big one; the order picks the far small one
        let s = snap(vec![player(10, 500.0, 0.0, 800.0), player(11, -700.0, 0.0, 300.0)]);
        let p = Params::default();
        let base = heuristic_plan(&s, &p);
        assert_eq!(base.target_id, Some(10));
        let a = apply_plan(base, &s, &p, &order(Verb::Hunt, &[11]));
        assert!(a.honoured);
        assert_eq!(a.plan.mode, Mode::Hunt);
        assert_eq!(a.plan.target_id, Some(11));
    }

    #[test]
    fn hunt_never_overrides_flee_reflex() {
        // a 9000 cell next to us: base plan is Flee, the order to hunt the small cell is ignored
        let s = snap(vec![player(10, 300.0, 0.0, 9000.0), player(11, -700.0, 0.0, 300.0)]);
        let p = Params::default();
        let base = heuristic_plan(&s, &p);
        assert_eq!(base.mode, Mode::Flee);
        let a = apply_plan(base, &s, &p, &order(Verb::Hunt, &[11]));
        assert!(!a.honoured);
        assert_eq!(a.plan.mode, Mode::Flee);
    }

    #[test]
    fn hunt_of_a_bigger_cell_is_refused() {
        let s = snap(vec![player(10, 1500.0, 0.0, 9000.0)]);
        let p = Params::default();
        let base = heuristic_plan(&s, &p);
        let a = apply_plan(base, &s, &p, &order(Verb::Hunt, &[10]));
        assert!(!a.honoured);
        assert_ne!(a.plan.target_id, Some(10));
    }

    #[test]
    fn split_off_vetoes_split() {
        let s = snap(vec![player(10, 400.0, 0.0, 300.0)]);
        let p = Params::default();
        let o = Order { verb: Verb::Hunt, ids: vec![10], split: Some(false) };
        assert!(!apply_plan(heuristic_plan(&s, &p), &s, &p, &o).plan.split_ok);
        let o = Order { verb: Verb::Hunt, ids: vec![10], split: None };
        assert!(apply_plan(heuristic_plan(&s, &p), &s, &p, &o).plan.split_ok);
    }

    #[test]
    fn harvest_prefers_the_cloud() {
        let mut others = vec![player(10, 900.0, 0.0, 700.0)];
        for i in 0..5 {
            others.push(player(20 + i, -900.0 - 40.0 * i as f64, 30.0 * i as f64, 300.0));
        }
        let s = snap(others);
        let p = Params::default();
        let ids: Vec<u32> = vec![10, 20, 21, 22, 23, 24];
        let a = apply_plan(heuristic_plan(&s, &p), &s, &p, &order(Verb::Harvest, &ids));
        assert!((20..25).contains(&a.plan.target_id.unwrap()));
    }
}
