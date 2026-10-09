//! Behaviour genes (port of bot/params.py). Same names and defaults; unknown JSON keys are ignored.

use serde_json::Value;

macro_rules! params {
    ($($n:ident = $d:expr,)*) => {
        #[derive(Clone, Debug)]
        pub struct Params { $(pub $n: f64,)* }

        impl Default for Params {
            fn default() -> Params { Params { $($n: $d,)* } }
        }

        impl Params {
            /// Build from a JSON object of gene values (missing genes keep their default).
            pub fn from_json(v: &Value) -> Params {
                let mut p = Params::default();
                $( if let Some(x) = v.get(stringify!($n)).and_then(|x| x.as_f64()) { p.$n = x; } )*
                p
            }
        }
    };
}

params! {
    danger_margin = 250.0,
    split_range = 650.0,
    threat_force = 30.0,
    flee_margin = 1.0,
    farm_prey = 0.5,
    farm_threat = 1.5,
    hunt_food = 0.3,
    hunt_threat = 1.5,
    flee_threat = 4.0,
    target_bonus = 6.0,
    food_pull = 3.0,
    hunt_range = 900.0,
    min_prey_frac = 0.1,
    split_safety = 2.0,
    split_reach = 0.8,
    split_max_cells = 1.0,
    virus_avoid = 150.0,
    border_margin = 200.0,
    switch_margin = 1.25,
    shoot_enable = 0.45,
    shoot_range = 600.0,
    shoot_align = 0.9,
    shoot_reach = 900.0,
    shoot_min_mass = 300.0,
    shoot_target_mult = 2.0,
    hide_enable = 0.45,
    hide_range = 500.0,
    lead = 0.0,
    virus_detour = 1.0,
    double_split_enable = 0.45,
    double_split_reach = 1.8,
    selffeed_enable = 0.45,
    selffeed_min_mass = 300.0,
    selffeed_frac = 0.7,
    big_hunt = 0.0,
    big_mass = 1500.0,
    half_split_safety = 0.0,
    piece_threats = 0.0,
    caution_mass = 0.0,
    flee_margin_big = 1.0,
    split_safety_big = 2.0,
    flee_plan = 0.0,
    flee_pieces = 0.0,
    aim_all_cells = 1.0,
    contender = 0.0,
    contender_gap = 200.0,
    wall_danger = 0.0,
    wall_wary = 0.0,
    wall_wary_ratio = 0.8,
    piece_detour = 0.0,
    split_ev = 0.0,
    escape_split = 0.0,
    flee_smooth = 0.0,
    shoot_threats = 0.0,
    harvest = 0.0,
    harvest_range = 1.0,
    giant_mass = 0.0,
    giant_hunt_range = 900.0,
    giant_min_prey_frac = 0.1,
    giant_split_safety = 2.0,
    giant_split_reach = 0.8,
    giant_farm_threat = 1.5,
    giant_flee_margin = 1.0,
    giant_danger_margin = 250.0,
    giant_target_bonus = 6.0,
    giant_food_pull = 3.0,
    shoot_stage = 0.0,
    shoot_walk = 1200.0,
    shoot_standoff = 90.0,
    shoot_safety = 0.7,
    shoot_max_cells = 1.0,
    evade = 0.0,
    evade_trigger = 0.0,
    evade_buffer = 0.0,
    evade_wall = 0.0,
    evade_cap = 0.0,
    evade_virus = 0.0,
    evade_pieces = 0.0,
}

impl Params {
    /// The genes for the giant stage (Python: Params.giant): cautious genes if the ramp is on, plus giant_* overrides.
    pub fn giant(&self) -> Params {
        let base = if self.caution_mass > 0.0 { self.cautious() } else { self.clone() };
        Params { hunt_range: self.giant_hunt_range, min_prey_frac: self.giant_min_prey_frac, split_safety: self.giant_split_safety, split_reach: self.giant_split_reach, farm_threat: self.giant_farm_threat, flee_margin: self.giant_flee_margin, danger_margin: self.giant_danger_margin, target_bonus: self.giant_target_bonus, food_pull: self.giant_food_pull, ..base }
    }

    /// These genes with the big-cell caution applied (caution ramp).
    pub fn cautious(&self) -> Params {
        Params { flee_margin: self.flee_margin_big, split_safety: self.split_safety_big,
                 half_split_safety: 1.0, ..self.clone() }
    }
}
