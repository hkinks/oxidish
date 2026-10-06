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
}

impl Params {
    /// These genes with the big-cell caution applied (caution ramp).
    pub fn cautious(&self) -> Params {
        Params { flee_margin: self.flee_margin_big, split_safety: self.split_safety_big,
                 half_split_safety: 1.0, ..self.clone() }
    }
}
