//! Effective MultiOgar settings: vendor config.ini (mts(x) = sqrt(100 x) + .5) plus arena.js overrides.

#[derive(Clone, Debug)]
pub struct Config {
    pub map: f64,
    pub bots: usize,
    pub food_min_amount: i64,
    pub food_spawn_amount: i64,
    pub food_size: f64,
    pub spawn_interval: u32,
    pub virus_min_size: f64,
    pub virus_max_size: f64,
    pub virus_min_amount: i64,
    pub virus_max_amount: i64,
    pub virus_eject_speed: f64,
    pub virus_split_div: f64,
    pub virus_eat_mult: f64,
    pub virus_max_cells: i64,
    pub eject_size: f64,
    pub eject_size_loss: f64,
    pub eject_cooldown: u32,
    pub eject_spawn_chance: f64,
    pub eject_speed: f64,
    pub player_min_decay: f64,
    pub player_max_size: f64,
    pub player_min_split: f64,
    pub player_min_eject: f64,
    pub player_start_size: f64,
    pub bot_start_size: f64,
    pub minion_start_size: f64,
    pub player_max_cells: i64,
    pub player_speed: f64,
    pub player_merge_time: f64,
    pub player_decay_rate: f64,
    pub player_split_speed: f64,
    pub player_split_div: f64,
    pub player_eat_mult: f64,
    pub split_restore_ticks: u32,
    pub view_base_x: f64,
    pub view_base_y: f64,
    pub min_scale: f64,
    pub spec_scale: f64,
    pub free_roam_speed: f64,
    pub max_lb: usize,
    pub lb_update: u32,
    pub player_max_nick: usize,
    pub disconnect_ticks: u32,
}

fn mts(mass: f64) -> f64 {
    (mass * 100.0).sqrt() + 0.5
}

/// JS Math.round
fn round(x: f64) -> f64 {
    (x + 0.5).floor()
}

impl Config {
    /// `food_mult`: showcase food density multiplier (arena.js --food-mult).
    pub fn new(map: f64, bots: usize, food_mult: f64) -> Config {
        let area = (map / 14142.0).powi(2);
        let base_food = round(1500.0 * area);
        let virus_min = (round(50.0 * area)).max(4.0);
        let virus_max = round(100.0 * area).max(virus_min + 2.0);
        Config {
            map,
            bots,
            food_min_amount: round(base_food * food_mult) as i64,
            food_spawn_amount: 30,
            food_size: 10.0,
            spawn_interval: 20,
            virus_min_size: mts(100.0),
            virus_max_size: mts(200.0),
            virus_min_amount: virus_min as i64,
            virus_max_amount: virus_max as i64,
            virus_eject_speed: 780.0,
            virus_split_div: 36.0,
            virus_eat_mult: 1.1576,
            virus_max_cells: 16,
            eject_size: mts(13.0),
            eject_size_loss: mts(17.0),
            eject_cooldown: 2,
            eject_spawn_chance: 0.5,
            eject_speed: 780.0,
            player_min_decay: mts(10.0).max(32.0),
            player_max_size: mts(22500.0),
            player_min_split: mts(35.0),
            player_min_eject: mts(35.0),
            player_start_size: mts(10.0),
            bot_start_size: mts(10.0),
            minion_start_size: mts(10.0),
            player_max_cells: 16,
            player_speed: 30.0,
            player_merge_time: 30.0,
            player_decay_rate: 0.002,
            player_split_speed: 780.0,
            player_split_div: 2.0,
            player_eat_mult: 1.15,
            split_restore_ticks: 13,
            view_base_x: 1920.0,
            view_base_y: 1080.0,
            min_scale: 0.15,
            spec_scale: 0.4,
            free_roam_speed: 25.0,
            max_lb: 10,
            lb_update: 25,
            player_max_nick: 30,
            disconnect_ticks: 1500,
        }
    }

    /// Showcase live food-density change (0xCC): u16 = multiplier * 100.
    pub fn set_food_mult(&mut self, base_food: f64, pct: u16) {
        self.food_min_amount = round(base_food * pct as f64 / 100.0) as i64;
    }

    pub fn base_food(&self) -> f64 {
        round(1500.0 * (self.map / 14142.0).powi(2))
    }
}
