//! Lineup specs of the command lines (`route:4 champ=runs/x.json:4`), shared by `arena_client` and `ogar`.

use serde_json::{json, Value};

/// "route:4 champ=runs/x.json:4" -> [(agent name, policy spec)] (training.eval.match.parse_lineup; rule bots only).
pub fn parse_lineup(specs: &[String]) -> Result<Vec<(String, Value)>, String> {
    let mut lineup = Vec::new();
    for spec in specs {
        let (head, tail) = match spec.rsplit_once(':') {
            Some((h, t)) => (h, t),
            None => ("", spec.as_str()),
        };
        let (key, count) = if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) { (head, tail) } else { (spec.as_str(), "1") };
        let (key, policy) = if let Some((k, path)) = key.split_once('=') {
            if path.ends_with(".pt") || path.contains("jev:") || path.contains("hybrid") || path.contains("guard") {
                return Err(format!("{spec}: NN / Jev policies are not supported (rule bots only)"));
            }
            let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            (k, serde_json::from_str::<Value>(&text).map_err(|e| format!("{path}: {e}"))?)
        } else if key == "route" || key == "field" {
            (key, json!({ "farm": key }))
        } else {
            return Err(format!("unknown policy {key:?}; known: route, field or name=file.json"));
        };
        let n: usize = if count.is_empty() { 1 } else { count.parse().map_err(|_| format!("bad count in {spec}"))? };
        for i in 0..n {
            lineup.push((format!("{key}-{i}"), policy.clone()));
        }
    }
    Ok(lineup)
}
