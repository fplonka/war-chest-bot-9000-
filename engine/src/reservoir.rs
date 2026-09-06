use crate::pbs::{Belief, Ctx};
use crate::rng::Rng;
use crate::state::State;

const CAPACITY: usize = 8_192;
const WARM: usize = 1_024;
const PRIORITY_SAMPLE: usize = 8;

#[derive(Clone)]
pub struct Entry {
    pub state: State,
    pub ctx: Ctx,
    pub belief: [Belief; 2],
    pub value: [Vec<f32>; 2],
    pub generation: u64,
}

pub struct Reservoir {
    entries: Vec<Entry>,
    admissions: u64,
    generation: u64,
    rng: Rng,
}

pub enum Task {
    Revisit { index: usize, entry: Box<Entry> },
    Probe { entries: Vec<(usize, Entry)>, scores: Vec<f32> },
}

impl Reservoir {
    pub fn new(seed: u64) -> Reservoir {
        Reservoir {
            entries: Vec::with_capacity(CAPACITY),
            admissions: 0,
            generation: 0,
            rng: Rng::new(seed),
        }
    }

    pub fn admit(&mut self, state: State, ctx: Ctx, belief: [Belief; 2], value: [Vec<f32>; 2]) {
        self.admissions += 1;
        let index = if self.entries.len() < CAPACITY {
            self.entries.len()
        } else {
            let pick = self.rng.below(self.admissions as usize);
            if pick >= CAPACITY {
                return;
            }
            pick
        };
        self.generation += 1;
        let entry = Entry {
            state,
            ctx,
            belief,
            value,
            generation: self.generation,
        };
        if index == self.entries.len() {
            self.entries.push(entry);
        } else {
            self.entries[index] = entry;
        }
    }

    pub fn refresh(&mut self, index: usize, generation: u64, value: [Vec<f32>; 2]) -> bool {
        let Some(entry) = self.entries.get_mut(index) else {
            return false;
        };
        if entry.generation != generation {
            return false;
        }
        self.generation += 1;
        entry.generation = self.generation;
        entry.value = value;
        true
    }

    pub fn pick(&mut self) -> Option<Task> {
        if self.entries.len() < WARM || self.rng.below(2) == 0 {
            return None;
        }
        if self.rng.below(2) == 0 {
            let index = self.rng.below(self.entries.len());
            return Some(Task::Revisit { index, entry: Box::new(self.entries[index].clone()) });
        }
        let mut indices = Vec::with_capacity(PRIORITY_SAMPLE);
        while indices.len() < PRIORITY_SAMPLE {
            let index = self.rng.below(self.entries.len());
            if !indices.contains(&index) {
                indices.push(index);
            }
        }
        let entries = indices.into_iter().map(|i| (i, self.entries[i].clone())).collect();
        Some(Task::Probe { entries, scores: Vec::new() })
    }

    #[cfg(test)]
    fn entries(&self) -> &[Entry] {
        &self.entries
    }
}

pub fn priority_index(entries: &[(usize, Entry)], scores: &[f32]) -> usize {
    assert_eq!(entries.len(), scores.len());
    let mut best = 0;
    for i in 1..entries.len() {
        if scores[i] > scores[best] || (scores[i] == scores[best] && entries[i].0 < entries[best].0)
        {
            best = i;
        }
    }
    best
}

pub fn disagreement(current: &[Vec<f32>; 2], stored: &[Vec<f32>; 2]) -> f32 {
    current
        .iter()
        .zip(stored)
        .map(|(a, b)| {
            assert_eq!(a.len(), b.len(), "a probe changed belief support");
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len().max(1) as f32
        })
        .sum::<f32>()
        / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selfplay::collect_roots;

    fn item(seed: u64, value: f32) -> (State, Ctx, [Belief; 2], [Vec<f32>; 2]) {
        let (state, belief) = collect_roots(1, seed).pop().unwrap();
        let ctx = Ctx::new(&state);
        let values = std::array::from_fn(|p| vec![value; belief[p].len()]);
        (state, ctx, belief, values)
    }

    #[test]
    fn a_small_reservoir_only_selects_fresh_queries() {
        let mut reservoir = Reservoir::new(11);
        let (state, ctx, belief, value) = item(1, 0.0);
        reservoir.admit(state, ctx, belief, value);
        for _ in 0..100 {
            assert!(reservoir.pick().is_none());
        }
    }

    #[test]
    fn stale_refresh_cannot_replace_a_new_occurrence() {
        let mut reservoir = Reservoir::new(3);
        let (state, ctx, belief, value) = item(1, 0.0);
        reservoir.admit(state, ctx, belief, value);
        let generation = reservoir.entries()[0].generation;
        assert!(reservoir.refresh(0, generation, [vec![0.5], vec![-0.5]]));
        assert_eq!(reservoir.admissions, 1);
        reservoir.generation += 1;
        reservoir.entries[0].generation = reservoir.generation;
        assert!(!reservoir.refresh(0, generation + 1, [vec![1.0], vec![1.0]]));
    }

    #[test]
    fn priority_ties_choose_the_lowest_reservoir_index() {
        let (state, ctx, belief, value) = item(1, 0.0);
        let entry = Entry {
            state,
            ctx,
            belief,
            value,
            generation: 1,
        };
        let entries = vec![(9, entry.clone()), (2, entry)];
        assert_eq!(priority_index(&entries, &[0.5, 0.5]), 1);
    }

    #[test]
    fn disagreement_averages_configurations_then_players() {
        let current = [vec![1.0, -1.0], vec![0.5]];
        let stored = [vec![0.0, 0.0], vec![-0.5]];
        assert_eq!(disagreement(&current, &stored), 1.0);
    }
}
