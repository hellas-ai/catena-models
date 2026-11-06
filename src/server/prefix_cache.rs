//! Two immutable prompt checkpoints; all GPU access stays on the inference thread.
use std::rc::Rc;

pub const PREFILL_BLOCK: usize = 1024;

pub struct Entry<T> {
    pub tokens: Vec<u64>,
    pub capacity: usize,
    pub state: T,
    pub next: u64,
}

pub struct PrefixCache<T> {
    pub stable: Option<Rc<Entry<T>>>,
    pub recent: Option<Rc<Entry<T>>>,
}

impl<T> Default for PrefixCache<T> {
    fn default() -> Self {
        Self {
            stable: None,
            recent: None,
        }
    }
}

impl<T> PrefixCache<T> {
    pub fn clear(&mut self) {
        self.stable = None;
        self.recent = None;
    }

    pub fn matching(&self, tokens: &[u64], capacity: usize, limit: usize) -> Option<&Rc<Entry<T>>> {
        [&self.stable, &self.recent]
            .into_iter()
            .flatten()
            .filter(|entry| {
                entry.capacity == capacity
                    && entry.tokens.len() <= limit
                    && tokens.starts_with(&entry.tokens)
            })
            .max_by_key(|entry| entry.tokens.len())
    }

    pub fn exact(&self, tokens: &[u64], capacity: usize) -> Option<&Rc<Entry<T>>> {
        self.matching(tokens, capacity, tokens.len())
            .filter(|entry| entry.tokens.len() == tokens.len())
    }
}

pub fn block_boundary(position: usize) -> usize {
    position / PREFILL_BLOCK * PREFILL_BLOCK
}

pub fn common_prefix(left: &[u64], right: &[u64]) -> usize {
    left.iter().zip(right).take_while(|(a, b)| a == b).count()
}

pub fn capacity_bucket(required: usize, context: usize) -> usize {
    required
        .checked_next_power_of_two()
        .unwrap_or(context)
        .min(context)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(tokens: &[u64], capacity: usize) -> Rc<Entry<()>> {
        Rc::new(Entry {
            tokens: tokens.into(),
            capacity,
            state: (),
            next: 42,
        })
    }

    #[test]
    fn longest_exact_token_prefix_with_same_capacity_only() {
        let mut cache = PrefixCache::default();
        cache.stable = Some(entry(&[1, 2], 8));
        cache.recent = Some(entry(&[1, 2, 3, 4], 8));
        assert_eq!(
            cache.matching(&[1, 2, 3, 4, 5], 8, 5).unwrap().tokens.len(),
            4
        );
        // A rewritten reasoning span invalidates the recent slot, not the stable one.
        assert_eq!(
            cache.matching(&[1, 2, 9, 4, 5], 8, 5).unwrap().tokens.len(),
            2
        );
        assert!(cache.matching(&[1, 7, 3, 4], 8, 4).is_none());
        assert!(cache.matching(&[1, 2, 3, 4], 16, 4).is_none());
        assert!(cache.matching(&[1], 8, 1).is_none());
        assert_eq!(cache.exact(&[1, 2, 3, 4], 8).unwrap().next, 42);
        assert!(cache.exact(&[1, 2, 3], 8).is_none());
        assert_eq!(cache.matching(&[1, 2, 3, 4], 8, 2).unwrap().tokens.len(), 2);
        cache.clear();
        assert!(cache.matching(&[1, 2], 8, 2).is_none());
    }

    #[test]
    fn absolute_blocks_and_capacity_buckets() {
        assert_eq!(block_boundary(1023), 0);
        assert_eq!(block_boundary(1024), 1024);
        assert_eq!(block_boundary(2050), 2048);
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 4]), 2);
        assert_eq!(capacity_bucket(1024, 5000), 1024);
        assert_eq!(capacity_bucket(1025, 5000), 2048);
        assert_eq!(capacity_bucket(4097, 5000), 5000);
        assert_eq!(capacity_bucket(usize::MAX, usize::MAX), usize::MAX);
    }
}
