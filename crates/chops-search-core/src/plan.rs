//! Turn missing row ids into coalesced byte ranges over `model.rows.i8`.
//!
//! Row i lives at byte offset i * dim, length dim. Nearby rows are merged
//! into one range when the gap is small — fetching a few dead rows is
//! cheaper than another HTTP round trip, and ingesting them warms the
//! cache for free.

/// Half-open byte range [start, end) into the rows file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    pub start: u32,
    pub end: u32,
}

/// `rows` must be sorted ascending and deduplicated (RowStore::missing
/// guarantees this). `max_gap_rows` is how many unneeded rows we're willing
/// to fetch to avoid splitting a range.
pub fn coalesce(rows: &[u32], dim: u32, max_gap_rows: u32) -> Vec<ByteRange> {
    let mut out = Vec::new();
    let mut iter = rows.iter().copied();
    let Some(first) = iter.next() else {
        return out;
    };
    let mut run_start = first;
    let mut run_end = first; // inclusive, in rows
    for r in iter {
        if r - run_end <= max_gap_rows + 1 {
            run_end = r;
        } else {
            out.push(ByteRange {
                start: run_start * dim,
                end: (run_end + 1) * dim,
            });
            run_start = r;
            run_end = r;
        }
    }
    out.push(ByteRange {
        start: run_start * dim,
        end: (run_end + 1) * dim,
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        assert!(coalesce(&[], 128, 8).is_empty());
    }

    #[test]
    fn single_row() {
        let r = coalesce(&[5], 128, 8);
        assert_eq!(r, vec![ByteRange { start: 640, end: 768 }]);
    }

    #[test]
    fn adjacent_rows_merge() {
        let r = coalesce(&[5, 6, 7], 128, 0);
        assert_eq!(r, vec![ByteRange { start: 640, end: 1024 }]);
    }

    #[test]
    fn small_gap_merges_large_gap_splits() {
        // gap of 8 rows merges (max_gap 8), gap of 100 splits
        let r = coalesce(&[0, 9, 200], 128, 8);
        assert_eq!(
            r,
            vec![
                ByteRange { start: 0, end: 10 * 128 },
                ByteRange { start: 200 * 128, end: 201 * 128 },
            ]
        );
    }
}
