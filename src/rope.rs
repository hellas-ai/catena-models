use crate::spec::{HEAD_DIM, ROPE_HALF_DIM, ROPE_THETA};

/// Packed `[cos(sequence, head_dim / 2), sin(sequence, head_dim / 2)]` table.
pub fn table(sequence_len: usize) -> Vec<f32> {
    table_range(0, sequence_len)
}

/// Packed RoPE rows for `start..start + sequence_len`.
pub fn table_range(start: usize, sequence_len: usize) -> Vec<f32> {
    let plane_len = sequence_len * ROPE_HALF_DIM;
    let mut values = vec![0.0; plane_len * 2];
    for local_position in 0..sequence_len {
        let position = start + local_position;
        for pair in 0..ROPE_HALF_DIM {
            let exponent = (2 * pair) as f32 / HEAD_DIM as f32;
            let frequency = ROPE_THETA.powf(-exponent);
            let angle = position as f32 * frequency;
            let offset = local_position * ROPE_HALF_DIM + pair;
            values[offset] = angle.cos();
            values[plane_len + offset] = angle.sin();
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_zero_is_identity() {
        let values = table(1);
        assert_eq!(&values[..ROPE_HALF_DIM], vec![1.0; ROPE_HALF_DIM]);
        assert_eq!(&values[ROPE_HALF_DIM..], vec![0.0; ROPE_HALF_DIM]);
    }

    #[test]
    fn table_has_two_planes() {
        assert_eq!(table(7).len(), 2 * 7 * ROPE_HALF_DIM);
    }

    #[test]
    fn ranged_rows_match_the_full_table() {
        let full = table(7);
        let range = table_range(3, 2);
        for local in 0..2 {
            for pair in 0..ROPE_HALF_DIM {
                assert_eq!(
                    range[local * ROPE_HALF_DIM + pair],
                    full[(local + 3) * ROPE_HALF_DIM + pair]
                );
                assert_eq!(
                    range[2 * ROPE_HALF_DIM + local * ROPE_HALF_DIM + pair],
                    full[7 * ROPE_HALF_DIM + (local + 3) * ROPE_HALF_DIM + pair]
                );
            }
        }
    }
}
