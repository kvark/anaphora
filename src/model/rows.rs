//! Row slicing and row concatenation on Meganeura's channel-shaped ops.
//!
//! Meganeura's [`Graph::split_a`], [`Graph::split_b`], and [`Graph::concat`]
//! address an `[N, C, spatial]` layout — they were built for convolutional
//! channel splits. A row-major `[rows, d]` matrix is the same buffer as
//! `[1, rows, d]`, so setting `N = 1`, `C = rows`, `spatial = d` turns a
//! channel split into a row slice with no data movement and no new operator.
//!
//! All three carry autodiff rules, so a slice is not a gradient dead end.
//! That is worth stating explicitly: `SplitA`/`SplitB` had no backward at all
//! until 2026-05-23, and the failure mode was silent — parameters upstream of
//! a split simply never moved.
//!
//! # Why every edge is flattened explicitly
//!
//! These three operators produce **rank-1** values: `concat` types its output
//! `[batch * (ca + cb) * spatial]`, and the splits do the same. Their
//! gradient rules are symmetric, so the gradient they hand back to an input
//! is also rank-1.
//!
//! That matters because of where the gradient lands. Feed a `[rows, d]`
//! matmul result straight into `split_b` and the gradient accumulated for
//! that matmul is rank-1 — and `Op::MatMul`'s backward calls
//! [`Graph::matmul_bt`], which asserts its operands are rank 2. The build
//! fails inside autodiff with a bare `assert_eq!(1, 2)` and a backtrace that
//! points at the differentiator rather than at the slice that caused it.
//!
//! So each edge into a split or a concat is an explicit `reshape` to rank 1
//! first. `reshape` is `Op::Identity` — no dispatch, no copy — and its
//! backward re-types the incoming gradient to whatever the *input* was. The
//! rank-1 gradient therefore lands on the rank-1 reshape node, and the
//! reshape hands a correctly-shaped rank-2 gradient back to the matmul.

use meganeura::{Graph, NodeId};

/// Rows `start .. start + count` of a `[rows, d]` value.
///
/// Costs at most two operators plus zero-dispatch reshapes.
#[track_caller]
pub fn slice_rows(
    g: &mut Graph,
    x: NodeId,
    rows: usize,
    d: usize,
    start: usize,
    count: usize,
) -> NodeId {
    assert!(
        start + count <= rows,
        "slice_rows: {start}..{} out of range for {rows} rows",
        start + count
    );
    assert!(count > 0, "slice_rows: empty slice");
    if start == 0 && count == rows {
        return x;
    }

    // Rank 1 from here down; see the module header.
    let mut cur = g.reshape(x, &[rows * d]);
    let mut cur_rows = rows;

    if start > 0 {
        let tail = cur_rows - start;
        cur = g.split_b(cur, 1, start as u32, tail as u32, d as u32);
        cur_rows = tail;
    }
    if count < cur_rows {
        let rest = cur_rows - count;
        cur = g.split_a(cur, 1, count as u32, rest as u32, d as u32);
    }
    g.reshape(cur, &[count, d])
}

/// Stack `[rows_i, d]` values into one `[sum(rows_i), d]` value.
///
/// Folds left, so the operator count is `parts - 1`.
#[track_caller]
pub fn concat_rows(g: &mut Graph, parts: &[(NodeId, usize)], d: usize) -> NodeId {
    assert!(!parts.is_empty(), "concat_rows: nothing to concatenate");
    let (first, first_rows) = parts[0];
    if parts.len() == 1 {
        return g.reshape(first, &[first_rows, d]);
    }

    let mut acc = g.reshape(first, &[first_rows * d]);
    let mut acc_rows = first_rows;
    for &(node, rows) in &parts[1..] {
        let flat = g.reshape(node, &[rows * d]);
        acc = g.concat(acc, flat, 1, acc_rows as u32, rows as u32, d as u32);
        acc_rows += rows;
    }
    g.reshape(acc, &[acc_rows, d])
}

/// Append a column of per-row scalars: `[rows, d] ++ [rows, 1] -> [rows, d+1]`.
///
/// Feature-wise rather than row-wise, so the `[N, C, spatial]` mapping is the
/// other one: `N = rows`, `C = d` and `1`, `spatial = 1`.
#[track_caller]
pub fn append_column(g: &mut Graph, x: NodeId, col: NodeId, rows: usize, d: usize) -> NodeId {
    let x_flat = g.reshape(x, &[rows * d]);
    let col_flat = g.reshape(col, &[rows]);
    let joined = g.concat(x_flat, col_flat, rows as u32, d as u32, 1, 1);
    g.reshape(joined, &[rows, d + 1])
}
