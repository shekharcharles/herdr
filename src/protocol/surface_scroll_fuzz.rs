//! Randomised round-trip fuzz for the pane-scroll codec, donated from the
//! throwaway Python model that verified 55k cases while triaging the sticky
//! row-splice report (upstream twin herdrdev/herdr#5750).
//!
//! The invariant under test is codec convergence, independent of upstream
//! content drift: when a scroll message is emitted for a baseline both peers
//! hold, the receiver's grid after [`surface_scroll::apply`] must equal the
//! grid the server commits by applying the original patch rows, and the
//! expanded rows the fast presentation path consumes must be pairwise
//! disjoint. Each case negotiates a random wire version: v2 cases ride the
//! baseline-hash guard, where the fuzz checks the hashes the sender puts on
//! the wire equal [`surface_scroll::rect_baseline_hash`] over the agreed
//! baseline's scroll regions, and `apply` succeeding on non-empty hashes
//! proves the receiver's pre-rotation recomputation agrees with them.
//!
//! Generators reproduce the shapes that made the bug visible: log-style
//! column rotations, fresh bottom rows, full-width border repaints crossing a
//! pane, and scrollbar-column edits beside it. Two-pane cases put disjoint
//! scrolls, and one border row straddling both regions, through one message.
//!
//! Cases are seeded deterministically (in-file xorshift64*, no rand
//! dependency), so a failure reproduces from its printed seed. Set
//! `HERDR_FUZZ_CASES` to scale the case count for deeper local runs.

#![cfg(test)]

use super::surface_scroll::{
    MESSAGE_KIND, MESSAGE_KIND_V2, apply, decode, decode_v2, message, message_v2,
    rect_baseline_hash,
};
use super::{
    CellData, FrameData, PaneSurfaceFrame, PaneSurfacePane, PaneSurfacePatch, PaneSurfacePatchRow,
    ServerMessage, SurfaceRect,
};

const DEFAULT_CASES: u64 = 10_000;

/// xorshift64* stream, one per case, seeded through a splitmix64 finalizer so
/// neighbouring seeds (and seed 0) start on uncorrelated non-zero states.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        state = (state ^ (state >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state = (state ^ (state >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self((state ^ (state >> 31)) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }

    fn between(&mut self, low: u64, high: u64) -> u64 {
        low + self.below(high - low + 1)
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

fn cell(symbol: &str, fg: u32, bg: u32) -> CellData {
    CellData {
        symbol: symbol.to_owned(),
        fg,
        bg,
        modifier: 0,
        skip: false,
        hyperlink: None,
    }
}

/// Painted cell like scrolled content leaves them: varying symbol and style.
fn random_content(rng: &mut Rng, symbols: u64) -> CellData {
    cell(
        &rng.below(symbols).to_string(),
        rng.below(2) as u32,
        rng.below(3) as u32,
    )
}

struct Case {
    width: u16,
    height: u16,
    /// One or two disjoint pane regions; the second sits right of the first.
    panes: Vec<SurfaceRect>,
    cells: Vec<CellData>,
}

/// Random small surface with painted pane interiors, matching the Python
/// model's size ranges. Returns `None` when no pane fits (skipped case).
fn make_case(rng: &mut Rng, two_panes: bool) -> Option<Case> {
    let width = rng.between(6, 20) as u16;
    let height = rng.between(4, 10) as u16;
    // Origins cluster at the surface corner, with the odd inset pane.
    let origin = |rng: &mut Rng| [0u16, 0, 1, 2][rng.below(4) as usize];
    let (x, y) = (origin(rng), origin(rng));
    if width - x < 2 || height - y < 2 {
        return None;
    }
    let width_1 = rng.between(2, u64::from(width - x)) as u16;
    let height_1 = rng.between(2, u64::from(height - y)) as u16;
    let mut panes = vec![SurfaceRect {
        x,
        y,
        width: width_1,
        height: height_1,
    }];
    if two_panes {
        let x2 = x + width_1 + rng.below(3) as u16;
        let rest = width.saturating_sub(x2);
        if rest < 2 {
            return None;
        }
        panes.push(SurfaceRect {
            x: x2,
            y,
            width: rng.between(2, u64::from(rest)) as u16,
            height: rng.between(2, u64::from(height - y)) as u16,
        });
    }
    let inside = |x: u16, y: u16| {
        panes.iter().any(|pane| {
            x >= pane.x && x - pane.x < pane.width && y >= pane.y && y - pane.y < pane.height
        })
    };
    let cells = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| {
            if inside(x, y) && !rng.chance(50) {
                random_content(rng, 10)
            } else {
                cell(" ", 0, 0)
            }
        })
        .collect();
    Some(Case {
        width,
        height,
        panes,
        cells,
    })
}

/// Advances the baseline like the pane emulator plus repaint noise: each pane
/// either rotates up by a random shift or stays put, vacated rows take fresh
/// content, then border rows, stray edits, and scrollbar-column cells land on
/// top. Edits outside a pane are what keep patch rows crossing regions.
fn mutate_next(rng: &mut Rng, cells: &mut [CellData], case: &Case) {
    let stride = usize::from(case.width);
    for pane in &case.panes {
        let (rows, cols) = (usize::from(pane.height), usize::from(pane.width));
        let shift = if rng.chance(50) {
            usize::try_from(rng.between(1, u64::from(pane.height - 1))).unwrap_or(0)
        } else {
            0
        };
        if shift > 0 {
            for column in 0..cols {
                let column = usize::from(pane.x) + column;
                let at = |row: usize| (usize::from(pane.y) + row) * stride + column;
                let source = (0..rows).map(|row| cells[at(row)].clone()).collect::<Vec<_>>();
                for row in 0..rows {
                    cells[at(row)] = source[(row + shift) % rows].clone();
                }
            }
        }
        for row in (rows - if shift > 0 { shift } else { 1 })..rows {
            for column in 0..cols {
                if rng.chance(85) {
                    let at = (usize::from(pane.y) + row) * stride + usize::from(pane.x) + column;
                    cells[at] = random_content(rng, 100);
                }
            }
        }
    }
    // Full-width border row crossing every pane: a table-border repaint.
    if rng.chance(45) {
        let row = rng.below(u64::from(case.height));
        let style = rng.below(3) as u32;
        let border = cell("=", style, if style > 0 { 1 } else { 0 });
        for column in 0..usize::from(case.width) {
            cells[row as usize * stride + column] = border.clone();
        }
    }
    for _ in 0..rng.below(3) {
        let pane = &case.panes[rng.below(case.panes.len() as u64) as usize];
        let row = usize::from(pane.y) + rng.below(u64::from(pane.height)) as usize;
        let column = usize::from(pane.x) + rng.below(u64::from(pane.width)) as usize;
        cells[row * stride + column] = random_content(rng, 100);
    }
    for pane in &case.panes {
        // Scrollbar column one cell right of the pane edge, when it exists.
        if pane.x + pane.width < case.width && rng.chance(50) {
            let row = usize::from(pane.y) + rng.below(u64::from(pane.height)) as usize;
            cells[row * stride + usize::from(pane.x + pane.width)] = cell("|", 1, 1);
        }
    }
}

/// The changed-cell spans the retained renderer would publish for `next`.
fn row_diff(
    base: &[CellData],
    next: &[CellData],
    width: usize,
    height: usize,
) -> Vec<PaneSurfacePatchRow> {
    let mut rows = Vec::new();
    for y in 0..height {
        let start = y * width;
        let mut x = 0;
        while x < width {
            if base[start + x] == next[start + x] {
                x += 1;
                continue;
            }
            let first = x;
            while x < width && base[start + x] != next[start + x] {
                x += 1;
            }
            rows.push(PaneSurfacePatchRow {
                x: first as u16,
                y: y as u16,
                cells: next[start + first..start + x].to_vec(),
            });
        }
    }
    rows
}

fn surface_panes(rects: &[SurfaceRect]) -> Vec<PaneSurfacePane> {
    rects
        .iter()
        .enumerate()
        .map(|(index, rect)| PaneSurfacePane {
            pane_id: format!("fuzz:p{index}"),
            content_revision: 1,
            rect: *rect,
            inner_rect: *rect,
            scrollbar_rect: None,
            scroll: None,
            focused: index == 0,
            mouse_reporting: false,
            sgr_pixel_mouse: false,
            alternate_screen_active: false,
            pixel_width: 0,
            pixel_height: 0,
        })
        .collect()
}

fn surface_frame(
    case: &Case,
    cells: Vec<CellData>,
    panes: Vec<PaneSurfacePane>,
) -> PaneSurfaceFrame {
    PaneSurfaceFrame {
        boot_id: "fuzz".into(),
        projection_revision: 1,
        surface_revision: 1,
        frame: FrameData {
            cells,
            width: case.width,
            height: case.height,
            cursor: None,
            hyperlinks: Vec::new(),
            graphics: Vec::new(),
        },
        panes,
        splits: Vec::new(),
        popup: None,
        graphics: Default::default(),
    }
}

/// Receivers only take the fast presentation path for non-overlapping rows.
fn assert_disjoint(seed: u64, rows: &[PaneSurfacePatchRow]) {
    for (index, a) in rows.iter().enumerate() {
        for b in &rows[index + 1..] {
            let overlap = a.y == b.y
                && usize::from(a.x) < usize::from(b.x) + b.cells.len()
                && usize::from(b.x) < usize::from(a.x) + a.cells.len();
            assert!(
                !overlap,
                "seed {seed}: expanded rows overlap: {a:?} / {b:?}"
            );
        }
    }
}

enum Outcome {
    Skipped,
    Plain,
    Scrolled,
}

/// One case: mutate a random surface, encode on a random wire version,
/// transmit through the real path (`message`/`message_v2` -> base64/bincode
/// -> `decode`/`decode_v2` -> `apply`), then assert the receiver converges on
/// the server-committed grid with disjoint rows, and that the baseline hashes
/// a v2 sender transports equal hashing the agreed baseline's scroll regions.
/// `apply` passing non-empty hashes proves the receiver's own pre-rotation
/// recomputation matched, which is exactly the #5750 guard firing on drift.
fn check_case(seed: u64, rng: &mut Rng, two_panes: bool) -> Outcome {
    let Some(case) = make_case(rng, two_panes) else {
        return Outcome::Skipped;
    };
    let stride = usize::from(case.width);
    let mut next = case.cells.clone();
    mutate_next(rng, &mut next, &case);

    let rows = row_diff(&case.cells, &next, stride, usize::from(case.height));
    let mut server = case.cells.clone();
    for row in &rows {
        let start = usize::from(row.y) * stride + usize::from(row.x);
        server[start..start + row.cells.len()].clone_from_slice(&row.cells);
    }

    let panes = surface_panes(&case.panes);
    let last = surface_frame(&case, case.cells.clone(), panes.clone());
    let patch = PaneSurfacePatch {
        boot_id: "fuzz".into(),
        projection_revision: 1,
        base_surface_revision: 1,
        surface_revision: 2,
        rows,
        panes,
        cursor: None,
    };
    let use_v2 = rng.chance(50);
    let encoded = if use_v2 {
        message_v2(&last, &patch)
    } else {
        message(&last, &patch)
    };
    let Some(ServerMessage::EndpointControl { kind, data }) = encoded else {
        return Outcome::Plain;
    };
    let expected_kind = if use_v2 { MESSAGE_KIND_V2 } else { MESSAGE_KIND };
    assert_eq!(kind, expected_kind, "seed {seed}: negotiated {use_v2}");
    let decoded = (if use_v2 {
        decode_v2(&data)
    } else {
        decode(&data)
    })
    .unwrap_or_else(|error| panic!("seed {seed}: decode failed: {error}"));
    let shifts = decoded
        .scrolls
        .iter()
        .map(|scroll| scroll.shift)
        .collect::<Vec<_>>();
    // The v2 sender must transport exactly the hashes of the baseline both
    // peers hold; v1 messages must carry none.
    if use_v2 {
        let expected = decoded
            .scrolls
            .iter()
            .map(|scroll| {
                rect_baseline_hash(&last.frame.cells, case.width, scroll.rect)
                    .expect("every generated pane region fits the baseline grid")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            decoded.hashes, expected,
            "seed {seed}: sender hashed something other than the shared baseline"
        );
    } else {
        assert!(
            decoded.hashes.is_empty(),
            "seed {seed}: v1 message carried baseline hashes"
        );
    }
    let mut client = last.frame.cells.clone();
    let expanded = apply(&mut client, case.width, case.height, decoded)
        .unwrap_or_else(|error| panic!("seed {seed}: apply rejected its own message: {error}"));

    assert_disjoint(seed, &expanded.rows);
    if client != server {
        let first = client
            .iter()
            .zip(&server)
            .position(|(client, server)| client != server)
            .unwrap_or(0);
        panic!(
            "seed {seed}: decoded grid differs from the committed grid at row {}, column {} \
             ({}x{}/{:?} shifts {shifts:?}); client {:?} server {:?}",
            first / stride,
            first % stride,
            case.width,
            case.height,
            case.panes,
            client[first],
            server[first],
        );
    }
    Outcome::Scrolled
}

fn case_count() -> u64 {
    std::env::var("HERDR_FUZZ_CASES")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(DEFAULT_CASES)
}

fn run_fuzz(label: &str, two_panes: bool) {
    let count = case_count();
    let (mut scrolled, mut plain, mut skipped) = (0u64, 0u64, 0u64);
    for seed in 0..count {
        let mut rng = Rng::new(seed);
        match check_case(seed, &mut rng, two_panes) {
            Outcome::Scrolled => scrolled += 1,
            Outcome::Plain => plain += 1,
            Outcome::Skipped => skipped += 1,
        }
    }
    // A generator that stopped producing scroll messages would silently make
    // the whole fuzz vacuous, so fail loudly on zero coverage.
    assert!(
        scrolled > 0,
        "{label}: no scroll message in {count} cases (plain {plain}, skipped {skipped}); \
         the fuzz is vacuous, check the generators"
    );
}

#[test]
fn random_single_pane_scrolls_converge_on_the_committed_grid() {
    run_fuzz("single-pane fuzz", false);
}

#[test]
fn random_two_pane_scrolls_converge_on_the_committed_grid() {
    run_fuzz("two-pane fuzz", true);
}
