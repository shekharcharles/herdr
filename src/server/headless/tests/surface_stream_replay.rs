//! Streaming-replay regression: OMP-style output through the surface-scroll
//! transport must keep every downstream grid equal to the server committed
//! grid, and the exact bytes the client blit encoder writes to the host
//! terminal must replay (via an independent strict VT interpreter with
//! autowrap disabled, matching the emitter's assumption) into that grid.
//!
//! Adjudication map (one assertion per candidate mechanism):
//! - ASSERT-WIRE: expanded rows returned by the decoder, re-applied to an
//!   independent expected copy, must reconstruct the server committed grid.
//!   Failing ⇒ mis-rotated-but-valid scroll patch (wrong shift / missing
//!   residual) painted grid row N+k at row N.
//! - ASSERT-BLIT: strict replay of the exact `BlitEncoder` bytes must equal
//!   the composed grid. Failing ⇒ client emitter/consumer desync: grid
//!   perfect, host pixels spliced, self-heals on the next full repaint —
//!   the observed incident shape.
//! - ASSERT-HEAL: after a forced full render the authoritative composed
//!   frame must equal both the expected copy and the replayed terminal.
//!   Failing while the two per-frame asserts pass ⇒ server baseline drifts
//!   from the emulator grid (dirty-bit / blank-push classes).

use super::*;
use crate::protocol::render_ansi::BlitEncoder;
use crate::protocol::{CellData, CursorState, FrameData, PaneSurfacePatchRow};
use std::cmp;
use unicode_width::UnicodeWidthStr;

#[derive(Clone, Copy, Debug)]
struct Case {
    rows: u16,
    cols: u16,
    /// Lines streamed per render frame: 1..=3 pass the halving and
    /// >=2-row gates easily; 5 at tall viewports probes the mid band;
    /// 25 exceeds the gate and must fall back to a plain patch.
    burst: usize,
    unicode_borders: bool,
    /// Composed-area offset for patch rows (surface_patch.rs:183-185).
    area: (u16, u16),
    draw_cursor: bool,
    per_chunk_render: bool,
    /// Control arm: full renders only (chain off; mirrors the
    /// reveal_hidden_cursor_for_cjk_ime bypass structurally).
    force_full: bool,
    expect_scroll: bool,
    /// OSC 633 prompt-marked cells adjacent to blank runs (semantic bits
    /// excluded from the blank-skip presentation mask).
    wacky_presentation: bool,
    /// CLASS-3 driver: runtime.resize() WITHOUT a geometry wave at frame 4
    /// (stale retained area vs live grid; surface_patch keeps old geometry).
    resize_cols: Option<u16>,
    pure_append: bool,
}

fn padded_line(text: &str, width: usize) -> Vec<u8> {
    let mut s = text.to_string();
    while s.chars().count() < width {
        s.push(' ');
    }
    let truncated: String = s.chars().take(width).collect();
    let mut bytes = truncated.into_bytes();
    bytes.extend_from_slice(b"\r\n");
    bytes
}

fn border_row(width: usize, unicode: bool) -> String {
    if unicode {
        format!("+{}+", "\u{2500}".repeat(width.saturating_sub(2)))
    } else {
        format!("+{}+", "-".repeat(width.saturating_sub(2)))
    }
}

/// Table block: full-inner-width border rows (cross any scrolled region,
/// mirroring the d4a068c3 fixture shape) + data rows, trailing-space
/// padded to width-1.
fn table_block(width: usize, seq: usize, unicode: bool) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend(padded_line(&border_row(width, unicode), width - 1));
    out.extend(padded_line(
        &format!("| tool {:02} | result {:02} |", seq % 10, seq % 100),
        width - 1,
    ));
    out.extend(padded_line(
        &format!("| files {:02} | rows   {:02} |", seq % 8, (seq * 7) % 100),
        width - 1,
    ));
    out.extend(padded_line(&border_row(width, unicode), width - 1));
    out
}

fn filler_line(width: usize, n: usize) -> Vec<u8> {
    // Unique per streaming position: shift recovery must come from content,
    // not identical-row hash ties.
    padded_line(
        &format!("filler {n:04} :: streaming output line {n:04}"),
        width - 1,
    )
}

/// One streaming frame at EMULATOR coordinates: status-line rewrite at an
/// absolute CUP inside the pane, `\n`-chained burst at the bottom row
/// (forces emulator scroll), table repaint anchored mid-pane (crossing the
/// scrolled region), optional OSC 633 prompt cells next to blanks.
fn streaming_chunk(case: &Case, seq: usize) -> Vec<u8> {
    let width = usize::from(case.cols.saturating_sub(1).max(1));
    let h = case.rows.max(2);
    let mut out = Vec::new();
    if case.pure_append {
        out.extend(format!("\x1b[{h};1d").as_bytes().to_vec());
        for n in 0..case.burst.max(1) {
            out.extend(filler_line(width, seq * 10 + n));
        }
        return out;
    }
    out.extend(format!("\x1b[2;1H").as_bytes());
    out.extend(padded_line(&format!("== OMP seq {seq:02} =="), width));
    out.extend(b"\x1b[0m");
    let anchor = h / 3 + 1;
    out.extend(format!("\x1b[{anchor};1H").as_bytes());
    out.extend(padded_line(&border_row(width, case.unicode_borders), width));
    out.extend(b"\r\n");
    out.extend(format!("\x1b[{h};2H").as_bytes());
    for n in 0..case.burst.max(1) {
        out.extend(filler_line(width, seq * 10 + n));
    }
    if case.wacky_presentation {
        out.extend(b"\x1b]633;A\x07");
        out.extend(padded_line("prompt ", width));
    }
    out
}

// ---------------------------------------------------------------------------
// Strict mini VT interpreter: models exactly the emitter's alphabet
// (CUP, SGR, ED-2 on first full redraw, DEC private modes, DECSCUSR, OSC 8/
// 52, printable UTF-8) with DEC autowrap DISABLED. Anything else fails the
// harness, keeping the assumption under test explicit.
// ---------------------------------------------------------------------------

struct MiniVt {
    width: usize,
    height: usize,
    grid: Vec<Vec<String>>,
    cx: usize,
    cy: usize,
}

#[derive(Debug)]
enum Mode {
    Ground,
    Esc,
    Csi,
    Osc,
    OscEsc,
    Utf8(usize),
}

impl MiniVt {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            grid: vec![vec![" ".to_string(); width]; height],
            cx: 0,
            cy: 0,
        }
    }

    fn put(&mut self, symbol: &str) {
        let advance = cmp::max(1, symbol.width());
        if self.cx < self.width && self.cy < self.height {
            self.grid[self.cy][self.cx] = symbol.to_string();
        }
        self.cx = cmp::min(self.cx + advance, self.width);
    }
    fn rows(&self) -> Vec<Vec<String>> {
        self.grid.clone()
    }


    fn feed(&mut self, bytes: &[u8]) {
        let mut mode = Mode::Ground;
        let mut seq: Vec<u8> = Vec::new();
        let mut utf8: Vec<u8> = Vec::new();
        let mut i = 0usize;
        while i < bytes.len() {
            let b = bytes[i];
            i += 1;
            match mode {
                Mode::Ground => match b {
                    0x1b => {
                        seq.clear();
                        mode = Mode::Esc;
                    }
                    b if b.is_ascii() => {
                        if b >= 0x20 {
                            self.put((b as char).to_string().as_str());
                        }
                    }
                    _ => {
                        let left = match b {
                            x if x & 0xE0 == 0xC0 => 1,
                            x if x & 0xF0 == 0xE0 => 2,
                            x if x & 0xF8 == 0xF0 => 3,
                            x => panic!("mini-vt: invalid UTF-8 lead {x:#04x}"),
                        };
                        utf8.clear();
                        utf8.push(b);
                        mode = Mode::Utf8(left);
                    }
                },
                Mode::Utf8(left) => {
                    utf8.push(b);
                    if left == 1 {
                        let s = std::str::from_utf8(&utf8)
                            .unwrap_or_else(|e| panic!("mini-vt: bad utf8: {e}"));
                        self.put(s);
                        mode = Mode::Ground;
                    } else {
                        mode = Mode::Utf8(left - 1);
                    }
                }
                Mode::Esc => match b {
                    b'[' => mode = Mode::Csi,
                    b']' => mode = Mode::Osc,
                    _ => mode = Mode::Ground,
                },
                Mode::Csi => {
                    seq.push(b);
                    if (0x40..=0x7e).contains(&b) {
                        self.csi(&seq);
                        mode = Mode::Ground;
                    }
                }
                Mode::Osc => {
                    if b == 0x07 {
                        mode = Mode::Ground;
                    } else if b == 0x1b {
                        mode = Mode::OscEsc;
                    }
                }
                Mode::OscEsc => {
                    mode = if b == b'\\' {
                        Mode::Ground
                    } else {
                        Mode::Osc
                    };
                }
            }
        }
        assert!(
            matches!(mode, Mode::Ground),
            "mini-vt: truncated escape at end of frame ({mode:?})"
        );
    }

    fn csi(&mut self, seq: &[u8]) {
        let final_byte = *seq.last().expect("csi final byte");
        let body = &seq[..seq.len() - 1];
        match final_byte {
            b'H' | b'f' => {
                let (row, col) = parse_params(body);
                self.cy = row.unwrap_or(1).saturating_sub(1).min(self.height - 1);
                self.cx = col.unwrap_or(1).saturating_sub(1).min(self.width);
            }
            b'J' if params_first(body) == Some(2) => {
                for row in &mut self.grid {
                    for cell in row {
                        *cell = " ".to_string();
                    }
                }
                self.cx = 0;
                self.cy = 0;
            }
            b'J' => panic!("mini-vt: unexpected ED param body={body:?}"),
            b'K' | b'@' | b'P' | b'X' => {
                panic!("mini-vt: unexpected erase/insert final {final_byte:#02x}")
            }
            b'm' | b'q' | b't' => {}
            b'd' => {
                self.cy = params_first(body)
                    .map(|n| n.saturating_sub(1))
                    .unwrap_or(0)
                    .min(self.height.saturating_sub(1));
            }
            b'h' | b'l' if body.starts_with(b"?") => {}
            other => panic!("mini-vt: unexpected CSI final {other:#02x} body={body:?}"),
        }
    }
}

fn parse_params(body: &[u8]) -> (Option<usize>, Option<usize>) {
    let text = String::from_utf8_lossy(body);
    let mut it = text.trim_start_matches('?').split(';');
    let first = it.next().and_then(|p| p.parse::<usize>().ok());
    let second = it.next().and_then(|p| p.parse::<usize>().ok());
    match (first, second) {
        (Some(a), Some(b)) => (Some(a), Some(b)),
        (Some(a), None) => (Some(a), Some(1)),
        _ => (None, None),
    }
}

fn params_first(body: &[u8]) -> Option<usize> {
    String::from_utf8_lossy(body)
        .trim_start_matches('?')
        .split(';')
        .next()?
        .parse()
        .ok()
}

// ---------------------------------------------------------------------------
// Compose / expected-grid helpers
// ---------------------------------------------------------------------------

fn blank_cell() -> CellData {
    CellData {
        symbol: " ".into(),
        fg: 0,
        bg: 0,
        modifier: 0,
        skip: false,
        hyperlink: None,
    }
}

fn grid_of(frame: &FrameData) -> Vec<Vec<String>> {
    (0..usize::from(frame.height))
        .map(|y| {
            (0..usize::from(frame.width))
                .map(|x| {
                    let cell = &frame.cells[y * usize::from(frame.width) + x];
                    if cell.skip || cell.symbol.is_empty() {
                        " ".to_string()
                    } else {
                        cell.symbol.clone()
                    }
                })
                .collect()
        })
        .collect()
}

fn compose(frame: &FrameData, area: (u16, u16)) -> FrameData {
    let (ax, ay) = (usize::from(area.0), usize::from(area.1));
    let width = usize::from(frame.width) + ax;
    let height = usize::from(frame.height) + ay;
    let mut cells = vec![blank_cell(); width * height];
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if row < ay || col < ax {
                cells[idx].symbol = "#".to_string();
            } else {
                cells[idx]
                    .symbol
                    .clone_from(&frame.cells[(row - ay) * usize::from(frame.width) + col - ax].symbol);
            }
        }
    }
    FrameData {
        cells,
        width: width as u16,
        height: height as u16,
        cursor: frame.cursor.clone().map(|cursor| CursorState {
            x: cursor.x + area.0,
            y: cursor.y + area.1,
            visible: cursor.visible,
            shape: cursor.shape,
        }),
        hyperlinks: Vec::new(),
        graphics: Vec::new(),
    }
}

fn frame_from_grid(grid: &[Vec<String>], cursor: Option<CursorState>) -> FrameData {
    let height = grid.len();
    let width = grid.first().map_or(0, |row| row.len());
    let mut cells = vec![blank_cell(); width * height];
    for (y, row) in grid.iter().enumerate() {
        for (x, symbol) in row.iter().enumerate() {
            cells[y * width + x].symbol.clone_from(symbol);
        }
    }
    FrameData {
        cells,
        width: width as u16,
        height: height as u16,
        cursor,
        hyperlinks: Vec::new(),
        graphics: Vec::new(),
    }
}

fn compose_patch_rows(
    rows: &[PaneSurfacePatchRow],
    cursor: Option<CursorState>,
    area: (u16, u16),
) -> (Vec<PaneSurfacePatchRow>, Option<CursorState>) {
    let rows = rows
        .iter()
        .map(|row| PaneSurfacePatchRow {
            x: row.x + area.0,
            y: row.y + area.1,
            cells: row.cells.clone(),
        })
        .collect();
    let cursor = cursor.map(|cursor| CursorState {
        x: cursor.x + area.0,
        y: cursor.y + area.1,
        visible: cursor.visible,
        shape: cursor.shape,
    });
    (rows, cursor)
}

fn apply_rows_to_grid(grid: &mut [Vec<String>], rows: &[PaneSurfacePatchRow]) {
    for row in rows {
        let y = usize::from(row.y);
        if y >= grid.len() {
            continue;
        }
        for (offset, cell) in row.cells.iter().enumerate() {
            let x = usize::from(row.x) + offset;
            if x < grid[y].len() {
                grid[y][x] = if cell.skip || cell.symbol.is_empty() {
                    " ".into()
                } else {
                    cell.symbol.clone()
                };
            }
        }
    }
}
fn apply_rows_to_frame(frame: &mut FrameData, rows: &[PaneSurfacePatchRow]) {
    for row in rows {
        let start = usize::from(row.y) * usize::from(frame.width) + usize::from(row.x);
        if let Some(target) = frame.cells.get_mut(start..start + row.cells.len()) {
            target.clone_from_slice(&row.cells);
        }
    }
}

fn server_grid(server: &mut HeadlessServer) -> Vec<Vec<String>> {
    let frame = server
        .clients
        .get_mut(&1)
        .unwrap()
        .render_state
        .last_pane_surface()
        .expect("committed baseline")
        .frame
        .clone();
    grid_of(&frame)
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------


fn replay_server(initial: &[u8], cols: u16, rows: u16) -> (
    HeadlessServer,
    std::sync::mpsc::Receiver<Vec<u8>>,
    std::sync::mpsc::Receiver<Vec<u8>>,
    crate::layout::PaneId,
) {
    let mut server = test_headless_server();
    let mut workspace = crate::workspace::Workspace::test_new("test");
    let pane_id = workspace.focused_pane_id().expect("focused pane");
    workspace.insert_test_runtime(
        pane_id,
        crate::terminal::TerminalRuntime::test_with_screen_bytes(cols, rows, initial),
    );
    server.app.state.workspaces = vec![workspace];
    server.app.state.active = Some(0);
    server.app.state.selected = 0;
    server.app.state.mode = crate::app::Mode::Terminal;
    let (writer, control_rx, render_rx) = test_client_writer();
    server.clients.insert(
        1,
        ClientConnection::new(
            (cols, rows),
            crate::kitty_graphics::HostCellSize::default(),
            1,
            RenderEncoding::SemanticFrame,
            Some(writer),
        ),
    );
    server.foreground_client_id = Some(1);
    server.sync_foreground_client_state();
    assert!(server.claim_unowned_shell_tab_geometry(1, true));
    (server, control_rx, render_rx, pane_id)
}

fn replay_case(case: Case) {
    let initial: Vec<u8> =
        (1..=case.rows * 4)
            .flat_map(|n| padded_line(&format!("initial row {n:03} :: baseline content"), 60))
            .collect();
    let (mut server, _control_rx, render_rx, pane_id) =
        replay_server(&initial, case.cols.max(10) as u16, case.rows.max(4) as u16);
    {
        let client = server.clients.get_mut(&1).expect("client 1");
        client.render_state.enable_surface_delta(true);
        client.render_state.enable_surface_scroll(true);
    }
    server.render_and_stream();

    let mut decoder = protocol::surface_reuse::Decoder::new(true, true);
    let mut blit = BlitEncoder::new();
    // MiniVt sized from the actual baseline frame below.

    // Baseline full frame.
    let baseline = decoder
        .decode(read_server_message(
            render_rx
                .recv_timeout(Duration::from_secs(2))
                .expect("baseline frame"),
        ))
        .expect("baseline decode");
    let surface0 = match baseline {
        ServerMessage::PaneSurface(surface) => surface,
        other => panic!("expected baseline pane surface, got {other:?}"),
    };
    let mut mini = MiniVt::new(
        usize::from(surface0.frame.width) + usize::from(case.area.0),
        usize::from(surface0.frame.height) + usize::from(case.area.1),
    );
    let mut last_surface = surface0.clone();
    let mut expected_frame = surface0.frame.clone();
    let composed0 = compose(&expected_frame, case.area);
    let encoded0 = blit.encode(&composed0, true);
    mini.feed(&encoded0.bytes);
    blit.commit(composed0.clone(), encoded0);
    assert_eq!(
        mini.rows(),
        grid_of(&composed0),
        "baseline replay mismatch before any streaming"
    );
    let mut expected_grid = grid_of(&expected_frame);

    let mut scroll_messages = 0usize;
    let mut patch_messages = 0usize;
    let mut seq = 0usize;
    let mut pending: Vec<Vec<u8>> = Vec::new();
    for frame_no in 0..14usize {
        pending.push(streaming_chunk(&case, seq));
        seq += 1;
        if let Some(cols) = case.resize_cols {
            if frame_no == 4 {
                let runtime = server
                    .app
                    .state
                    .runtime_for_pane_in_workspace(&server.app.terminal_runtimes, 0, pane_id)
                    .expect("pane runtime");
                runtime.resize(case.rows, cols, 10, 20);
            }
        }
        if frame_no % 3 == 2 {
            pending.push(table_block(usize::from(case.cols), seq, case.unicode_borders));
            seq += 1;
        }
        if case.force_full {
            server.clients.get_mut(&1).unwrap().request_recompute();
        }
        if !case.per_chunk_render && frame_no % 2 == 0 {
            continue;
        }
        for chunk in pending.drain(..) {
            write_shared_test_pane(&mut server, pane_id, &chunk);
        }
        server.render_retained_pane_surface_and_stream(&HashSet::from([pane_id]));
        server.render_and_stream();
        loop {
            let Ok(bytes) = render_rx.recv_timeout(Duration::from_millis(400)) else {
                break;
            };
            let message = read_server_message(bytes);
            if matches!(
                &message,
                ServerMessage::EndpointControl { kind, .. }
                    if kind == protocol::surface_scroll::MESSAGE_KIND
            ) {
                scroll_messages += 1;
            }
            let decoded = decoder.decode(message).expect("wire decode");
            match decoded {
                ServerMessage::PaneSurface(surface) => {
                    expected_frame = surface.frame.clone();
                    expected_grid = grid_of(&surface.frame);
                    last_surface = surface.clone();
                    let composed = compose(&surface.frame, case.area);
                    let encoded = blit.encode(&composed, false);
                    mini.feed(&encoded.bytes);
                    blit.commit(composed.clone(), encoded);
                    assert_eq!(
                        mini.rows(),
                        grid_of(&composed),
                        "ASSERT-BLIT FAIL (full frame) {frame_no}: replay diverged"
                    );
                }
                ServerMessage::PaneSurfacePatch(patch) => {
                    patch_messages += 1;
                    {
                        let old_rows = grid_of(&expected_frame);
                        let mut new_rows = old_rows.clone();
                        apply_rows_to_grid(&mut new_rows, &patch.rows);
                        let dirty: Vec<usize> =
                            patch.rows.iter().map(|row| usize::from(row.y)).collect();
                        let mut report = format!(
                            "PROBE frame {frame_no}: patch_rows={} dirty_y={dirty:?} ",
                            patch.rows.len()
                        );
                        for shift in [0isize, 1, 2, 3, 4] {
                            let m = (0..old_rows.len())
                                .filter(|&y| {
                                    let s = y as isize + shift;
                                    s >= 0
                                        && (s as usize) < old_rows.len()
                                        && new_rows[y] == old_rows[s as usize]
                                })
                                .count();
                            report.push_str(&format!("m({shift})={m} "));
                        }
                        let would_scroll =
                            protocol::surface_scroll::message(&last_surface, &patch).is_some();
                        report.push_str(&format!("would_scroll={would_scroll}"));
                        eprintln!("{report}");
                    }
                    // ASSERT-WIRE: independently re-apply the returned rows.
                    apply_rows_to_grid(&mut expected_grid, &patch.rows);
                    let committed = server_grid(&mut server);
                    assert_eq!(
                        expected_grid, committed,
                        "ASSERT-WIRE FAIL frame {frame_no}: decoder-expanded rows do not \
                         reconstruct the server committed grid (mis-rotated scroll patch)"
                    );
                    let (rows, cursor) =
                        compose_patch_rows(&patch.rows, patch.cursor.clone(), case.area);
                    let rows = if case.draw_cursor {
                        blit.patch_rows_with_drawn_cursor(&rows, cursor.as_ref())
                            .unwrap_or_default()
                    } else {
                        rows
                    };
                    match blit.encode_patch(&rows, cursor.clone(), case.draw_cursor) {
                        Some(encoded) => {
                            mini.feed(&encoded.bytes);
                            assert!(
                                blit.commit_patch(&rows, cursor.clone(), encoded),
                                "ASSERT-BLIT: commit_patch refused rows outside the frame"
                            );
                        }
                        None => {
                            // Mirror the present_surface_patch encode fallback:
                            // client repaints the composed frame wholesale.
                            let repaint = compose(
                                &frame_from_grid(&expected_grid, patch.cursor.clone()),
                                case.area,
                            );
                            let encoded = blit.encode(&repaint, true);
                            mini.feed(&encoded.bytes);
                            blit.commit(repaint, encoded);
                        }
                    }
                    // ASSERT-BLIT: strict terminal replay vs composed grid.
                    let composed_expected = compose(
                        &frame_from_grid(&expected_grid, patch.cursor.clone()),
                        case.area,
                    );
                    assert_eq!(
                        mini.rows(),
                        grid_of(&composed_expected),
                        "ASSERT-BLIT FAIL frame {frame_no}: emitted bytes replay into a \
                         different terminal grid (sticky splice class)"
                    );
                    expected_frame = frame_from_grid(&expected_grid, patch.cursor.clone());
                    apply_rows_to_frame(&mut last_surface.frame, &patch.rows);
                    last_surface.frame.cursor = patch.cursor.clone();
                    last_surface.panes = patch.panes.clone();
                    last_surface.surface_revision = patch.surface_revision;
                }
                other => {
                    let _ = other;
                }
            }
        }
    }

    // ASSERT-HEAL: forced full render re-synchronizes everything.
    server.app.state.reveal_hidden_cursor_for_cjk_ime = true;
    server.clients.get_mut(&1).unwrap().request_recompute();
    server.render_and_stream();
    let heal = decoder
        .decode(read_server_message(
            render_rx
                .recv_timeout(Duration::from_secs(2))
                .expect("heal frame"),
        ))
        .expect("heal decode");
    let surface = match heal {
        ServerMessage::PaneSurface(surface) => surface,
        other => panic!("expected heal pane surface, got {other:?}"),
    };
    let heal_grid = grid_of(&surface.frame);
    assert_eq!(
        expected_grid, heal_grid,
        "ASSERT-HEAL FAIL: server committed grid diverged from the authoritative \
         recomputed grid (dirty-bit / blank-push class)"
    );
    let composed = compose(&surface.frame, case.area);
    let encoded = blit.encode(&composed, true);
    mini.feed(&encoded.bytes);
    blit.commit(composed.clone(), encoded);
    assert_eq!(
        mini.rows(),
        grid_of(&composed),
        "ASSERT-HEAL FAIL: full repaint did not resynchronize the terminal"
    );

    assert!(
        patch_messages > 0 || case.force_full,
        "workload produced no patches at all ({case:?})"
    );
    if case.expect_scroll {
        assert!(
            scroll_messages > 0,
            "workload never triggered the scroll transport ({case:?}, \
             patches={patch_messages}); tune burst/geometry so the halving \
             and >=2-row gates pass"
        );
    }
    shutdown_test_runtimes(&mut server);
}

fn base_case() -> Case {
    Case {
        rows: 24,
        cols: 80,
        burst: 2,
        unicode_borders: false,
        area: (0, 0),
        draw_cursor: false,
        per_chunk_render: true,
        force_full: false,
        expect_scroll: true,
        wacky_presentation: false,
        resize_cols: None,
        pure_append: true,
    }
}

#[tokio::test]
async fn stream_replay_omp_tables_scroll_transport() {
    replay_case(base_case());
}

#[tokio::test]
async fn stream_replay_unicode_borders_area_offset_drawn_cursor() {
    replay_case(Case {
        unicode_borders: true,
        area: (1, 2),
        draw_cursor: true,
        pure_append: false,
        expect_scroll: false,
        ..base_case()
    });
}

#[tokio::test]
async fn stream_replay_burst_five_tall_viewport_mid_gate() {
    replay_case(Case {
        rows: 40,
        burst: 5,
        ..base_case()
    });
}

#[tokio::test]
async fn stream_replay_accumulated_chunks_coalesce_shift() {
    replay_case(Case {
        rows: 40,
        burst: 3,
        per_chunk_render: false,
        ..base_case()
    });
}

#[tokio::test]
async fn stream_replay_burst_25_gate_fail_plain_patch() {
    replay_case(Case {
        burst: 25,
        expect_scroll: false,
        pure_append: false,
        ..base_case()
    });
}

#[tokio::test]
async fn stream_replay_wacky_presentation_cells_next_to_blanks() {
    replay_case(Case {
        wacky_presentation: true,
        ..base_case()
    });
}

#[tokio::test]
async fn stream_replay_full_render_control_arm() {
    replay_case(Case {
        force_full: true,
        expect_scroll: false,
        burst: 3,
        ..base_case()
    });
}

#[tokio::test]
async fn stream_replay_resize_without_wave_class3() {
    replay_case(Case {
        resize_cols: Some(60),
        ..base_case()
    });
}
