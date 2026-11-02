//! Regression guards: patch-applied retained surface must equal a fresh full
//! recomposition after EVERY write batch at production cadence
//! (write -> collect(+clear) -> apply), over the cell classes the shallow
//! migration gate does not exercise: repeated scroll rotations, OSC 11
//! background changes across collect boundaries, bg runs with trailing
//! blanks, and heavy grapheme clusters.

use super::*;

struct Probe {
    pane: PaneTerminal,
    tx: mpsc::Sender<Bytes>,
    w: u16,
    h: u16,
    retained: Vec<CellData>,
}

impl Probe {
    fn new(w: u16, h: u16) -> Self {
        let (tx, _rx) = mpsc::channel(16);
        let terminal = crate::ghostty::Terminal::new(w, h, 256).unwrap();
        Self {
            pane: PaneTerminal::new(GhosttyPaneTerminal::new(terminal, tx.clone()).unwrap()),
            tx,
            w,
            h,
            retained: vec![CellData::from_ratatui_cell(&ratatui::buffer::Cell::default())
                ; usize::from(w) * usize::from(h)],
        }
    }

    fn write(&mut self, bytes: &[u8]) {
        let _ = self
            .pane
            .process_pty_bytes(PaneId::from_raw(1), 0, bytes, &self.tx);
    }

    /// Independent full-render oracle (identical to migration_tests.rs:72).
    fn full_cells(&self) -> Vec<CellData> {
        let backend = ratatui::backend::TestBackend::new(self.w, self.h);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        let mut cells = Vec::new();
        terminal
            .draw(|frame| {
                self.pane
                    .render(frame, Rect::new(0, 0, self.w, self.h), false);
                cells = frame
                    .buffer_mut()
                    .content
                    .iter()
                    .map(CellData::from_ratatui_cell)
                    .collect();
            })
            .unwrap();
        cells
    }

    /// Production cadence: collect consumes (and on success clears) dirty
    /// state; the patch is what the server retains client-side.
    fn collect_and_apply(&mut self) -> TerminalDirtyPatchOutcome {
        let outcome = self.pane.collect_dirty_patch(self.w, self.h);
        if let TerminalDirtyPatchOutcome::Patch(patch) = &outcome {
            for (row, cells) in &patch.rows {
                let start = usize::from(*row) * usize::from(self.w);
                self.retained[start..start + usize::from(self.w)].clone_from_slice(cells);
            }
        }
        outcome
    }

    fn assert_retained_matches_full(&self, context: &str) {
        let expected = self.full_cells();
        for (i, (got, want)) in self.retained.iter().zip(&expected).enumerate() {
            if got != want {
                let y = i / usize::from(self.w);
                let x = i % usize::from(self.w);
                panic!(
                    "{context}: first divergence at row {y} col {x}\n  patch-applied: {got:?}\n  recompose:     {want:?}"
                );
            }
        }
    }
}

/// Invariant: a blank pushed by blank-skip must byte-equal the recomposed cell
/// for the same blank WITH its real per-cell/default attrs, including OSC 11
/// custom backgrounds (OMP status bars/tables paint bg) across collect
/// boundaries.
#[test]
fn blank_fidelity_osc11_and_sgr_runs() {
    let mut p = Probe::new(10, 4);
    let batches: Vec<Vec<u8>> = vec![
        b"\x1b]11;#123456\x07".to_vec(), // custom background
        b"\x1b[41mSTATUS \x1b[0m tail\r\n".to_vec(), // colored run + trailing blanks
        b"\x1b]11;#abcdef\x07".to_vec(), // bg change ACROSS a collect boundary
        b"plain\r\n".to_vec(),
        b"\x1b]11;#000000\x07".to_vec(),
        b"after\r\n".to_vec(),
    ];
    for batch in &batches {
        p.write(batch);
        let outcome = p.collect_and_apply();
        assert!(
            !matches!(outcome, TerminalDirtyPatchOutcome::Fallback),
            "batch {batch:?}: unexpected collect fallback"
        );
        p.assert_retained_matches_full(&format!("batch {batch:?}"));
    }
}

/// Invariant: ONLY a default-background change, after a collect that
/// baked and cleared blanks. If the color change does not dirty every visible
/// row, the retained surface keeps stale baked blanks while recomposition
/// re-tints.
#[test]
fn stale_blank_after_default_bg_change_forces_repush() {
    let mut p = Probe::new(10, 3);
    p.write(b"abc\r\ndef\r\nghi");
    p.collect_and_apply();
    p.assert_retained_matches_full("baseline");
    p.write(b"\x1b]11;#00ff00\x07");
    let outcome = p.collect_and_apply();
    // A Clean outcome here violates this invariant:
    if matches!(outcome, TerminalDirtyPatchOutcome::Clean) {
        panic!(
            "OSC 11 background change reported Clean: every previously pushed \
             blank keeps the pre-change baked background on the client until INV-4"
        );
    }
    p.assert_retained_matches_full("after OSC 11 change");
}

/// Invariant: per-frame grid-vs-patch-applied equality through CONTINUOUS
/// scroll rotation (one rotation per batch, 40 frames = viewport cycles),
/// with wide CJK + combining marks + SGR bg runs in the streaming content.
#[test]
fn scroll_frame_fidelity_production_cadence() {
    let mut p = Probe::new(12, 5);
    for frame in 0..40u32 {
        let batch = format!("\x1b[42mf{frame:02}\x1b[0m \u{754c}\u{9762}e\u{301}\r\nline {frame}\r\n");
        p.write(batch.as_bytes());
        let outcome = p.collect_and_apply();
        assert!(
            !matches!(outcome, TerminalDirtyPatchOutcome::Fallback),
            "frame {frame}: unexpected fallback {outcome:?}"
        );
        p.assert_retained_matches_full(&format!("scroll frame {frame} batch {batch:?}"));
    }
}

/// Err-path regression guard: heavy grapheme clusters through collect. If
/// ghostty_buffer_symbol_into errs (or its width check blanks, terminal.rs:3229),
/// the patch carries blank+sticky while the grid holds the cluster.
#[test]
fn grapheme_heavy_clusters_survive_collect() {
    let mut p = Probe::new(20, 3);
    let marks = "\u{300}".repeat(160);
    let zwj = "\u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f467}\u{200d}\u{1f466}";
    let batches: Vec<Vec<u8>> =
        vec![format!("a{marks}b\r\n").into_bytes(), format!("{zwj}x{zwj}\r\n").into_bytes()];
    for batch in &batches {
        p.write(batch);
        p.collect_and_apply();
        p.assert_retained_matches_full(&format!("grapheme batch len {}", batch.len()));
    }
}

/// Invariant: ONLY a default-foreground change (OSC 10) between collects.
/// Same defect class as the OSC 11 background guard: previously pushed
/// default-attribute cells keep the pre-change baked foreground on the
/// client while recomposition re-tints them.
#[test]
fn stale_cells_after_fg_only_color_change_between_collects() {
    let mut p = Probe::new(10, 3);
    p.write(b"abc\r\ndef\r\nghi");
    p.collect_and_apply();
    p.assert_retained_matches_full("baseline");
    p.write(b"\x1b]10;#00ff00\x07");
    let outcome = p.collect_and_apply();
    if matches!(outcome, TerminalDirtyPatchOutcome::Clean) {
        panic!(
            "OSC 10 foreground change reported Clean: every previously pushed \
             default-attribute cell keeps the pre-change baked foreground on \
             the client until INV-4"
        );
    }
    p.assert_retained_matches_full("after OSC 10 change");
}

/// Re-setting the SAME default background between collects must not force a
/// repaint: once the change has been pushed, an OSC 11 to the identical
/// color leaves the collect Clean (no spurious full render).
#[test]
fn same_color_osc11_between_collects_stays_clean() {
    let mut p = Probe::new(10, 3);
    p.write(b"abc\r\ndef\r\nghi");
    p.collect_and_apply();
    p.write(b"\x1b]11;#00ff00\x07");
    let first = p.collect_and_apply();
    assert!(
        !matches!(first, TerminalDirtyPatchOutcome::Clean),
        "default background change must not report Clean (outcome {first:?})"
    );
    p.assert_retained_matches_full("after bg change");
    p.write(b"\x1b]11;#00ff00\x07");
    let second = p.collect_and_apply();
    assert!(
        matches!(second, TerminalDirtyPatchOutcome::Clean),
        "re-setting the SAME background must stay Clean (no spurious full \
         render), got {second:?}"
    );
    p.assert_retained_matches_full("after same-color OSC 11");
}

/// The forced full-viewport push must resolve after ONE collect: a second
/// collect with no further writes is Clean (the recorded pair absorbed the
/// change), and a subsequent full render — the production heal path, which
/// also records the baked pair — must not re-arm the force either. Guards
/// against a sticky repaint loop.
#[test]
fn default_color_force_does_not_stick_across_collects() {
    let mut p = Probe::new(10, 3);
    p.write(b"abc\r\ndef\r\nghi");
    p.collect_and_apply();
    p.write(b"\x1b]11;#00ff00\x07");
    let forced = p.collect_and_apply();
    let TerminalDirtyPatchOutcome::Patch(patch) = &forced else {
        panic!(
            "expected full-viewport patch after default bg change, got {forced:?}"
        );
    };
    assert_eq!(
        patch.rows.len(),
        usize::from(p.h),
        "forced patch must cover every viewport row"
    );
    p.assert_retained_matches_full("forced re-push");
    assert!(
        matches!(p.collect_and_apply(), TerminalDirtyPatchOutcome::Clean),
        "second collect after one forced push must be Clean (no repaint loop)"
    );

    // Re-arm check: a fresh default-color move after the pair settled must
    // force again (the guard is a live comparison, not a one-shot latch).
    p.write(b"\x1b]11;#ff0000\x07");
    let forced_again = p.collect_and_apply();
    assert!(
        matches!(
            forced_again,
            TerminalDirtyPatchOutcome::Patch(_)
        ),
        "a second default-color change must re-force cleanly, got {forced_again:?}"
    );
    p.assert_retained_matches_full("second forced re-push");
    assert!(
        matches!(p.collect_and_apply(), TerminalDirtyPatchOutcome::Clean),
        "collect after the second forced push must be Clean again"
    );
    let recompose = p.full_cells();
    assert_eq!(recompose.len(), usize::from(p.w) * usize::from(p.h));
    assert!(
        matches!(p.collect_and_apply(), TerminalDirtyPatchOutcome::Clean),
        "collect after a full render (which records the baked pair) must \
         stay Clean (no sticky repaint loop)"
    );
    p.assert_retained_matches_full("settled");
}

