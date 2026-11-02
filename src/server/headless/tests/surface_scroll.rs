use super::*;

use base64::engine::general_purpose::STANDARD_NO_PAD;

fn scrolling_lines(range: std::ops::Range<usize>) -> Vec<u8> {
    range
        .map(|n| {
            format!(
                "build step {n:>3}: compiled module_{n} in {}ms\r\n",
                n * 7 % 97
            )
        })
        .collect::<String>()
        .into_bytes()
}

fn apply_rows(frame: &mut FrameData, rows: &[crate::protocol::PaneSurfacePatchRow]) {
    for row in rows {
        let start = usize::from(row.y) * usize::from(frame.width) + usize::from(row.x);
        frame.cells[start..start + row.cells.len()].clone_from_slice(&row.cells);
    }
}

#[tokio::test]
async fn surface_scroll_sends_scrolling_output_as_a_shift_and_new_rows() {
    let (mut server, _control_rx, render_rx, pane_id) =
        retained_test_server_with_control(&scrolling_lines(0..40));
    server
        .clients
        .get_mut(&1)
        .expect("scroll client")
        .render_state
        .enable_surface_scroll(true);
    server.render_and_stream();
    let mut decoder = protocol::surface_reuse::Decoder::new(false, true);
    let ServerMessage::PaneSurface(mut shell) = decoder
        .decode(read_server_message(
            render_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        ))
        .expect("initial surface")
    else {
        panic!("expected the initial pane surface");
    };

    write_shared_test_pane(&mut server, pane_id, &scrolling_lines(40..42));
    assert!(server.render_retained_pane_surface_and_stream(&HashSet::from([pane_id])));
    let bytes = render_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    let message = read_server_message(bytes.clone());
    assert!(matches!(
        &message,
        ServerMessage::EndpointControl { kind, .. } if kind == protocol::surface_scroll::MESSAGE_KIND
    ));

    // The decoder hands the client shell an ordinary patch that reaches the
    // server's committed surface exactly.
    let ServerMessage::PaneSurfacePatch(patch) = decoder.decode(message).expect("scroll decode")
    else {
        panic!("expected an expanded pane patch");
    };
    apply_rows(&mut shell.frame, &patch.rows);
    let committed = server.clients[&1]
        .render_state
        .last_pane_surface()
        .expect("committed surface");
    assert_eq!(shell.frame.cells, committed.frame.cells);
    assert!(frame_text(&shell.frame).contains("module_41"));

    let expanded = HeadlessServer::frame_server_message(&ServerMessage::PaneSurfacePatch(patch))
        .expect("expanded frame");
    assert!(
        bytes.len() * 4 < expanded.len(),
        "scroll frame {} should be far smaller than the rows it replaces {}",
        bytes.len(),
        expanded.len()
    );
    shutdown_test_runtimes(&mut server);
}

#[tokio::test]
async fn surface_scroll_is_not_sent_to_a_peer_that_did_not_negotiate_it() {
    let (mut server, _control_rx, render_rx, pane_id) =
        retained_test_server_with_control(&scrolling_lines(0..40));
    server.render_and_stream();
    let _ = render_rx.recv_timeout(Duration::from_secs(1)).unwrap();

    write_shared_test_pane(&mut server, pane_id, &scrolling_lines(40..42));
    assert!(server.render_retained_pane_surface_and_stream(&HashSet::from([pane_id])));
    let message = read_server_message(render_rx.recv_timeout(Duration::from_secs(1)).unwrap());
    assert!(matches!(message, ServerMessage::PaneSurfacePatch(_)));
    shutdown_test_runtimes(&mut server);
}

/// Two bottom-row lines with a nearly-full-width border row interleaved:
/// every scrolled frame then carries a residual row that straddles the
/// scrolled region (the d4a068c3 crossing shape), which is exactly the patch
/// shape a mis-rotated scroll would splice. The border stops one column
/// short of the pane width — an exactly-full-width row sets pending-wrap at
/// the bottom row and swallows the scrolling line feed, the same width-1
/// padding idiom the stream-replay harness documents for its table rows.
fn table_burst(seq: usize) -> Vec<u8> {
    let border = format!("+{}+\r\n", "-".repeat(77));
    let row = format!("build step {seq:>3}: compiled module_{seq} in {}ms\r\n", seq * 7 % 97);
    [border.into_bytes(), row.into_bytes()].concat()
}

fn enable_scroll_v2(server: &mut HeadlessServer, client_id: u64) {
    let client = server
        .clients
        .get_mut(&client_id)
        .expect("scroll v2 client");
    client.render_state.enable_surface_scroll(true);
    client.render_state.enable_surface_scroll_v2(true);
}

/// Curated summary for the kind asserts so a failure names what actually
/// arrived without dumping whole frames into the test log.
fn message_summary(message: &ServerMessage) -> String {
    match message {
        ServerMessage::EndpointControl { kind, data } => {
            format!("endpoint control {kind} ({} payload bytes)", data.len())
        }
        ServerMessage::PaneSurfacePatch(patch) => {
            format!("plain patch ({} rows)", patch.rows.len())
        }
        other => {
            let mut text = format!("{other:?}");
            text.truncate(text.len().min(160));
            text
        }
    }
}

/// Flips the first byte of the first per-rect baseline hash in a v2 payload,
/// leaving every other byte — including the framed inner patch — identical.
fn tamper_first_hash(data: &str) -> String {
    let scroll_count = protocol::surface_scroll::decode_v2(data)
        .expect("untampered v2 payload decodes")
        .scrolls
        .len();
    let mut bytes = STANDARD_NO_PAD.decode(data).expect("v2 payload is base64");
    // Frozen v2 layout: one count byte, then scroll_count x 10 scroll bytes;
    // the hash section follows the scroll section.
    bytes[1 + scroll_count * 10] ^= 0x01;
    STANDARD_NO_PAD.encode(bytes)
}

#[tokio::test]
async fn surface_scroll_v2_streams_table_bursts_with_hash_guarded_baselines() {
    let (mut server, _control_rx, render_rx, pane_id) =
        retained_test_server_with_control(&scrolling_lines(0..40));
    enable_scroll_v2(&mut server, 1);
    server.render_and_stream();
    let mut decoder = protocol::surface_reuse::Decoder::new(false, true);
    let ServerMessage::PaneSurface(mut shell) = decoder
        .decode(read_server_message(
            render_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        ))
        .expect("initial surface")
    else {
        panic!("expected the initial pane surface");
    };

    let mut v2_frames = 0usize;
    for seq in 40..44 {
        write_shared_test_pane(&mut server, pane_id, &table_burst(seq));
        assert!(server.render_retained_pane_surface_and_stream(&HashSet::from([pane_id])));
        let bytes = render_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let message = read_server_message(bytes.clone());
        let shipped_v2 = matches!(
            &message,
            ServerMessage::EndpointControl { kind, .. }
                if kind == protocol::surface_scroll::MESSAGE_KIND_V2
        );
        v2_frames += shipped_v2 as usize;
        eprintln!("KIND {seq}: {}", message_summary(&message));
        let patch = match decoder.decode(message) {
            Ok(ServerMessage::PaneSurfacePatch(patch)) => patch,
            other => panic!("frame {seq}: expected a pane patch, got {other:?}"),
        };
        apply_rows(&mut shell.frame, &patch.rows);
        let committed = server.clients[&1]
            .render_state
            .last_pane_surface()
            .expect("committed surface");
        assert_eq!(shell.frame.cells, committed.frame.cells);
        assert!(frame_text(&committed.frame).contains(&format!("module_{seq}")));
        if shipped_v2 {
            let expanded =
                HeadlessServer::frame_server_message(&ServerMessage::PaneSurfacePatch(patch))
                    .expect("expanded frame");
            assert!(
                bytes.len() * 2 < expanded.len(),
                "v2 scroll frame {} should be far smaller than the rows it replaces {}",
                bytes.len(),
                expanded.len()
            );
        }
        let text = frame_text(&committed.frame);
        for (row, line) in text.lines().enumerate() {
            eprintln!("GRID {seq} {row:>2}: |{}|", line.trim_end());
        }
    }
    assert!(v2_frames >= 2, "stream produced only {v2_frames} v2 frames");
    shutdown_test_runtimes(&mut server);
}

#[tokio::test]
async fn surface_scroll_v2_hash_mismatch_repairs_within_one_reconnect() {
    let (mut server, _control_rx, render_rx, pane_id) =
        retained_test_server_with_control(&scrolling_lines(0..40));
    enable_scroll_v2(&mut server, 1);
    server.render_and_stream();
    let mut decoder = protocol::surface_reuse::Decoder::new(false, true);
    let ServerMessage::PaneSurface(mut shell) = decoder
        .decode(read_server_message(
            render_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        ))
        .expect("initial surface")
    else {
        panic!("expected the initial pane surface");
    };

    // The scroll gate decides per frame and may legitimately fall back to
    // a plain patch, so the corruption targets the first frame that
    // actually ships as v2; everything before it still has to land the
    // committed grid exactly.
    let mut seq = 40usize;
    let message = loop {
        write_shared_test_pane(&mut server, pane_id, &table_burst(seq));
        assert!(server.render_retained_pane_surface_and_stream(&HashSet::from([pane_id])));
        let message = read_server_message(render_rx.recv_timeout(Duration::from_secs(1)).unwrap());
        if matches!(
            &message,
            ServerMessage::EndpointControl { kind, .. }
                if kind == protocol::surface_scroll::MESSAGE_KIND_V2
        ) {
            break message;
        }
        let ServerMessage::PaneSurfacePatch(patch) =
            decoder.decode(message).expect("pre-tamper decode")
        else {
            panic!("expected a pane patch before the first v2 frame");
        };
        apply_rows(&mut shell.frame, &patch.rows);
        seq += 1;
        assert!(seq < 48, "the stream never produced a v2 scroll frame");
    };

    // Corrupt one baseline hash on the wire of that v2 frame.
    let ServerMessage::EndpointControl { kind, data } = &message else {
        unreachable!("the loop only breaks on a v2 endpoint control");
    };
    let tampered = ServerMessage::EndpointControl {
        kind: kind.clone(),
        data: tamper_first_hash(data),
    };
    // The client's real transport turns this Err into Io(InvalidData) and
    // drops the connection (src/client/transport.rs); headless asserts the
    // decoder verdict directly.
    let error = decoder
        .decode(tampered)
        .expect_err("a corrupted baseline hash must reject the scroll");
    assert!(
        error.contains("surface scroll baseline hash mismatch"),
        "unexpected guard error: {error}"
    );

    // Zero mutation on mismatch: the untampered same-revision frame still
    // decodes and lands the committed grid exactly. Had the rejected attempt
    // rotated the baseline, this re-delivery would itself hash-mismatch.
    let ServerMessage::PaneSurfacePatch(patch) = decoder
        .decode(message)
        .expect("untampered re-delivery still applies")
    else {
        panic!("expected an expanded pane patch");
    };
    apply_rows(&mut shell.frame, &patch.rows);
    assert_eq!(
        shell.frame.cells,
        server.clients[&1]
            .render_state
            .last_pane_surface()
            .expect("committed surface")
            .frame
            .cells
    );

    // Bounded repair: reconnect (fresh decoder; the server answers a client
    // without a baseline with one full frame) and stream one more burst. The
    // very next frame must re-equalize client grid, committed grid, and the
    // fresh emulator content — no sticky splice survives the reconnect.
    let mut reconnect = protocol::surface_reuse::Decoder::new(false, true);
    server
        .clients
        .get_mut(&1)
        .expect("client")
        .render_state
        .reset_baseline();
    write_shared_test_pane(&mut server, pane_id, &table_burst(seq + 1));
    server.render_and_stream();
    let recovery = reconnect
        .decode(read_server_message(
            render_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        ))
        .expect("reconnect resync");
    let ServerMessage::PaneSurface(surface) = recovery else {
        panic!("expected a full pane surface after resync");
    };
    assert_eq!(
        surface.frame.cells,
        server.clients[&1]
            .render_state
            .last_pane_surface()
            .expect("committed surface")
            .frame
            .cells
    );
    let text = frame_text(&surface.frame);
    assert!(
        text.contains(&format!("module_{}", seq + 1)) && text.contains("+------"),
        "resync frame must carry the newest border-crossing burst"
    );
    shutdown_test_runtimes(&mut server);
}

#[tokio::test]
async fn surface_scroll_v1_frames_are_unchanged_without_the_v2_flag() {
    let (mut server, _control_rx, render_rx, pane_id) =
        retained_test_server_with_control(&scrolling_lines(0..40));
    server
        .clients
        .get_mut(&1)
        .expect("scroll client")
        .render_state
        .enable_surface_scroll(true);
    server.render_and_stream();
    // A decoder that would happily accept v2 kinds still round-trips the
    // untouched v1 wire format from a server that never saw the flag.
    let mut decoder = protocol::surface_reuse::Decoder::new(false, true);
    let ServerMessage::PaneSurface(mut shell) = decoder
        .decode(read_server_message(
            render_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        ))
        .expect("initial surface")
    else {
        panic!("expected the initial pane surface");
    };

    write_shared_test_pane(&mut server, pane_id, &table_burst(40));
    assert!(server.render_retained_pane_surface_and_stream(&HashSet::from([pane_id])));
    let bytes = render_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    let message = read_server_message(bytes.clone());
    assert!(
        matches!(
            &message,
            ServerMessage::EndpointControl { kind, .. }
                if kind == protocol::surface_scroll::MESSAGE_KIND
        ),
        "without the v2 flag the server must keep emitting frozen v1 frames; got {}",
        message_summary(&message)
    );
    let ServerMessage::PaneSurfacePatch(patch) =
        decoder.decode(message).expect("v1 scroll decode")
    else {
        panic!("expected an expanded pane patch");
    };
    apply_rows(&mut shell.frame, &patch.rows);
    assert_eq!(
        shell.frame.cells,
        server.clients[&1]
            .render_state
            .last_pane_surface()
            .expect("committed surface")
            .frame
            .cells
    );
    assert!(frame_text(&shell.frame).contains("module_40"));
    shutdown_test_runtimes(&mut server);
}

#[tokio::test]
async fn surface_scroll_v2_without_the_scroll_capability_streams_plain_patches() {
    let (mut server, _control_rx, render_rx, pane_id) =
        retained_test_server_with_control(&scrolling_lines(0..40));
    server
        .clients
        .get_mut(&1)
        .expect("v2-only client")
        .render_state
        .enable_surface_scroll_v2(true);
    server.render_and_stream();
    let _ = render_rx.recv_timeout(Duration::from_secs(1)).unwrap();

    // The v2 flag only refines scroll delivery; a client that never
    // negotiated scroll keeps receiving ordinary expanded patches.
    write_shared_test_pane(&mut server, pane_id, &table_burst(40));
    assert!(server.render_retained_pane_surface_and_stream(&HashSet::from([pane_id])));
    let message = read_server_message(render_rx.recv_timeout(Duration::from_secs(1)).unwrap());
    assert!(matches!(message, ServerMessage::PaneSurfacePatch(_)));
    shutdown_test_runtimes(&mut server);
}
