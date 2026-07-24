use rlx_wakeword::bundle::stub_bundle;
use rlx_wakeword::session::WakeEvent;

#[test]
fn silence_mostly_idle() {
    let bundle = stub_bundle("wake", 40);
    let mut sess = bundle.open_session().unwrap();
    let pcm = vec![0.0f32; 16_000];
    let events = sess.push(&pcm);
    assert!(!events.is_empty());
    let idle = events
        .iter()
        .filter(|e| matches!(e, WakeEvent::Idle { .. }))
        .count();
    assert!(idle > 0);
}

#[test]
fn hop_20_and_40_run() {
    for hop in [20u32, 40] {
        let bundle = stub_bundle("wake", hop);
        let mut sess = bundle.open_session().unwrap();
        let _ = sess.push(&vec![0.01f32; 3200]);
    }
}

#[test]
fn threshold_hot_swap() {
    let bundle = stub_bundle("wake", 40);
    let mut sess = bundle.open_session().unwrap();
    sess.set_phrase_threshold("wake", 0.99);
    assert!((sess.config().phrases[0].threshold - 0.99).abs() < 1e-6);
}
