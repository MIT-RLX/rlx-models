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

/// A network that scores every window `p` (all weights zero, the last bias the logit).
fn constant(p: f32) -> rlx_wakeword_core::WakeCnnWeights {
    let mut w = rlx_wakeword_core::WakeCnnWeights::stub(rlx_wakeword_core::WakeCnnConfig::lite());
    for v in [
        &mut w.conv1_w,
        &mut w.conv2_w,
        &mut w.conv3_w,
        &mut w.fc1_w,
        &mut w.fc2_w,
    ] {
        v.iter_mut().for_each(|x| *x = 0.0);
    }
    w.fc2_b = vec![(p / (1.0 - p)).ln()];
    w
}

fn fires(sess: &mut rlx_wakeword::session::WakewordSession) -> usize {
    let tone: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.05).sin() * 0.1).collect();
    sess.push(&tone)
        .iter()
        .filter(|e| matches!(e, WakeEvent::Candidate { .. }))
        .count()
}

fn always_on() -> rlx_wakeword::WakewordBundle {
    let mut b = stub_bundle("wake", 40);
    b.weights[0].1 = constant(0.9);
    b.config.vad_gate = false;
    b.config.phrases[0].threshold = 0.5;
    b
}

/// The verifier vetoes what the detector fires on, unless it agrees.
#[test]
fn verifier_vetoes_unless_it_agrees() {
    assert!(
        fires(&mut always_on().open_session().unwrap()) > 0,
        "detector alone fires"
    );
    let mut no = always_on();
    no.verifier = Some((constant(0.1), 0.5, 1200.0));
    assert_eq!(
        fires(&mut no.open_session().unwrap()),
        0,
        "a disagreeing verifier vetoes"
    );
    let mut yes = always_on();
    yes.verifier = Some((constant(0.8), 0.5, 1200.0));
    assert!(
        fires(&mut yes.open_session().unwrap()) > 0,
        "an agreeing verifier lets it through"
    );
    let mut strict = always_on();
    strict.verifier = Some((constant(0.8), 0.5, 1200.0));
    strict.config.phrases[0].verify_threshold = Some(0.9);
    assert_eq!(
        fires(&mut strict.open_session().unwrap()),
        0,
        "the phrase's own verify threshold"
    );
}

/// Smoothing averages the last hops: a steady score above the threshold still fires.
#[test]
fn smoothing_keeps_a_steady_score() {
    let mut b = always_on();
    b.config.smooth_hops = 4;
    assert!(fires(&mut b.open_session().unwrap()) > 0);
}
