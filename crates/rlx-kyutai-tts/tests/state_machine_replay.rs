//! Replay Python temp=0 sampled tokens through the Rust state machine.

use rlx_kyutai_tts::download::{default_kyutai_tts_dir, tokenizer_path};
use rlx_kyutai_tts::state_machine::{StateMachine, script_to_entries};
use rlx_kyutai_tts::tokenizer::KyutaiTokenizer;

const PY_SAMPLED: &[u32] = &[3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 0];

const PY_OUT: &[u32] = &[
    3, 3, 8002, 2613794, 3, 3, 3, 8897, 2432565, 3, 3, 3, 3, 3, 3, 3, 8326, 2136270, 3, 8304,
];

#[test]
fn rust_state_machine_matches_python_temp0_trace() {
    let dir = std::env::var("RLX_KYUTAI_TTS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| default_kyutai_tts_dir());
    if !dir.join("dsm_tts_1e68beda@240.safetensors").is_file() {
        eprintln!("skip: missing weights");
        return;
    }
    let tok = KyutaiTokenizer::load(tokenizer_path(&dir)).expect("tok");
    let prompt = "Hello world, this is a test of the Kyutai text to speech system.";
    let entries = script_to_entries(&tok, prompt).expect("entries");
    let sm = StateMachine::for_config(8000, 2);
    let mut st = sm.new_state(entries);
    for step in 0..PY_SAMPLED.len() {
        let out = sm.process(step, &mut st, PY_SAMPLED[step]);
        let expected = PY_OUT[step];
        assert_eq!(
            out, expected,
            "step {step}: sampled={} rust_out={out} py_out={expected} end={:?}",
            PY_SAMPLED[step], st.end_step
        );
    }
}
