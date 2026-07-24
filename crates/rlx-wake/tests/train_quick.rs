//! RLX-only wake training must reduce loss on a synthetic pos/neg set.

use rlx_wake::train::{
    CnnTrainConfig, synth_pos_neg_dataset, train_new_lite_cnn, train_wake_cnn,
};
use rlx_wake::{WakeCnnConfig, WakeCnnWeights};

#[test]
fn cnn_train_loss_drops() {
    let clips = synth_pos_neg_dataset(6, 6, 1.0);
    let (w, report) = train_new_lite_cnn(&clips, "hey rlx", 25);
    assert!(
        report.improved(),
        "expected loss drop {:.4} -> {:.4}",
        report.initial_loss,
        report.final_loss
    );
    assert!(report.train_acc >= 0.5, "acc={}", report.train_acc);
    // Round-trip save/load shape sanity via stub cfg match
    assert_eq!(w.cfg.n_mels, WakeCnnConfig::lite().n_mels);
}

#[test]
fn cnn_train_config_runs() {
    let clips = synth_pos_neg_dataset(4, 4, 0.8);
    let mut w = WakeCnnWeights::stub(WakeCnnConfig::lite());
    let mut cfg = CnnTrainConfig::default();
    cfg.sgd.epochs = 5;
    cfg.sgd.log_every = 0;
    cfg.keyword = "test".into();
    let report = train_wake_cnn(&mut w, &clips, &cfg);
    assert!(report.final_loss.is_finite());
}
