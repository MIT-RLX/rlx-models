//! Train path must bind every available RLX backend (same slotting as infer).

use rlx_wake::train::{CnnTrainConfig, synth_pos_neg_dataset, train_wake_cnn};
use rlx_wake::{WakeCnnConfig, WakeCnnWeights, available_devices, bind_streaming_device};

#[test]
fn cnn_train_binds_all_available_backends() {
    let clips = synth_pos_neg_dataset(4, 4, 0.8);
    for device in available_devices() {
        let (exec, label) = bind_streaming_device(device).expect("bind");
        assert_eq!(exec, device);
        let mut w = WakeCnnWeights::stub(WakeCnnConfig::lite());
        let mut cfg = CnnTrainConfig::default();
        cfg.sgd.epochs = 3;
        cfg.sgd.log_every = 0;
        cfg.keyword = format!("wake-{label}");
        let report = train_wake_cnn(&mut w, &clips, &cfg);
        assert!(
            report.final_loss.is_finite(),
            "non-finite loss on {label}: {:?}",
            report
        );
    }
}
