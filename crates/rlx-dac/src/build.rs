use crate::config::DacConfig;
use crate::layers::{
    Decoder, DecoderBlock, Encoder, EncoderBlock, ResidualUnit, WnConv1d, WnConvTranspose1d,
};
use crate::ops::weight_norm;
use crate::weights::WeightStore;
use anyhow::{Result, ensure};
use ndarray::Array1;

fn load_alpha(store: &WeightStore, key: &str) -> Result<Array1<f32>> {
    let (data, shape) = store.get(key)?;
    ensure!(shape.len() == 3 && shape[0] == 1, "{key} shape {shape:?}");
    let channels = shape[1];
    Ok(Array1::from_shape_vec(channels, data.to_vec())?)
}

fn load_wn_conv1d(
    store: &WeightStore,
    prefix: &str,
    stride: usize,
    pad: usize,
    dilation: usize,
    same_length: bool,
) -> Result<WnConv1d> {
    let (g_data, g_shape) = store.get(&format!("{prefix}.weight_g"))?;
    let (v_data, v_shape) = store.get(&format!("{prefix}.weight_v"))?;
    ensure!(
        g_shape.len() == 3 && v_shape.len() == 3 && g_shape[0] == v_shape[0],
        "{prefix} weight_g/v out-channel mismatch: g={g_shape:?} v={v_shape:?}"
    );
    let weight = weight_norm(
        ndarray::ArrayView3::from_shape((g_shape[0], g_shape[1], g_shape[2]), g_data)?,
        ndarray::ArrayView3::from_shape((v_shape[0], v_shape[1], v_shape[2]), v_data)?,
    );
    let bias = store
        .get_optional(&format!("{prefix}.bias"))?
        .map(|(d, s)| Array1::from_shape_vec(s[0], d).expect("bias shape"));
    Ok(WnConv1d {
        weight,
        bias,
        stride,
        pad,
        dilation,
        same_length,
    })
}

fn load_wn_conv_transpose1d(
    store: &WeightStore,
    prefix: &str,
    stride: usize,
    pad: usize,
) -> Result<WnConvTranspose1d> {
    let (g_data, g_shape) = store.get(&format!("{prefix}.weight_g"))?;
    let (v_data, v_shape) = store.get(&format!("{prefix}.weight_v"))?;
    ensure!(
        g_shape.len() == 3 && v_shape.len() == 3 && g_shape[0] == v_shape[0],
        "{prefix} weight_g/v out-channel mismatch: g={g_shape:?} v={v_shape:?}"
    );
    let weight = weight_norm(
        ndarray::ArrayView3::from_shape((g_shape[0], g_shape[1], g_shape[2]), g_data)?,
        ndarray::ArrayView3::from_shape((v_shape[0], v_shape[1], v_shape[2]), v_data)?,
    );
    let bias = store
        .get_optional(&format!("{prefix}.bias"))?
        .map(|(d, s)| Array1::from_shape_vec(s[0], d).expect("bias shape"));
    Ok(WnConvTranspose1d {
        weight,
        bias,
        stride,
        pad,
        out_offset: 0,
    })
}

fn load_residual_unit(
    store: &WeightStore,
    prefix: &str,
    _dim: usize,
    dilation: usize,
) -> Result<ResidualUnit> {
    let pad = ((7 - 1) * dilation) / 2;
    Ok(ResidualUnit {
        snake1_alpha: load_alpha(store, &format!("{prefix}.0.alpha"))?,
        conv1: load_wn_conv1d(store, &format!("{prefix}.1"), 1, pad, dilation, true)?,
        snake2_alpha: load_alpha(store, &format!("{prefix}.2.alpha"))?,
        conv2: load_wn_conv1d(store, &format!("{prefix}.3"), 1, 0, 1, true)?,
    })
}

fn load_encoder_block(
    store: &WeightStore,
    prefix: &str,
    dim: usize,
    stride: usize,
) -> Result<EncoderBlock> {
    let half = dim / 2;
    let pad = stride.div_ceil(2);
    Ok(EncoderBlock {
        residual_units: [
            load_residual_unit(store, &format!("{prefix}.block.0.block"), half, 1)?,
            load_residual_unit(store, &format!("{prefix}.block.1.block"), half, 3)?,
            load_residual_unit(store, &format!("{prefix}.block.2.block"), half, 9)?,
        ],
        snake_alpha: load_alpha(store, &format!("{prefix}.block.3.alpha"))?,
        downsample: load_wn_conv1d(store, &format!("{prefix}.block.4"), stride, pad, 1, false)?,
    })
}

pub fn build_encoder(cfg: &DacConfig, store: &WeightStore) -> Result<Encoder> {
    let stem = load_wn_conv1d(store, "encoder.block.0", 1, 3, 1, true)?;
    let mut blocks = Vec::with_capacity(cfg.encoder_rates.len());
    let mut dim = cfg.encoder_dim;
    for (i, &stride) in cfg.encoder_rates.iter().enumerate() {
        dim *= 2;
        blocks.push(load_encoder_block(
            store,
            &format!("encoder.block.{}", i + 1),
            dim,
            stride,
        )?);
    }
    let tail = cfg.encoder_rates.len() + 1;
    let snake_alpha = load_alpha(store, &format!("encoder.block.{tail}.alpha"))?;
    let head = load_wn_conv1d(store, &format!("encoder.block.{}", tail + 1), 1, 1, 1, true)?;
    Ok(Encoder {
        stem,
        blocks,
        snake_alpha,
        head,
    })
}

fn load_decoder_block(
    store: &WeightStore,
    prefix: &str,
    _input_dim: usize,
    output_dim: usize,
    stride: usize,
) -> Result<DecoderBlock> {
    let pad = stride.div_ceil(2);
    Ok(DecoderBlock {
        snake_alpha: load_alpha(store, &format!("{prefix}.block.0.alpha"))?,
        upsample: load_wn_conv_transpose1d(store, &format!("{prefix}.block.1"), stride, pad)?,
        residual_units: [
            load_residual_unit(store, &format!("{prefix}.block.2.block"), output_dim, 1)?,
            load_residual_unit(store, &format!("{prefix}.block.3.block"), output_dim, 3)?,
            load_residual_unit(store, &format!("{prefix}.block.4.block"), output_dim, 9)?,
        ],
    })
}

pub fn build_decoder(cfg: &DacConfig, store: &WeightStore) -> Result<Decoder> {
    let stem = load_wn_conv1d(store, "decoder.model.0", 1, 3, 1, true)?;
    let mut blocks = Vec::with_capacity(cfg.decoder_rates.len());
    for (i, &stride) in cfg.decoder_rates.iter().enumerate() {
        let input_dim = cfg.decoder_dim / 2usize.pow(i as u32);
        let output_dim = cfg.decoder_dim / 2usize.pow(i as u32 + 1);
        blocks.push(load_decoder_block(
            store,
            &format!("decoder.model.{}", i + 1),
            input_dim,
            output_dim,
            stride,
        )?);
    }
    let tail = cfg.decoder_rates.len() + 1;
    let snake_alpha = load_alpha(store, &format!("decoder.model.{tail}.alpha"))?;
    let head = load_wn_conv1d(store, &format!("decoder.model.{}", tail + 1), 1, 3, 1, true)?;
    Ok(Decoder {
        stem,
        blocks,
        snake_alpha,
        head,
        c_snake: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn residual_unit_crop_matches_manual() {
        let x = ndarray::Array2::<f32>::ones((4, 10));
        let w = Array3::<f32>::ones((4, 4, 1));
        let ru = ResidualUnit {
            snake1_alpha: Array1::ones(4),
            conv1: WnConv1d {
                weight: w.clone(),
                bias: None,
                stride: 1,
                pad: 0,
                dilation: 1,
                same_length: true,
            },
            snake2_alpha: Array1::ones(4),
            conv2: WnConv1d {
                weight: w,
                bias: None,
                stride: 1,
                pad: 0,
                dilation: 1,
                same_length: true,
            },
        };
        let y = ru.forward(x.view());
        assert_eq!(y.shape(), x.shape());
    }
}
