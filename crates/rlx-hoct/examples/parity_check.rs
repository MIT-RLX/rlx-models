use ndarray::Array3;
use rlx_hoct::model::HoctModel;
fn main() {
    let model =
        HoctModel::from_weights("/tmp/hoct-inspect/weights/general_v0.safetensors").unwrap();
    let nf: Array3<f32> = ndarray_npy::read_npy("/tmp/hoct_ref_logits_node_features.npy").unwrap();
    let npos: Array3<f32> = ndarray_npy::read_npy("/tmp/hoct_ref_logits_node_pos.npy").unwrap();
    let epos: Array3<f32> = ndarray_npy::read_npy("/tmp/hoct_ref_logits_edge_pos.npy").unwrap();
    let eidx: Array3<i64> = ndarray_npy::read_npy("/tmp/hoct_ref_logits_edge_indices.npy").unwrap();
    let nmask: ndarray::Array2<bool> =
        ndarray_npy::read_npy("/tmp/hoct_ref_logits_node_mask.npy").unwrap();
    let emask: ndarray::Array2<bool> =
        ndarray_npy::read_npy("/tmp/hoct_ref_logits_edge_mask.npy").unwrap();
    let out = model.forward(
        &nf.view(),
        &npos.view(),
        &epos.view(),
        &eidx,
        &nmask,
        &emask,
    );
    let rlog: Array3<f32> = ndarray_npy::read_npy("/tmp/hoct_ref_logits.npy").unwrap();
    let rnode: Array3<f32> = ndarray_npy::read_npy("/tmp/hoct_ref_node_h.npy").unwrap();
    let redge: Array3<f32> = ndarray_npy::read_npy("/tmp/hoct_ref_edge_h.npy").unwrap();
    let dlog = (&out.edge_logits - &rlog)
        .mapv(f32::abs)
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    let dnode = (&out.node_hidden - &rnode)
        .mapv(f32::abs)
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    let dedge = (&out.edge_hidden - &redge)
        .mapv(f32::abs)
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    println!("max logit diff {dlog}");
    println!("max node diff {dnode}");
    println!("max edge diff {dedge}");
    println!("ours {:?}", out.edge_logits.iter().collect::<Vec<_>>());
    println!("ref  {:?}", rlog.iter().collect::<Vec<_>>());
}
