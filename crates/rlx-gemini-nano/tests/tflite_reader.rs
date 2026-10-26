//! The TFLite reader against files written by the upstream `flatbuffers`
//! builder, so field slots are checked against an independent encoder.

use flatbuffers::{FlatBufferBuilder, TableFinishedWIPOffset, WIPOffset};
use rlx_gemini_nano::tflite::{BufferLoc, TensorType, TfliteModel};
use rlx_gemini_nano::{inspect, proto};

fn slot(i: u16) -> u16 {
    4 + 2 * i
}

/// Tensor bytes stored past the flatbuffer, the layout large models use.
const EXTERNAL: [u8; 6] = [10, 20, 30, 40, 50, 60];

/// `LlmParameters`-shaped metadata: field 1 = 18 (varint), field 2 = nested
/// message with field 3 = 2048, field 4 = "gelu".
fn metadata_proto() -> Vec<u8> {
    let nested = [0x18, 0x80, 0x10, 0x22, 4, b'g', b'e', b'l', b'u'];
    let mut out = vec![0x08, 18, 0x12, nested.len() as u8];
    out.extend_from_slice(&nested);
    out
}

fn build(external_offset: u64) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();

    // Buffers: 0 = sentinel, 1 = inline f32, 2 = external, 3 = metadata.
    let inline = b.create_vector(&[1.0f32, 2.0, 3.0, 4.0].map(f32::to_le_bytes).concat());
    let meta_bytes = b.create_vector(&metadata_proto());
    let mut buffers = Vec::new();
    for data in [None, Some(inline), None, Some(meta_bytes)] {
        let t = b.start_table();
        if let Some(d) = data {
            b.push_slot_always::<WIPOffset<_>>(slot(0), d);
        }
        buffers.push(b.end_table(t));
    }
    // Re-open buffer 2 as external: offset/size slots.
    let t = b.start_table();
    b.push_slot_always::<u64>(slot(1), external_offset);
    b.push_slot_always::<u64>(slot(2), EXTERNAL.len() as u64);
    buffers[2] = b.end_table(t);
    let buffers = b.create_vector(&buffers);

    // Quantization for the int8 tensor.
    let scale = b.create_vector(&[0.5f32, 0.25]);
    let zp = b.create_vector(&[0i64, 3]);
    let q = b.start_table();
    b.push_slot_always::<WIPOffset<_>>(slot(2), scale);
    b.push_slot_always::<WIPOffset<_>>(slot(3), zp);
    b.push_slot_always::<i32>(slot(6), 1);
    let q = b.end_table(q);

    let tensor = |b: &mut FlatBufferBuilder,
                  name: &str,
                  shape: &[i32],
                  ty: i8,
                  buf: u32,
                  q: Option<WIPOffset<TableFinishedWIPOffset>>| {
        let name = b.create_string(name);
        let shape = b.create_vector(shape);
        let t = b.start_table();
        b.push_slot_always::<WIPOffset<_>>(slot(0), shape);
        b.push_slot_always::<i8>(slot(1), ty);
        b.push_slot_always::<u32>(slot(2), buf);
        b.push_slot_always::<WIPOffset<_>>(slot(3), name);
        if let Some(q) = q {
            b.push_slot_always::<WIPOffset<_>>(slot(4), q);
        }
        b.end_table(t)
    };
    let t0 = tensor(&mut b, "params.lm.embed", &[2, 2], 0, 1, None);
    let t1 = tensor(&mut b, "params.lm.w_int8", &[2, 3], 9, 2, Some(q));
    let t2 = tensor(&mut b, "out", &[2, 3], 0, 0, None);
    let tensors = b.create_vector(&[t0, t1, t2]);

    let op_in = b.create_vector(&[0i32, 1]);
    let op_out = b.create_vector(&[2i32]);
    let op = b.start_table();
    b.push_slot_always::<u32>(slot(0), 1);
    b.push_slot_always::<WIPOffset<_>>(slot(1), op_in);
    b.push_slot_always::<WIPOffset<_>>(slot(2), op_out);
    let op = b.end_table(op);
    let ops = b.create_vector(&[op]);
    let sg_in = b.create_vector(&[0i32]);
    let sg_out = b.create_vector(&[2i32]);
    let sg_name = b.create_string("main");
    let sg = b.start_table();
    b.push_slot_always::<WIPOffset<_>>(slot(0), tensors);
    b.push_slot_always::<WIPOffset<_>>(slot(1), sg_in);
    b.push_slot_always::<WIPOffset<_>>(slot(2), sg_out);
    b.push_slot_always::<WIPOffset<_>>(slot(3), ops);
    b.push_slot_always::<WIPOffset<_>>(slot(4), sg_name);
    let sg = b.end_table(sg);
    let subgraphs = b.create_vector(&[sg]);

    // Opcode 0: FULLY_CONNECTED via the deprecated byte only (old files).
    // Opcode 1: a custom op. Opcode 2: GELU (150) via builtin_code only.
    let oc0 = b.start_table();
    b.push_slot_always::<i8>(slot(0), 9);
    let oc0 = b.end_table(oc0);
    let custom = b.create_string("odml.rms_norm");
    let oc1 = b.start_table();
    b.push_slot_always::<i8>(slot(0), 32);
    b.push_slot_always::<WIPOffset<_>>(slot(1), custom);
    b.push_slot_always::<i32>(slot(3), 32);
    let oc1 = b.end_table(oc1);
    let oc2 = b.start_table();
    b.push_slot_always::<i8>(slot(0), 127);
    b.push_slot_always::<i32>(slot(3), 150);
    let oc2 = b.end_table(oc2);
    let opcodes = b.create_vector(&[oc0, oc1, oc2]);

    let meta_name = b.create_string("odml.infra.LlmParameters");
    let m = b.start_table();
    b.push_slot_always::<WIPOffset<_>>(slot(0), meta_name);
    b.push_slot_always::<u32>(slot(1), 3);
    let m = b.end_table(m);
    let metadata = b.create_vector(&[m]);

    let desc = b.create_string("synthetic nano");
    let model = b.start_table();
    b.push_slot_always::<u32>(slot(0), 3);
    b.push_slot_always::<WIPOffset<_>>(slot(1), opcodes);
    b.push_slot_always::<WIPOffset<_>>(slot(2), subgraphs);
    b.push_slot_always::<WIPOffset<_>>(slot(3), desc);
    b.push_slot_always::<WIPOffset<_>>(slot(4), buffers);
    b.push_slot_always::<WIPOffset<_>>(slot(6), metadata);
    let model = b.end_table(model);
    b.finish(model, Some("TFL3"));
    b.finished_data().to_vec()
}

/// A file whose external buffer sits right after the flatbuffer. The offset
/// is a fixed-width u64 slot, so the flatbuffer's size does not depend on it.
fn file() -> Vec<u8> {
    let probe = build(2);
    let mut bytes = build(probe.len() as u64);
    assert_eq!(bytes.len(), probe.len());
    bytes.extend_from_slice(&EXTERNAL);
    bytes
}

#[test]
fn reads_tensors_buffers_and_quantization() {
    let m = TfliteModel::from_bytes(file()).unwrap();
    assert_eq!(m.version, 3);
    assert_eq!(m.description, "synthetic nano");
    assert_eq!(m.subgraphs.len(), 1);
    let sg = &m.subgraphs[0];
    assert_eq!(sg.name, "main");
    assert_eq!(
        (sg.inputs.as_slice(), sg.outputs.as_slice()),
        (&[0][..], &[2][..])
    );

    let embed = m.find_tensor("params.lm.embed").unwrap();
    assert_eq!(embed.shape, [2, 2]);
    assert_eq!(embed.dtype, TensorType::F32);
    assert!(embed.quantization.is_none());
    let vals: Vec<f32> = m
        .tensor_bytes(embed)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(vals, [1.0, 2.0, 3.0, 4.0]);

    let w = m.find_tensor("params.lm.w_int8").unwrap();
    assert_eq!(w.dtype, TensorType::I8);
    assert!(matches!(m.buffers[2], BufferLoc::External { .. }));
    assert_eq!(m.tensor_bytes(w).unwrap(), EXTERNAL);
    assert_eq!(w.packed_bytes(), Some(6));
    let q = w.quantization.as_ref().unwrap();
    assert_eq!(
        (q.scale.as_slice(), q.zero_point.as_slice()),
        (&[0.5, 0.25][..], &[0, 3][..])
    );
    assert_eq!(q.quantized_dimension, 1);

    assert_eq!(
        m.tensor_bytes(m.find_tensor("out").unwrap()).unwrap(),
        &[] as &[u8]
    );
}

#[test]
fn opcode_names_cover_deprecated_custom_and_extended_codes() {
    let m = TfliteModel::from_bytes(file()).unwrap();
    let names: Vec<String> = m.operator_codes.iter().map(|c| c.name()).collect();
    assert_eq!(names, ["FULLY_CONNECTED", "odml.rms_norm", "GELU"]);
    let hist = inspect::op_histogram(&m);
    assert_eq!(hist.get("odml.rms_norm"), Some(&1));
}

#[test]
fn metadata_protobuf_decodes() {
    let m = TfliteModel::from_bytes(file()).unwrap();
    let bytes = m
        .metadata_bytes("odml.infra.LlmParameters")
        .unwrap()
        .unwrap();
    assert_eq!(bytes, metadata_proto());
    let text = proto::render(bytes).unwrap();
    assert_eq!(text, "1: 18\n2 {\n  3: 2048\n  4: \"gelu\"\n}\n");
    assert!(inspect::report(&m, 10).unwrap().contains("4: \"gelu\""));
}

#[test]
fn truncated_and_foreign_files_are_errors() {
    let bytes = file();
    // Cut the external tensor's bytes off: lookups fail instead of reading past the end.
    let cut = TfliteModel::from_bytes(bytes[..bytes.len() - 1].to_vec()).unwrap();
    let w = cut.find_tensor("params.lm.w_int8").unwrap();
    assert!(cut.tensor_bytes(w).is_err());
    // Cutting into the flatbuffer itself fails at parse time.
    assert!(TfliteModel::from_bytes(bytes[..bytes.len() / 2].to_vec()).is_err());
    let mut foreign = bytes.clone();
    foreign[4..8].copy_from_slice(b"GGUF");
    assert!(TfliteModel::from_bytes(foreign).is_err());
}
