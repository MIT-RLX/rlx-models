# Committed linear-Resize import-parity fixture (+ onnxruntime ref). No Python at test time.
import onnx, numpy as np, onnxruntime as ort, os
from onnx import helper, TensorProto
FIX=os.path.dirname(os.path.abspath(__file__))
scales=np.array([1,1,2],dtype=np.float32)
g=helper.make_graph(
  [helper.make_node("Resize",["x","","scales"],["y"],mode="linear",coordinate_transformation_mode="half_pixel")],
  "resize_linear",[helper.make_tensor_value_info("x",TensorProto.FLOAT,[1,4,7])],
  [helper.make_tensor_value_info("y",TensorProto.FLOAT,[1,4,14])],
  [helper.make_tensor("scales",TensorProto.FLOAT,[3],scales.tobytes(),raw=True)])
m=helper.make_model(g,opset_imports=[helper.make_opsetid("",17)]); onnx.save(m,f"{FIX}/resize_linear.onnx")
x=np.random.RandomState(3).randn(1,4,7).astype(np.float32); x.tofile(f"{FIX}/x.f32")
ref=ort.InferenceSession(m.SerializeToString(),providers=["CPUExecutionProvider"]).run(["y"],{"x":x})[0]
ref.astype(np.float32).tofile(f"{FIX}/y_ref.f32"); print("wrote fixture, out",ref.shape)
