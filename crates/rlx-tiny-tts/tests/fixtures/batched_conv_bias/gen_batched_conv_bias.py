# Generates a tiny BATCH-2 Conv-with-bias fixture + onnxruntime reference.
# Guards the rlx-onnx-import bias-broadcast bug: the per-channel bias must
# broadcast over the batch axis (reshape leading dim = 1), not be sized to the
# actual batch — otherwise batch elements >=1 read out-of-bounds bias.
# Depthwise + L_in != L_out mirrors the Supertonic CFG vector_estimator case.
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort, os
FX=os.path.dirname(os.path.abspath(__file__)); rng=np.random.default_rng(20260711)
N,C,L,K = 2,8,20,5    # depthwise, SAME-length conv (pad 2) -> L_out = 20
x=rng.standard_normal((N,C,L)).astype(np.float32)
w=rng.standard_normal((C,1,K)).astype(np.float32)
b=(rng.standard_normal((C,)).astype(np.float32))*3.0   # large bias so a wrong/OOB bias is obvious
node=helper.make_node("Conv",["x","w","b"],["y"],group=C,kernel_shape=[K],pads=[2,2],strides=[1],dilations=[1])
g=helper.make_graph([node],"batched_conv_bias",
    [helper.make_tensor_value_info("x",TensorProto.FLOAT,[N,C,L])],
    [helper.make_tensor_value_info("y",TensorProto.FLOAT,[N,C,L])],
    [numpy_helper.from_array(w,"w"),numpy_helper.from_array(b,"b")])
m=helper.make_model(g,opset_imports=[helper.make_opsetid("",13)]); onnx.checker.check_model(m)
onnx.save(m,f"{FX}/batched_conv_bias_fixture.onnx")
s=ort.InferenceSession(f"{FX}/batched_conv_bias_fixture.onnx",providers=["CPUExecutionProvider"])
y=s.run(None,{"x":x})[0].astype(np.float32)
x.tofile(f"{FX}/x.f32"); y.tofile(f"{FX}/y_ref.f32")
print("wrote fixture: x",x.shape,"-> y",y.shape,"(N=2 depthwise conv+bias)")
