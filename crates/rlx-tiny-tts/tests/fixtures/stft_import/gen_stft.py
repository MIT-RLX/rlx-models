# Generates the committed STFT import-parity fixture (tiny STFT + onnxruntime ref).
# Run once with onnx+onnxruntime installed; no Python is needed at test time.
import onnx, numpy as np, onnxruntime as ort, os
FIX=os.path.dirname(os.path.abspath(__file__))
N,hop,L=20,5,200
win=(0.5-0.5*np.cos(2*np.pi*np.arange(N)/N)).astype(np.float32)  # Hann (win[0]=0)
g=onnx.helper.make_graph(
  [onnx.helper.make_node("STFT",["signal","step","win","flen"],["out"],onesided=1)],
  "stft_import_fixture",
  [onnx.helper.make_tensor_value_info("signal",1,[1,L])],
  [onnx.helper.make_tensor_value_info("out",1,[1,(L-N)//hop+1,N//2+1,2])],
  [onnx.numpy_helper.from_array(np.array(hop,dtype=np.int64),"step"),
   onnx.numpy_helper.from_array(win,"win"),
   onnx.numpy_helper.from_array(np.array(N,dtype=np.int64),"flen")])
mm=onnx.helper.make_model(g,opset_imports=[onnx.helper.make_opsetid("",17)])
onnx.save(mm, f"{FIX}/stft_import_fixture.onnx")
sig=np.random.RandomState(7).randn(1,L).astype(np.float32)
sig.tofile(f"{FIX}/signal.f32")
ref=ort.InferenceSession(mm.SerializeToString(),providers=["CPUExecutionProvider"]).run(["out"],{"signal":sig})[0]
ref.astype(np.float32).tofile(f"{FIX}/stft_ref.f32")
print("wrote fixture: out", ref.shape, "->", ref.size, "f32")
