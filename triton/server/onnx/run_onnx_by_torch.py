import onnxruntime
import torch
import numpy as np


batch_size = 1    # just a random number

x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
torch_out = torch_model(x)
ort_session = onnxruntime.InferenceSession("super_resolution.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(
    to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
