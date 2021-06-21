from pathlib import Path
from os import environ
from psutil import cpu_count

# Constants from the performance optimization available in onnxruntime
# It needs to be done before importing onnxruntime
environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
environ["OMP_WAIT_POLICY"] = 'ACTIVE'

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers

from contextlib import contextmanager
from dataclasses import dataclass
from time import time
from tqdm import trange


def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:
    """
    这里解释一下ExecutionProvider，ONNXRuntime用Provider表示不同的运行设备比如CUDAProvider等。
    目前ONNX Runtime v1.0支持了包括CPU，CUDA，TensorRT，MKL等七种Providers。
    :param model_path:
    :param provider:
    :return:
    """
    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()

    return session


@contextmanager
def track_infer_time(buffer: [int]):
    start = time()
    yield
    end = time()

    buffer.append(end - start)


@dataclass
class OnnxInferenceResult:
    model_inference_time: [int]
    optimized_model_path: str


"""
一个推理的例子
"""
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
cpu_model = create_model_for_provider("onnx/bert-base-cased.onnx", "CPUExecutionProvider")

# Inputs are provided through numpy array
model_inputs = tokenizer("My name is Bert", return_tensors="pt")
inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}

# Run the model (None = get all the outputs)
sequence, pooled = cpu_model.run(None, inputs_onnx)

# Print information about outputs

print(f"sequence output: {sequence.shape}, Pooled output: {pooled.shape}")


"""
Benchmarking PyTorch model
"""
from transformers import BertModel
PROVIDERS = {
    ("cpu", "PyTorch CPU"),
    # Uncomment this line to enable GPU benchmarking
    # ("cuda:0", "PyTorch GPU")
}

results = {}
for device, label in PROVIDERS:
    # Move inputs to the correct device
    model_inputs_on_device = {arg_name: tensor.to(device) for arg_name, tensor in model_inputs.items()}

    # Add PyTorch to the providers
    model_pt = BertModel.from_pretrained("bert-base-cased").to(device)
    for _ in trange(10, desc="Warming up"):
        model_pt(**model_inputs_on_device)

    # Compute
    time_buffer = []
    for _ in trange(100, desc=f"Tracking inference time on PyTorch"):
        with track_infer_time(time_buffer):
            model_pt(**model_inputs_on_device)

    # Store the result
    results[label] = OnnxInferenceResult(time_buffer, None)


# print(results)


"""
Benchmarking PyTorch & ONNX on CPU
"""
PROVIDERS = {
    ("CPUExecutionProvider", "ONNX CPU"),
    # Uncomment this line to enable GPU benchmarking
    # ("CUDAExecutionProvider", "ONNX GPU")  # 很奇怪用的还是CPU
}

for provider, label in PROVIDERS:
    # Create the model with the specified provider
    model = create_model_for_provider("onnx/bert-base-cased.onnx", provider)

    # Keep track of the inference time
    time_buffer = []

    # Warm up the model
    model.run(None, inputs_onnx)

    # Compute
    for _ in trange(100, desc=f"Tracking inference time on {provider}"):
        with track_infer_time(time_buffer):
            model.run(None, inputs_onnx)

    # Store the result
    results[label] = OnnxInferenceResult(time_buffer, model.get_session_options().optimized_model_filepath)


# print(results)


"""
将耗时画出来
"""
import matplotlib.pyplot as plt
import numpy as np

# Compute average inference time + std
time_results = {k: np.mean(v.model_inference_time) * 1e3 for k, v in results.items()}
time_results_std = np.std([v.model_inference_time for v in results.values()]) * 1000

plt.rcdefaults()
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_ylabel("Avg Inference time (ms)")
ax.set_title("Average inference time (ms) for each provider")
ax.bar(time_results.keys(), time_results.values(), yerr=time_results_std)
plt.savefig("./average_inference_time.png")
plt.show()


"""
Benchmarking PyTorch quantized model
"""
import torch

# Quantize
model_pt_quantized = torch.quantization.quantize_dynamic(BertModel.from_pretrained("bert-base-cased").to("cpu"),
                                                         {torch.nn.Linear}, dtype=torch.qint8)

# Warm up
model_pt_quantized(**model_inputs)

# Benchmark PyTorch quantized model
time_buffer = []
for _ in trange(100):
    with track_infer_time(time_buffer):
        model_pt_quantized(**model_inputs)

results["PyTorch CPU Quantized"] = OnnxInferenceResult(time_buffer, None)

print(results)


"""
Benchmarking ONNX quantized model
"""
from transformers.convert_graph_to_onnx import quantize

# Transformers allow you to easily convert float32 model to quantize int8 with ONNX Runtime
quantized_model_path = quantize(Path("onnx/bert.opt.onnx"))

# Then you just have to load through ONNX runtime as you would normally do
quantized_model = create_model_for_provider(quantized_model_path.as_posix(), "CPUExecutionProvider")

# Warm up the overall model to have a fair comparision
outputs = quantized_model.run(None, inputs_onnx)

# Evaluate performances
time_buffer = []
for _ in trange(100, desc=f"Tracking inference time on CPUExecutionProvider with quantized model"):
    with track_infer_time(time_buffer):
        outputs = quantized_model.run(None, inputs_onnx)

# Store the result
results["ONNX CPU Quantized"] = OnnxInferenceResult(time_buffer, quantized_model_path)


"""
Show the inference performance of each providers
"""
import matplotlib.pyplot as plt
import numpy as np

# Compute average inference time + std
time_results = {k: np.mean(v.model_inference_time) * 1e3 for k, v in results.items()}
time_results_std = np.std([v.model_inference_time for v in results.values()]) * 1000

plt.rcdefaults()
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_ylabel("Avg Inference tiem (ms)")
ax.set_title("Average inference time (ms) for each provider")
ax.bar(time_results.keys(), time_results.values(), yerr=time_results_std)
plt.savefig("./average_inference_time.png")
plt.show()
