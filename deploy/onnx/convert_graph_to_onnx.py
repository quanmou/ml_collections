from pathlib import Path
from transformers.convert_graph_to_onnx import convert

# Handles all the above steps for you
convert(framework="pt", model="bert-base-cased", output=Path("onnx/bert-base-cased.onnx"), opset=11)

# Tensorflow
# convert(framework="tf", model="bert-base-cased", output="onnx/bert-base-cased.onnx", opset=11)
