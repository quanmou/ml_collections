# # An optional step unless
# # you want to get a model with mixed precision for perf accelartion on newer GPU
# # or you are working with Tensorflow(tf.keras) models or pytorch models other than bert

# !pip install onnxruntime-tools
# from onnxruntime_tools import optimizer

# # Mixed precision conversion for bert-base-cased model converted from Pytorch
# optimized_model = optimizer.optimize_model("bert-base-cased.onnx", model_type='bert', num_heads=12, hidden_size=768)
# optimized_model.convert_model_float32_to_float16()
# optimized_model.save_model_to_file("bert-base-cased.onnx")

# # optimizations for bert-base-cased model converted from Tensorflow(tf.keras)
# optimized_model = optimizer.optimize_model("bert-base-cased.onnx", model_type='bert_keras', num_heads=12, hidden_size=768)
# optimized_model.save_model_to_file("bert-base-cased.onnx")


# optimize transformer-based models with onnxruntime-tools
from onnxruntime_tools import optimizer
from onnxruntime_tools.transformers.onnx_model_bert import BertOptimizationOptions

# disable embedding layer norm optimization for better model size reduction
opt_options = BertOptimizationOptions('bert')
opt_options.enable_embed_layer_norm = False

opt_model = optimizer.optimize_model(
    'onnx/bert-base-cased.onnx',
    'bert',
    num_heads=12,
    hidden_size=768,
    optimization_options=opt_options)
opt_model.save_model_to_file('onnx/bert.opt.onnx')
