"""Xuất efficientnet.keras sang ONNX. Đóng băng biến trước để không sót hằng số chuẩn hóa.

Chỉ cần lúc xuất, không phải dependency của app; chạy trong môi trường riêng để không đụng venv:
py -3.12 -m venv E:\\_ox
E:\\_ox\\Scripts\\python.exe -m pip install tensorflow==2.20.0 keras==3.13.2 onnx onnxruntime numpy==1.26.4
E:\\_ox\\Scripts\\python.exe -m pip install --no-deps tf2onnx==1.16.1
E:\\_ox\\Scripts\\python.exe scripts/emotion/export_onnx.py models/emotion/efficientnet.keras models/emotion/efficientnet.onnx
Sau đó chạy scripts/emotion/check_onnx.py trong venv của dự án để so số và ghi file khớp.
"""
import sys
import keras
import tensorflow as tf
import tf2onnx
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

src, dst = sys.argv[1:3]
model = keras.models.load_model(src, compile=False)
function = tf.function(lambda images: model(images, training=False))
concrete = function.get_concrete_function(tf.TensorSpec((None, 100, 100, 3), tf.float32, name='images'))
frozen = convert_variables_to_constants_v2(concrete)
graph = frozen.graph.as_graph_def()
proto, _ = tf2onnx.convert.from_graph_def(graph, input_names=[frozen.inputs[0].name],
                                          output_names=[frozen.outputs[0].name], opset=17, output_path=dst)
print('ok', [i.name for i in proto.graph.input], [o.name for o in proto.graph.output])
