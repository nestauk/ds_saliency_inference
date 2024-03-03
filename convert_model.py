# use this script to convert any of the models saved to be
# compatible with tf2: https://drive.google.com/drive/folders/1GI7i6GpfI-FoklP3vCc6vxe3T9nk3V2n

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants, tag_constants

export_dir = "./app/"
# update the below line to point at the desired model downloaded
# from the above google drive link
graph_pb = "./app/model_salicon_cpu.pb"

with tf.io.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

sig = {}

builder = tf.compat.v1.saved_model.Builder(export_dir)

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def, name="")
    g = tf.compat.v1.get_default_graph()

    input = g.get_tensor_by_name("input:0")
    output = g.get_tensor_by_name("output:0")

    sig_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    sig[sig_key] = tf.compat.v1.saved_model.predict_signature_def({"input": input},
                                                                  {"output": output})
    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sig)
builder.save()