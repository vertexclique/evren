# -*- coding: utf-8 -*-

import tensorflow as tf
from keras import backend as K
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


def get_output_node_names():
    return [n.name for n in tf.get_default_graph().as_graph_def().node]


def export_keras(model_name, input_names, output_names, with_weights=True):
    out_name = ','.join(output_names)
    saver = tf.train.Saver()
    tf.train.write_graph(K.get_session().graph_def, 'out',
                         model_name + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + model_name + '.chkp')

    output_graph = None
    if with_weights:
        freeze_graph.freeze_graph(
            input_graph='out/' + model_name + '_graph.pbtxt',
            input_saver=None,
            input_binary=False,
            input_checkpoint='out/' + model_name + '.chkp',
            output_node_names=out_name,
            restore_op_name="save/restore_all",
            filename_tensor_name="save/Const:0",
            output_graph='out/frozen_' + model_name + '.pb',
            clear_devices=True,
            initializer_nodes="")

        input_graph_def = tf.GraphDef()
        with tf.gfile.Open('out/frozen_' + model_name + '.pb', "rb") as f:
            input_graph_def.ParseFromString(f.read())

        output_graph = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_names, output_names,
            tf.float32.as_datatype_enum)
    else:
        output_graph = tf.train.write_graph(K.get_session().graph,
                             'out/' + model_name, model_name + 'wo_weights.pb')

    return output_graph


def export_keras_to_file(model_name, input_names, output_names, out_file_name, with_weights=True):
    exported_model = export_keras(model_name, input_names, output_names, with_weights)

    with tf.gfile.FastGFile('out/' + out_file_name + '.pb', "wb") as f:
        f.write(exported_model.SerializeToString())

    return exported_model
