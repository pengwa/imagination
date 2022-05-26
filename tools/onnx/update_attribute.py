### Be noted: this script is developed against the model exported from Megatron GPT2 Pretraining script.

import sys
import onnx
from onnx import helper, shape_inference
from onnx import TensorProto
import numpy as np
from onnx import numpy_helper

if len(sys.argv) < 2:
    print("Please give model path...")
    exit(1)

input_model_name = sys.argv[1]
output_model_name = input_model_name[:-5] + '_optimized.onnx'

model = onnx.load(input_model_name)


for node in model.graph.node:
    if node.op_type == 'AdamOptimizer':
        assert node.attribute[0].name == 'alpha'
        alpha = node.attribute[0].f
        assert node.attribute[1].name == 'beta'
        beta = node.attribute[1].f
        assert node.attribute[2].name == 'do_bias_correction'
        do_bias_correction = node.attribute[2].i
        assert node.attribute[3].name == 'epsilon'
        epsilon = node.attribute[3].f
        assert node.attribute[4].name == 'lambda'
        lambda_ = node.attribute[4].f
        assert node.attribute[5].name == 'max_norm_clip'
        max_norm_clip = node.attribute[5].i
        assert node.attribute[6].name == 'weight_decay_mode'
        weight_decay_mode = node.attribute[6].i

        del node.attribute[6]
        del node.attribute[5]
        del node.attribute[4]
        del node.attribute[3]
        del node.attribute[2]
        del node.attribute[1]
        del node.attribute[0]

        attr = node.attribute.add()
        attr.name = 'alpha'
        attr.type = 1
        attr.f = alpha

        attr = node.attribute.add()
        attr.name = 'beta'
        attr.type = 1
        attr.f = beta

        attr = node.attribute.add()
        attr.name = 'do_bias_correction'
        attr.type = 2
        attr.i = do_bias_correction

        attr = node.attribute.add()
        attr.name = 'epsilon'
        attr.type = 1
        attr.f = epsilon

        attr = node.attribute.add()
        attr.name = 'lambda'
        attr.type = 1
        attr.f = lambda_

        attr = node.attribute.add()
        attr.name = 'max_norm_clip'
        attr.type = 1
        attr.f = 0.

        attr = node.attribute.add()
        attr.name = 'weight_decay_mode'
        attr.type = 2
        attr.i = weight_decay_mode

#set opset version to 10
model.opset_import[0].version = 1

f = open(output_model_name, "wb")
f.write(model.SerializeToString())
f.close()
