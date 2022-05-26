# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Import external libraries.
import onnxruntime
import pytest
import torch
from torch.nn.parameter import Parameter
from distutils.version import LooseVersion

# Import ORT modules.
from _test_helpers import *
from onnxruntime.training.ortmodule import ORTModule

torch.manual_seed(1)
onnxruntime.set_seed(1)

from onnxruntime.capi._pybind_state import save_checkpoint


trainable_weight_names = [
    "bert.encoder.layer.2.output.LayerNorm.weight",
    "bert.encoder.layer.2.output.LayerNorm.bias",
    "add1_initializerr",
    "cls.predictions.transform.LayerNorm.weight",
    "cls.predictions.transform.LayerNorm.bias",
    "bert.embeddings.word_embeddings.weight_transposed",
    "cls.predictions.bias",
]

import onnx
onnx_model = onnx.load("/bert_ort/pengwa/adhoc/onnxruntime/test/testdata/transform/computation_reduction/e2e.onnx")

trainable_param_tensor_protos = []
non_trainable_param_tensor_protos = []
for i in onnx_model.graph.initializer:
    if i.name in trainable_weight_names:
        trainable_param_tensor_protos.append(i.SerializeToString())
    else:
        non_trainable_param_tensor_protos.append(i.SerializeToString())

save_checkpoint(trainable_param_tensor_protos, non_trainable_param_tensor_protos, "/tmp/e2e_ckpt_new")


# save_checkpoint("/bert_ort/pengwa/adhoc/onnxruntime/test/testdata/transform/computation_reduction/e2e.onnx", trainable_weight_names, "/tmp/e2e_ckpt")
