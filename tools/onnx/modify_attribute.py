import onnx
from onnx import helper, numpy_helper
import numpy as np
import copy

exported_model = onnx.load("source.onnx")

for node in exported_model.graph.node:
    if node.op_type in ["RandomUniformLike"]:
        attributes = list(node.attribute)
        new_attributes = []
        for attr in attributes:
            if attr.name == "seed":
                new_attributes.append(helper.make_attribute('seed', float(np.random.randint(1e6))))
            else:
                new_attributes.append(copy.deepcopy(attr))

        del node.attribute[:]
        node.attribute.extend(new_attributes)


onnx.save(exported_model, "modified.onnx")
