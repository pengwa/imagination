# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Import external libraries.
from functools import wraps
import onnxruntime
import pytest
import torch
from torch.nn.parameter import Parameter

# Import ORT modules.
from _test_helpers import *
from onnxruntime.training.ortmodule import ORTModule

import argparse
from typing import Text

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int)
parser.add_argument("--hidden", type=int)
parser.add_argument("--layer", type=int)
parser.add_argument('--ort', action='store_true')
parser.add_argument("--tag", type=str)

args= parser.parse_args()

torch.manual_seed(1)
onnxruntime.set_seed(1)

def test_CustomFunctionOverheadTest(b, h, run_with_ort):
    class CustomFunctionOverheadTestFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight):
            with nvtx.annotate("forward", color="green"):
                # val = input.clamp(min=0)
                #new_val=weight
                #for i in range(1):
                new_val = torch.matmul(input, weight)
                ctx.save_for_backward(input)
                return new_val

        @staticmethod
        def backward(ctx, grad_output):
            with nvtx.annotate("backward", color="green"):
                input, = ctx.saved_tensors
                return input * grad_output, torch.matmul(torch.transpose(input, 1, 0).contiguous(), grad_output)

    class CustomFunctionDelegation(torch.nn.Module):
        def __init__(self, output_size):
            super(CustomFunctionDelegation, self).__init__()
            self._p = Parameter(torch.empty(
                (output_size, output_size),
                device=torch.cuda.current_device(),
                dtype=torch.float))

            with torch.no_grad():
                self._p.uniform_()
            self.custom_func = CustomFunctionOverheadTestFunction.apply

        def forward(self, x):
            return self.custom_func(x, self._p)

    class CustomFunctionOverheadTestModel(torch.nn.Module):
        def __init__(self, output_size):
            super(CustomFunctionOverheadTestModel, self).__init__()
            self._layer_count = args.layer
            self._layers = torch.nn.ModuleList(
                [CustomFunctionDelegation(output_size) for i in range(self._layer_count)]
                )


        def forward(self, x):
            for index, val in enumerate(self._layers):
                x = self._layers[index](x)
            return x

    output_size = h
    batch_size = b
    def model_builder():
        return CustomFunctionOverheadTestModel(output_size)

    def input_generator():
        return torch.randn(batch_size, output_size, dtype=torch.float) #.requires_grad_()

    # generate a label that have same shape as forward output.
    label_input = torch.ones([batch_size * output_size]).reshape(batch_size, output_size).contiguous()

    def cuda_barrier_func():
        torch.cuda.synchronize()
    cuda = torch.device('cuda:0')

    for i in range(1):
        m = model_builder()
        x = input_generator()

        if run_with_ort:
            #with nvtx.annotate("run_with_ort_on_device", color="red"):
            outputs_ort, grads_ort, start, end = run_with_ort_on_device(
                cuda, m, [x], label_input)
            cuda_barrier_func()
            print(args.tag, ", ort run elapsed time (ms): ", start.elapsed_time(end))
        else:
            #with nvtx.annotate("run_with_pytorch_on_device", color="red"):
            outputs, grads, start, end = run_with_pytorch_on_device(
                cuda, m, [x], label_input)
            cuda_barrier_func()
            print(args.tag, ", pt run elapsed time (ms): ", start.elapsed_time(end))


test_CustomFunctionOverheadTest(args.batch, args.hidden, args.ort)
