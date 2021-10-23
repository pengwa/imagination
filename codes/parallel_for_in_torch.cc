// Copyright (c) Microsoft Corporation. All rights reserved.

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <vector>
#define _OPENMP
#include <ATen/ParallelOpenMP.h>

// PYBIND11_MAKE_OPAQUE(std::vector<OrtValue>);
// PYBIND11_MAKE_OPAQUE(std::vector<std::vector<at::Tensor>>);
PYBIND11_MAKE_OPAQUE(std::vector<at::Tensor>);

// This function is adapted from microsoft/DeepSpeed fused_adam_frontend.cpp
void multi_tensor_adam_cuda(int chunk_size,
                            at::Tensor noop_flag,
                            std::vector<at::Tensor>& g,
                            std::vector<at::Tensor>& p,
                            std::vector<at::Tensor>& m,
                            std::vector<at::Tensor>& v,
                            // std::vector<std::vector<at::Tensor>> tensor_lists,
                            const float lr,
                            const float beta1,
                            const float beta2,
                            const float epsilon,
                            const int step,
                            const int mode,
                            const int bias_correction,
                            const float weight_decay);

// This function is adapted from NVIDIA/apex 
// https://github.com/NVIDIA/apex/blob/0c7d8e3fa9a095a1641a2290877436d0314b69c6/csrc/amp_C_frontend.cpp#L3
void multi_tensor_scale_cuda(int chunk_size,
                             at::Tensor noop_flag,
                             std::vector<std::vector<at::Tensor>>& tensor_lists,
                             float scale);


// This function is adapted from NVIDIA/apex 
// https://github.com/NVIDIA/apex/blob/0c7d8e3fa9a095a1641a2290877436d0314b69c6/csrc/amp_C_frontend.cpp#L22
void multi_tensor_axpby_cuda(int chunk_size,
                             at::Tensor noop_flag,
                             std::vector<std::vector<at::Tensor>>& tensor_lists,
                             float a,
                             float b,
                             int arg_to_check);

const int fixed_chunk_size = 2048 * 32;

class MemoryBuffer {
    public:
        MemoryBuffer(size_t numel, at::Tensor val){
            data_buffer_ = at::empty({numel}, val.options());
        }

        at::Tensor Get(at::Tensor param, size_t start_index) {
            size_t end_index = start_index + param.numel();
            return data_buffer_.slice(0, start_index, end_index).view(param.sizes());
        }

    private:
        at::Tensor data_buffer_;
};

class MyClass {
    public:
    std::vector<at::Tensor> contents;
};

// This function is trying to move into C++ implementation from Python logic
// https://github.com/NVIDIA/apex/blob/0c7d8e3fa9a095a1641a2290877436d0314b69c6/apex/amp/_process_optimizer.py#L161.
// This would reduce the overhead of long loops.
// void unscale_fp16_grads_into_fp32_grads(MyClass fp16_cls, MyClass fp32_from_fp16_cls,
//                                         at::Tensor is_overflow_buffer,
//                                         float scale) {
void unscale_fp16_grads_into_fp32_grads(std::vector<at::Tensor>& all_fp16_params,
                                        std::vector<at::Tensor>& all_fp32_from_fp16_params,
                                        at::Tensor is_overflow_buffer,
                                        float scale) {
    const float inv_scale = 1.0 / scale;
    TORCH_CHECK(all_fp16_params.size() == all_fp32_from_fp16_params.size(), 
                "mismatch param size between fp16_param and fp32_from_fp16_param.");
    std::vector<at::Tensor> fp16_grads_needing_unscale; 
    std::vector<at::Tensor> fp16_grads_needing_unscale_with_stash;
    std::vector<at::Tensor> preexisting_fp32_grads;

    std::vector<at::Tensor> fp32_from_fp16_params;
    std::vector<size_t> offsets_mapping;
    size_t fp32_from_fp16_param_buffer_size = 0;

    for (size_t i = 0; i < all_fp16_params.size(); ++i) {
        auto& fp16_param_grad = all_fp16_params[i].grad();
        bool fp16_param_has_grad = fp16_param_grad.defined();

        auto& fp32_from_fp16_param = all_fp32_from_fp16_params[i];
        auto& fp32_from_fp16_param_grad = fp32_from_fp16_param.grad();
        bool fp32_from_fp16_param_has_grad = fp32_from_fp16_param_grad.defined();

        if (fp16_param_has_grad && !fp32_from_fp16_param_has_grad) {
            fp32_from_fp16_params.emplace_back(fp32_from_fp16_param);
            fp16_grads_needing_unscale.emplace_back(fp16_param_grad);
            offsets_mapping.emplace_back(fp32_from_fp16_param_buffer_size);
            fp32_from_fp16_param_buffer_size += fp32_from_fp16_param.numel();
        } else if (fp16_param_has_grad && fp32_from_fp16_param_has_grad) {
            fp16_grads_needing_unscale_with_stash.emplace_back(fp16_param_grad);
            preexisting_fp32_grads.emplace_back(fp32_from_fp16_param_grad);
        }
    }

    if (fp32_from_fp16_params.size() > 0) {
        auto mem_buffer = MemoryBuffer(fp32_from_fp16_param_buffer_size, fp32_from_fp16_params[0]);
        const size_t CHUNK_SIZE = 256;
        size_t chunk_num = (fp32_from_fp16_params.size() + CHUNK_SIZE - 1) / CHUNK_SIZE;

        at::parallel_for(0, chunk_num, 0, [&](int64_t start, int64_t end) {
            for (size_t j = static_cast<size_t>(start); j < static_cast<size_t>(end); j++)
            {
                size_t start_index = j * CHUNK_SIZE;
                std::vector<at::Tensor> new_fp32_grads_part;
                std::vector<at::Tensor> fp16_grads_needing_unscale_part; 
                size_t last_index = std::min(start_index + CHUNK_SIZE - 1, fp32_from_fp16_params.size() - 1);
                size_t toal_num = last_index - start_index + 1;
                new_fp32_grads_part.resize(toal_num);
                fp16_grads_needing_unscale_part.resize(toal_num);
                std::cout << "inner loop " << j << ", start_index: " << start_index << " , last_index: " << last_index << std::endl;
                for (size_t i = start_index; i <= last_index; ++i) {
                    // std::cout << "inner loop " << j << " - " << i << std::endl;
                    fp32_from_fp16_params[i].mutable_grad() = mem_buffer.Get(fp32_from_fp16_params[i], offsets_mapping[i]);
                    new_fp32_grads_part[i - start_index] = fp32_from_fp16_params[i].grad();
                    fp16_grads_needing_unscale_part[i - start_index] = fp16_grads_needing_unscale[i];
                }

                if (fp16_grads_needing_unscale_part.size() > 0) {
                    std::vector<std::vector<at::Tensor>> tensor_lists;
                    tensor_lists.emplace_back(fp16_grads_needing_unscale_part);
                    tensor_lists.emplace_back(new_fp32_grads_part);
                    multi_tensor_scale_cuda(fixed_chunk_size, is_overflow_buffer, tensor_lists, inv_scale);
                }
            }
        });
    }

    if (fp16_grads_needing_unscale_with_stash.size() > 0) {
        std::vector<std::vector<at::Tensor>> tensor_lists;
        tensor_lists.emplace_back(fp16_grads_needing_unscale_with_stash);
        tensor_lists.emplace_back(preexisting_fp32_grads);
        tensor_lists.emplace_back(preexisting_fp32_grads);
        multi_tensor_axpby_cuda(fixed_chunk_size, is_overflow_buffer, tensor_lists, inv_scale, float(1.0), 0);
    }

};




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // py::class_<std::vector<std::vector<at::Tensor>>>(m, "IntVectorVector")
    //     .def(py::init<>())
    //     .def("clear", &std::vector<std::vector<at::Tensor>>::clear)
    //     .def("pop_back", &std::vector<std::vector<at::Tensor>>::pop_back)
    //     .def("__len__", [](const std::vector<std::vector<at::Tensor>> &v) { return v.size(); })
    //     .def("__iter__", [](std::vector<std::vector<at::Tensor>> &v) {
    //         return py::make_iterator(v.begin(), v.end());
    //     }, py::keep_alive<0, 1>())
    //     .def("extend", [](std::vector<std::vector<at::Tensor>> &v, const std::vector<std::vector<at::Tensor>> &src) {
    //         v.insert(v.end(), src.begin(), src.end());
    //     })
    //     .def(py::init([](const py::iterable &it) {
    //         auto v = std::unique_ptr<std::vector<std::vector<at::Tensor>>>(new std::vector<std::vector<at::Tensor>>());
    //         v->reserve(py::len_hint(it));
    //         for (const py::iterable& hs : it) {
    //             auto w = std::vector<at::Tensor>();
    //             w.reserve(py::len_hint(hs));
    //             for (py::handle h : hs)
    //                 w.push_back(h.cast<at::Tensor>());
    //             v->push_back(w);
    //         }
    //         return v.release();
    //     }));
    // py::bind_vector<std::vector<std::vector<at::Tensor>>>(m, "IntVectorVector");
    py::class_<std::vector<at::Tensor>>(m, "IntVector")
        .def(py::init<>())
        .def("clear", &std::vector<at::Tensor>::clear)
        .def("pop_back", &std::vector<at::Tensor>::pop_back)
        .def("__len__", [](const std::vector<at::Tensor> &v) { return v.size(); })
        .def("__iter__", [](std::vector<at::Tensor> &v) {
            return py::make_iterator(v.begin(), v.end());
        }, py::keep_alive<0, 1>())
        .def("extend", [](std::vector<at::Tensor> &v, const std::vector<at::Tensor> &src) {
            v.insert(v.end(), src.begin(), src.end());
        })
        .def(py::init([](const py::iterable &it) {
            auto v = std::unique_ptr<std::vector<at::Tensor>>(new std::vector<at::Tensor>());
            v->reserve(py::len_hint(it));
            for (py::handle h : it) {
                v->push_back(h.cast<at::Tensor>());
            }
            return v.release();
        }));

    // Cannot use this because https://github.com/pybind/pybind11/issues/1470
    // py::bind_vector<std::vector<at::Tensor>>(m, "IntVector");


    py::class_<MyClass>(m, "MyClass")
        .def(py::init<>())
        .def_readwrite("contents", &MyClass::contents);
    // py::class_<std::vector<at::Tensor>>(m, "ACTensorVector")
    //     .def(py::init<>())
    //     .def("push_back", [](std::vector<at::Tensor>* v, const at::Tensor& ortvalue) {
    //       v->push_back(ortvalue);
    //     })
    //     .def("reserve", [](std::vector<at::Tensor>* v, const size_t len) { v->reserve(len); })
    //     .def("shrink_to_fit", [](std::vector<at::Tensor>* v) { v->shrink_to_fit(); })
    //     .def("__len__", [](const std::vector<at::Tensor> &v) { return v.size(); })
    //     .def("__iter__", [](const std::vector<at::Tensor> &v) {
    //       return py::make_iterator(v.cbegin(), v.cend());
    //     }, py::keep_alive<0, 1>())
    //     .def("__getitem__", [](const std::vector<at::Tensor> &v, const size_t idx) {
    //       return v.at(idx);
    //     });
    // py::bind_vector<std::vector<at::Tensor>>(m, "AAOrtValueVector");
    m.def("multi_tensor_adam",
          &multi_tensor_adam_cuda,
          "Compute and apply gradient update to parameters for Adam optimizer");
    m.def("unscale_fp16_grads_into_fp32_grads",
          &unscale_fp16_grads_into_fp32_grads,
          "Unscale those fp16 gradients into fp32 gradient buffers.");
}

