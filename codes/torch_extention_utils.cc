#include <torch/extension.h>
#include <chrono>
#include <iostream>
// #include <omp.h>

#define CHUNK 16
#define THREADS 8

// #define _OPENMP
// #include <ATen/ParallelOpenMP.h>

// Don't forget this
// #include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>
PYBIND11_MAKE_OPAQUE(std::vector<at::Tensor>);
// This function is adapted from microsoft/DeepSpeed fused_adam_frontend.cpp
void multi_tensor_adam_cuda(int chunk_size,
                            at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists,
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

/*
class MemoryBuffer:
    def __init__(self, numel, dtype):
        self.numel = numel
        self.dtype = dtype
        self.data = torch.empty(self.numel,
                                dtype=self.dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False)

    def get(self, shape, start_index):
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, \
            'requested tensor is out of the buffer range.'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor

*/

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


// This function is trying to move into C++ implementation from Python logic
// https://github.com/NVIDIA/apex/blob/0c7d8e3fa9a095a1641a2290877436d0314b69c6/apex/amp/_process_optimizer.py#L161.
// This would reduce the overhead of long loops.
void unscale_fp16_grads_into_fp32_grads(std::vector<at::Tensor>& all_fp16_params, 
                                        std::vector<at::Tensor>& all_fp32_from_fp16_params,
                                        at::Tensor is_overflow_buffer,
                                        float scale) {
    const float inv_scale = 1.0 / scale;
    TORCH_CHECK(all_fp16_params.size() == all_fp32_from_fp16_params.size(), 
                "mismatch param size between fp16_param and fp32_from_fp16_param.");
    std::vector<at::Tensor> fp16_grads_needing_unscale; 
    std::vector<at::Tensor> new_fp32_grads;
    std::vector<at::Tensor> fp16_grads_needing_unscale_with_stash;
    std::vector<at::Tensor> preexisting_fp32_grads;

    std::vector<at::Tensor> fp32_from_fp16_params;
    std::vector<size_t> offsets_mapping;
    size_t fo32_from_fp16_param_buffer_size = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < all_fp16_params.size(); ++i) {
        auto& fp16_param_grad = all_fp16_params[i].grad();
        bool fp16_param_has_grad = fp16_param_grad.defined();

        auto& fp32_from_fp16_param = all_fp32_from_fp16_params[i];
        auto& fp32_from_fp16_param_grad = fp32_from_fp16_param.grad();
        bool fp32_from_fp16_param_has_grad = fp32_from_fp16_param_grad.defined();

        if (fp16_param_has_grad && !fp32_from_fp16_param_has_grad) {
            fp32_from_fp16_params.emplace_back(fp32_from_fp16_param);
            // fp32_from_fp16_param.mutable_grad() = at::empty_like(fp32_from_fp16_param);
            fp16_grads_needing_unscale.emplace_back(fp16_param_grad);
            offsets_mapping.emplace_back(fo32_from_fp16_param_buffer_size);
            fo32_from_fp16_param_buffer_size += fp32_from_fp16_param.numel();
            // new_fp32_grads.emplace_back(fp32_from_fp16_param.grad());
        } else if (fp16_param_has_grad && fp32_from_fp16_param_has_grad) {
            fp16_grads_needing_unscale_with_stash.emplace_back(fp16_param_grad);
            preexisting_fp32_grads.emplace_back(fp32_from_fp16_param_grad);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> fs = t1 - t0;


    auto mem_buffer = MemoryBuffer(fo32_from_fp16_param_buffer_size, fp32_from_fp16_params[0]);

    auto t3 = std::chrono::high_resolution_clock::now();
    new_fp32_grads.resize(fp32_from_fp16_params.size());
    //##ifdef _OPENMP
    //##pragma omp parallel for num_threads(THREADS)
    //##endif
    // at::parallel_for(0, fp32_from_fp16_params.size(), 0, [&](int64_t start, int64_t end) {
    //     for (int64_t i = start; i < end; i++)
    //     {   
    //         // std::cout << "hi there from " << omp_get_thread_num() << ", handling i: " << i << std::endl;
    //         fp32_from_fp16_params[i].mutable_grad() = mem_buffer.Get(fp32_from_fp16_params[i], offsets_mapping[i]);
    //         //at::empty_like(fp32_from_fp16_params[i]);
    //         new_fp32_grads[i] = fp32_from_fp16_params[i].grad();
    //     }
    // });
    //    for (size_t i = 0; i < fp32_from_fp16_params.size(); ++i) {
    //       std::cout << "omp_get_thread_num( ): " << omp_get_thread_num() << std::endl;
    //        fp32_from_fp16_params[i].mutable_grad() = at::empty_like(fp32_from_fp16_params[i]);
    //        new_fp32_grads[i] = fp32_from_fp16_params[i].grad();
    //    }
       for (size_t i = 0; i < fp32_from_fp16_params.size(); ++i) {
            // std::cout << "hi there from " << omp_get_thread_num() << ", handling i: " << i << std::endl;
            fp32_from_fp16_params[i].mutable_grad() = mem_buffer.Get(fp32_from_fp16_params[i], offsets_mapping[i]);
            //at::empty_like(fp32_from_fp16_params[i]);
            new_fp32_grads[i] = fp32_from_fp16_params[i].grad();
       }

    auto t4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> fs1 = t4 - t3;
    std::cout << "latency(s): " << fs.count() << ", " << fs1.count() << std::endl;
    if (fp16_grads_needing_unscale.size() > 0) {
        std::vector<std::vector<at::Tensor>> tensor_lists;
        tensor_lists.emplace_back(fp16_grads_needing_unscale);
        tensor_lists.emplace_back(new_fp32_grads);
        multi_tensor_scale_cuda(fixed_chunk_size, is_overflow_buffer, tensor_lists, inv_scale);
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
    // py::class_<std::vector<at::Tensor>>(m, "TensorVector")
    //     .def(py::init<>())
    //     .def("push_back", [](std::vector<at::Tensor>* v, const at::Tensor& tensor_value) {
    //         v->push_back(tensor_value);
    //     })
    //     .def("__len__", [](const std::vector<at::Tensor>& v) { return v.size(); })
    //     .def("__iter__", [](const std::vector<at::Tensor>& v) {
    //         return py::make_iterator(v.cbegin(), v.cend());
    //     }, py::keep_alive<0, 1>())
    //     .def("__getitem__", [](const std::vector<<at::Tensor>& v, const size_t idx) {
    //         return v.at(idx);
    //     });

    py::bind_vector<std::vector<at::Tensor>>(m, "AAOrtValueVector");

    m.def("multi_tensor_adam",
          &multi_tensor_adam_cuda,
          "Compute and apply gradient update to parameters for Adam optimizer");
    m.def("unscale_fp16_grads_into_fp32_grads",
          &unscale_fp16_grads_into_fp32_grads,
          "Unscale those fp16 gradients into fp32 gradient buffers.");
}

