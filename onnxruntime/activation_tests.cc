// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/compare_provider_test_utils.h"

namespace onnxruntime {
namespace test {

#if USE_CUDA
constexpr const char* kGpuExecutionProvider = kCudaExecutionProvider;
#elif USE_ROCM
constexpr const char* kGpuExecutionProvider = kRocmExecutionProvider;
#endif

static void TestActivations(const std::vector<int64_t>& tensor_dim,
                            const std::string& operator_name,
                            bool is_grad_op,
                            double per_sample_tolerance = 2e-4,
                            double relative_per_sample_tolerance = 2e-4) {
  CompareOpTester test(operator_name.c_str(), 1, onnxruntime::kMSDomain);

  // create rand inputs
  RandomValueGenerator random{};
  if (is_grad_op) {
    std::vector<float> dY_data = random.Uniform<float>(tensor_dim, -10.0f, 10.0f);
    test.AddInput<float>("dY", tensor_dim, dY_data);
  }
  std::vector<float> X_data = random.Uniform<float>(tensor_dim, -10.0f, 10.0f);
  test.AddInput<float>("X", tensor_dim, X_data);

  // create output tensors
  if (is_grad_op) {
    std::vector<float> dX_data = FillZeros<float>(tensor_dim);
    test.AddOutput<float>("dX", tensor_dim, dX_data);
  } else {
    std::vector<float> Y_data = FillZeros<float>(tensor_dim);
    test.AddOutput<float>("Y", tensor_dim, Y_data);
  }

  test.CompareWithCPU(kGpuExecutionProvider, per_sample_tolerance, relative_per_sample_tolerance);
}

TEST(CudaKernelTest, Gelu_basic) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    // bump up the tolerance due to the nature of gelu op accumulation computation complexity
    TestActivations(test_dim, "Gelu", false /* grad_op */, 1e-3, 1e-3);
  }
}

TEST(CudaKernelTest, FastGelu_basic) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    TestActivations(test_dim, "FastGelu", false /* grad_op */);
  }
}

TEST(CudaKernelTest, GeluGrad_basic) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    TestActivations(test_dim, "GeluGrad", true /* grad_op */);
  }
}

TEST(CudaKernelTest, FastGeluGrad_basic) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    TestActivations(test_dim, "FastGeluGrad", true /* grad_op */);
  }
}

TEST(CudaKernelTest, ReluGrad_basic) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    TestActivations(test_dim, "ReluGrad", true /* grad_op */);
  }
}

TEST(CudaKernelTest, SigmoidGrad_basic) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    TestActivations(test_dim, "SigmoidGrad", true /* grad_op */);
  }
}

TEST(CudaKernelTest, TanhGrad_basic) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    TestActivations(test_dim, "TanhGrad", true /* grad_op */);
  }
}

static void TestActivationsWithBroadcastBias(
    const std::vector<int64_t>& tensor_dim,
    const std::string& operator_name,
    bool is_grad_op,
    const bool test_fp16 = false,
    double per_sample_tolerance = 1e-1,
    double relative_per_sample_tolerance = 1e-1) {
  ORT_ENFORCE(tensor_dim.size() >= 1);
  const std::vector<int64_t> bias_dim(tensor_dim.end() - 1, tensor_dim.end());

  CompareOpTester test(operator_name.c_str(), 1, onnxruntime::kMSDomain);

  // create rand inputs
  RandomValueGenerator random{};
  if (is_grad_op) {
    std::vector<float> dY_data = random.Uniform<float>(tensor_dim, -10.0f, 10.0f);
    if (test_fp16) {
      std::vector<MLFloat16> dY_data_half(dY_data.size());
      ConvertFloatToMLFloat16(dY_data.data(), dY_data_half.data(), int(dY_data.size()));
      test.AddInput<MLFloat16>("dY", tensor_dim, dY_data_half);
    } else {
      test.AddInput<float>("dY", tensor_dim, dY_data);
    }
  }

  std::vector<float> X_data = random.Uniform<float>(tensor_dim, -10.0f, 10.0f);
  if (test_fp16) {
    std::vector<MLFloat16> X_data_half(X_data.size());
    ConvertFloatToMLFloat16(X_data.data(), X_data_half.data(), int(X_data.size()));
    test.AddInput<MLFloat16>("X", tensor_dim, X_data_half);
  } else {
    test.AddInput<float>("X", tensor_dim, X_data);
  }

  std::vector<float> B_data = random.Uniform<float>(bias_dim, -10.0f, 10.0f);
  if (test_fp16) {
    std::vector<MLFloat16> B_data_half(B_data.size());
    ConvertFloatToMLFloat16(B_data.data(), B_data_half.data(), int(B_data.size()));
    test.AddInput<MLFloat16>("B", bias_dim, B_data_half);
  } else {
    test.AddInput<float>("B", bias_dim, B_data);
  }

  // create output tensors
  if (is_grad_op) {
    std::vector<float> dX_data = FillZeros<float>(tensor_dim);
    if (test_fp16) {
      std::vector<MLFloat16> dX_data_half(dX_data.size());
      ConvertFloatToMLFloat16(dX_data.data(), dX_data_half.data(), int(dX_data.size()));
      test.AddOutput<MLFloat16>("dX", tensor_dim, dX_data_half);
    } else {
      test.AddOutput<float>("dX", tensor_dim, dX_data);
    }
  } else {
    std::vector<float> Y_data = FillZeros<float>(tensor_dim);
    if (test_fp16) {
      std::vector<MLFloat16> Y_data_half(Y_data.size());
      ConvertFloatToMLFloat16(Y_data.data(), Y_data_half.data(), int(Y_data.size()));
      test.AddOutput<MLFloat16>("Y", tensor_dim, Y_data_half);
    } else {
      test.AddOutput<float>("Y", tensor_dim, Y_data);
    }
  }

  test.CompareWithCPU(kGpuExecutionProvider, per_sample_tolerance, relative_per_sample_tolerance);
}

TEST(CudaKernelTest, FastGelu_bias) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    TestActivationsWithBroadcastBias(test_dim, "FastGelu", false);
  }
}

TEST(CudaKernelTest, BiasGeluGradDx_basic) {
  std::vector<std::vector<int64_t>> test_dims{
      {1},
      {2},
      {3},
      {4},
      {8},
      {16},
      {32},
      {64},
      {128},
      {256},
      {512},
      {16, 1},
      {16, 2},
      {16, 3},
      {16, 4},
      {16, 8},
      {16, 16},
      {16, 32},
      {16, 64},
      {16, 128},
      {16, 192},
      {16, 256},
      {16, 258},
      {8, 2, 128, 1},
      {8, 2, 128, 2},
      {8, 2, 128, 4},
      {8, 2, 128, 8},
      {8, 2, 128, 16},
      {9, 2, 128, 32},
      {8, 2, 128, 64},
      {9, 2, 128, 128},
      {16, 128, 6144},
      {16, 127, 6144},
      {16, 128, 6143},
      {16, 3, 224, 224},
      {15, 3, 223, 223},
      // multiplier of the initial 3 dims > 65535
      // {128, 3, 224, 2},
      // // {128, 3, 224, 3},
      // {128, 3, 224, 128},
      //{128, 3, 224, 223},
      //{128, 3, 224, 224},
      //{128, 3, 224, 6143},
      //{128, 3, 224, 6144},
  };

  for (const auto& test_dim : test_dims) {
    TestActivationsWithBroadcastBias(test_dim, "BiasGeluGrad_dX", true, false);
  }

  for (const auto& test_dim : test_dims) {
    TestActivationsWithBroadcastBias(test_dim, "BiasGeluGrad_dX", true, true);
  }
}

TEST(CudaKernelTest, BiasFastGeluGradDx_basic) {
  std::vector<std::vector<int64_t>> test_dims{
      {1},
      {2},
      {3},
      {4},
      {8},
      {16},
      {32},
      {64},
      {128},
      {256},
      {512},
      {16, 1},
      {16, 2},
      {16, 3},
      {16, 4},
      {16, 8},
      {16, 16},
      {16, 32},
      {16, 64},
      {16, 128},
      {16, 192},
      {16, 256},
      {16, 258},
      {8, 2, 128, 1},
      {8, 2, 128, 2},
      {8, 2, 128, 4},
      {8, 2, 128, 8},
      {8, 2, 128, 16},
      {9, 2, 128, 32},
      {8, 2, 128, 64},
      {9, 2, 128, 128},
      {16, 128, 6144},
      {16, 127, 6144},
      {16, 128, 6143},
      {16, 3, 224, 224},
      {15, 3, 223, 223},
      // multiplier of the initial 3 dims > 65535
      // {128, 3, 224, 2},
      // // {128, 3, 224, 3},
      // {128, 3, 224, 128},
  };
  for (const auto& test_dim : test_dims) {
    TestActivationsWithBroadcastBias(test_dim, "BiasFastGeluGrad_dX", true, false);
  }

  for (const auto& test_dim : test_dims) {
    TestActivationsWithBroadcastBias(test_dim, "BiasFastGeluGrad_dX", true, true);
  }
}

}  // namespace test
}  // namespace onnxruntime
