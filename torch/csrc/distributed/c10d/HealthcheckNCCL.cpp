#include <torch/csrc/distributed/c10d/HealthcheckNCCL.hpp>

#include <fmt/format.h>

#include <ATen/ATen.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/Healthcheck.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

namespace c10d {

HealthcheckNCCL::HealthcheckNCCL(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int worldSize,
    int localWorldSize,
    bool abortOnError,
    std::chrono::milliseconds interval,
    std::chrono::milliseconds timeout)
    : Healthcheck(abortOnError, interval, timeout),
      rank_(rank),
      worldSize_(worldSize),
      localWorldSize_(localWorldSize) {
  if (worldSize % localWorldSize != 0) {
    throw std::runtime_error(
        "World size must be divisible by local world size");
  }
  if (rank >= worldSize) {
    throw std::runtime_error("Rank must be less than world size");
  }
  if (worldSize / localWorldSize < 2) {
    throw std::runtime_error("At least two hosts are required");
  }

  streams_.reserve(2);
  processGroups_.reserve(2);
}

void HealthcheckNCCL::setup(int side) {
  auto hostRank = rank_ / localWorldSize_;
  auto hostCount = worldSize_ / localWorldSize_;

  auto group = (hostRank + side) % 2;
  auto groupSize = 2 * localWorldSize_;
  auto groupRank = rank_ % groupSize;

  auto storePrefix = fmt::format("/healthcheck/{}/{}", side, group);
  auto store = c10::make_intrusive<PrefixStore>(storePrefix, store_);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  streams_.emplace_back(c10::cuda::getStreamFromExternal(stream, -1));
  processGroups_.emplace_back(
      c10::make_intrusive<ProcessGroupNCCL>(store, groupRank, groupSize));
}

void HealthcheckNCCL::runHealthcheck(int side) {
  at::cuda::setCurrentCUDAStream(streams_.at(side));
  auto& pg = processGroups_.at(side);

  // TODO fix + device
  at::Tensor t = at::ones({1}, at::device(at::kCUDA).dtype(at::kFloat));
  std::vector<at::Tensor> tensors{t};

  auto work = pg->allreduce(tensors);
  work->wait(timeout_);

  if (t.item().to<double>() != 2.0 * localWorldSize_) {
    throw std::runtime_error(
        "Health check all reduce returned invalid results");
  }
}

} // namespace c10d
