#pragma once

#include <memory>
#include <vector>

#include "taichi/codegen/compiled_kernel_data.h"
#include "taichi/codegen/cuda_c/codegen_cuda_c.h"

namespace taichi::lang {

enum class CudaCKernelKind {
  kDevice,
  kAccessorReader,
  kAccessorWriter
};

class CudaCCompiledKernelData : public CompiledKernelData {
 public:
  explicit CudaCCompiledKernelData(CudaCKernelKind kind) : kind_(kind) {
  }

  Arch arch() const override {
    return Arch::cuda_c;
  }

  std::unique_ptr<CompiledKernelData> clone() const override {
    auto ptr = std::make_unique<CudaCCompiledKernelData>(kind_);
    ptr->device_infos_ = device_infos_;
    ptr->handles_ = handles_;
    ptr->accessor_snode_ = accessor_snode_;
    return ptr;
  }

  CudaCKernelKind kind() const {
    return kind_;
  }

  void add_device_info(CudaCDeviceKernelInfo info) {
    device_infos_.push_back(std::move(info));
    handles_.emplace_back();
  }

  int num_device_kernels() const {
    return static_cast<int>(device_infos_.size());
  }

  const CudaCDeviceKernelInfo &device_info(int index) const {
    return device_infos_.at(index);
  }

  bool has_handle(int index) const {
    return handles_.at(index).has_value();
  }

  KernelLaunchHandle get_handle(int index) const {
    TI_ASSERT(handles_.at(index).has_value());
    return *handles_[index];
  }

  void set_handle(int index, const KernelLaunchHandle &handle) const {
    handles_.at(index) = handle;
  }

  void set_accessor_snode(const SNode *snode) {
    accessor_snode_ = snode;
  }

  const SNode *accessor_snode() const {
    return accessor_snode_;
  }

 protected:
  Err load_impl(const CompiledKernelDataFile &) override {
    return Err::kNotTicFile;
  }
  Err dump_impl(CompiledKernelDataFile &) const override {
    return Err::kIOStreamError;
  }

 private:
  CudaCKernelKind kind_;
  std::vector<CudaCDeviceKernelInfo> device_infos_;
  mutable std::vector<std::optional<KernelLaunchHandle>> handles_;
  const SNode *accessor_snode_{nullptr};
};

}  // namespace taichi::lang
