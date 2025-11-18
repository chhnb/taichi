#pragma once

#include "taichi/program/program_impl.h"
#include "taichi/ir/snode.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/common/host_memory_pool.h"

namespace taichi::lang {

class CudaCProgramImpl : public ProgramImpl {
 public:
  explicit CudaCProgramImpl(CompileConfig &config);
  ~CudaCProgramImpl() override;

  void materialize_runtime(KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

  void materialize_snode_tree(SNodeTree *tree,
                              uint64 *result_buffer_ptr) override;

  void destroy_snode_tree(SNodeTree *snode_tree) override;

  std::size_t get_snode_num_dynamically_allocated(
      SNode *snode,
      uint64 *result_buffer) override;

  void synchronize() override;

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder(
      const DeviceCapabilityConfig &caps) override;

  uint64 *result_buffer() const {
    return result_buffer_;
  }

  struct FieldStorage {
    void *device_ptr{nullptr};
    size_t bytes{0};
    int64 num_elements{0};
    DataType dtype{PrimitiveType::unknown};
  };

  const FieldStorage *lookup_field(const SNode *snode) const;

  KernelProfilerBase *profiler() const {
    return profiler_;
  }

  void *temp_buffer() const {
    return temp_buffer_;
  }

  std::pair<const StructType *, size_t> get_struct_type_with_data_layout(
      const StructType *old_ty,
      const std::string &layout) override;

 protected:
  std::unique_ptr<KernelCompiler> make_kernel_compiler() override;

  std::unique_ptr<KernelLauncher> make_kernel_launcher() override;

 private:
  void allocate_field(SNode *snode);
  void free_field(SNode *snode);

  KernelProfilerBase *profiler_{nullptr};
  uint64 *result_buffer_{nullptr};
  void *temp_buffer_{nullptr};
  std::unordered_map<const SNode *, FieldStorage> field_storage_;
};

}  // namespace taichi::lang
