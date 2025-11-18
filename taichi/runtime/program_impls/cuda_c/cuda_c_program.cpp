#include "taichi/runtime/program_impls/cuda_c/cuda_c_program.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>

#include "taichi/codegen/cuda_c/cuda_c_compiled_data.h"
#include "taichi/codegen/cuda_c/cuda_c_kernel_compiler.h"
#include "taichi/common/core.h"
#include "taichi/ir/type_utils.h"
#include "taichi/inc/constants.h"
#include "taichi/program/kernel.h"
#include "taichi/program/launch_context_builder.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"

namespace taichi::lang {

namespace {

using KernelHandle = KernelLaunchHandle;

size_t packed_type_size(const Type *type);
std::pair<const StructType *, size_t> pack_struct_type(
    const StructType *old_ty);

size_t packed_type_size(const Type *type) {
  if (auto primitive = type->cast<PrimitiveType>()) {
    return data_type_size(primitive);
  }
  if (auto tensor = type->cast<TensorType>()) {
    return data_type_size(tensor);
  }
  if (auto pointer = type->cast<PointerType>()) {
    return sizeof(uint64);
  }
  if (auto struct_type = type->cast<StructType>()) {
    return pack_struct_type(struct_type).second;
  }
  if (auto argpack = type->cast<ArgPackType>()) {
    auto *struct_type_old = TypeFactory::get_instance()
                                .get_struct_type(argpack->elements(),
                                                 argpack->get_layout())
                                ->as<StructType>();
    return pack_struct_type(struct_type_old).second;
  }
  TI_ERROR("Unsupported type {} in cuda_c layout.", type->to_string());
}

std::pair<const StructType *, size_t> pack_struct_type(
    const StructType *old_ty) {
  auto members = old_ty->elements();
  size_t offset = 0;
  for (auto &member : members) {
    size_t member_size = 0;
    if (auto struct_type = member.type->cast<StructType>()) {
      auto [new_ty, size] = pack_struct_type(struct_type);
      member.type = new_ty;
      member_size = size;
    } else if (auto argpack_type = member.type->cast<ArgPackType>()) {
      auto *struct_type_old = TypeFactory::get_instance()
                                  .get_struct_type(argpack_type->elements(),
                                                   argpack_type->get_layout())
                                  ->as<StructType>();
      auto [packed, size] = pack_struct_type(struct_type_old);
      member.type = TypeFactory::get_instance()
                        .get_argpack_type(packed->elements(),
                                          packed->get_layout());
      member_size = size;
    } else {
      member_size = packed_type_size(member.type);
    }
    member.offset = offset;
    offset += member_size;
  }
  auto *new_ty =
      TypeFactory::get_instance()
          .get_struct_type(members, old_ty->get_layout())
          ->as<StructType>();
  return {new_ty, offset};
}

size_t data_type_bytes(DataType dt) {
  return data_type_size(dt);
}

class CudaCKernelLauncher : public KernelLauncher {
 public:
  struct Config {
    CudaCProgramImpl *program{nullptr};
  };

  explicit CudaCKernelLauncher(Config config) : config_(config) {
  }

  ~CudaCKernelLauncher() override = default;

  void launch_kernel(const CompiledKernelData &compiled_kernel_data,
                     LaunchContextBuilder &ctx) override {
    auto *data =
        dynamic_cast<const CudaCCompiledKernelData *>(&compiled_kernel_data);
    TI_ASSERT(data);

    if (data->kind() == CudaCKernelKind::kAccessorReader) {
      launch_accessor_reader(*data, ctx);
      return;
    }
    if (data->kind() == CudaCKernelKind::kAccessorWriter) {
      launch_accessor_writer(*data, ctx);
      return;
    }
    launch_device_kernel(*data, ctx);
  }

 private:
struct DeviceKernelInstance {
  void *module{nullptr};
  void *function{nullptr};
  CudaCDeviceKernelInfo info;
};

  void launch_accessor_writer(const CudaCCompiledKernelData &data,
                              LaunchContextBuilder &ctx) {
    const auto *snode = data.accessor_snode();
    auto storage = config_.program->lookup_field(snode);
    TI_ASSERT(storage);
    int32_t linear_idx = 0;
    if (snode->num_active_indices > 0) {
      linear_idx = ctx.get_struct_arg<int32_t>({0});
    }
    size_t offset =
        static_cast<size_t>(linear_idx) * data_type_bytes(storage->dtype);
    auto val = ctx.get_struct_arg<float>({snode->num_active_indices});
    CUDADriver::get_instance().memcpy_host_to_device(
        static_cast<char *>(storage->device_ptr) + offset, &val,
        data_type_bytes(storage->dtype));
  }

  void launch_accessor_reader(const CudaCCompiledKernelData &data,
                              LaunchContextBuilder &ctx) {
    const auto *snode = data.accessor_snode();
    auto storage = config_.program->lookup_field(snode);
    TI_ASSERT(storage);
    int32_t linear_idx = 0;
    if (snode->num_active_indices > 0) {
      linear_idx = ctx.get_struct_arg<int32_t>({0});
    }
    size_t offset =
        static_cast<size_t>(linear_idx) * data_type_bytes(storage->dtype);
    float value;
    CUDADriver::get_instance().memcpy_device_to_host(
        &value, static_cast<char *>(storage->device_ptr) + offset,
        sizeof(float));
    uint64 bits = 0;
    std::memcpy(&bits, &value, sizeof(value));
    ctx.get_context().result_buffer[0] = bits;
  }

  void launch_device_kernel(const CudaCCompiledKernelData &data,
                            LaunchContextBuilder &ctx) {
    std::unordered_map<std::vector<int>, void *, hashing::Hasher<std::vector<int>>>
        external_arrays;
    std::unordered_map<std::vector<int>, size_t, hashing::Hasher<std::vector<int>>>
        external_array_sizes;

    for (const auto &kv : ctx.array_runtime_sizes) {
      const auto &arg_key = kv.first;
      uint64 arr_sz = kv.second;
      if (arr_sz == 0) {
        continue;
      }
      auto it = ctx.device_allocation_type.find(arg_key);
      LaunchContextBuilder::DevAllocType dev_type =
          (it == ctx.device_allocation_type.end()
               ? LaunchContextBuilder::DevAllocType::kNone
               : it->second);
      if (dev_type != LaunchContextBuilder::DevAllocType::kNone) {
        TI_WARN("Ndarray arguments are not fully supported on cuda_c backend");
        continue;
      }
      auto data_idx = arg_key;
      data_idx.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
      auto host_ptr_it = ctx.array_ptrs.find(data_idx);
      TI_ASSERT(host_ptr_it != ctx.array_ptrs.end());
      void *device_ptr = nullptr;
      CUDADriver::get_instance().malloc(&device_ptr, arr_sz);
      CUDADriver::get_instance().memcpy_host_to_device(device_ptr,
                                                       host_ptr_it->second,
                                                       arr_sz);
      ctx.set_ndarray_ptrs(arg_key, (uint64)device_ptr, 0);
      external_arrays[data_idx] = device_ptr;
      external_array_sizes[data_idx] = arr_sz;
    }

    for (int kernel_idx = 0; kernel_idx < data.num_device_kernels();
         ++kernel_idx) {
      auto &info = data.device_info(kernel_idx);
      KernelHandle handle;
      if (data.has_handle(kernel_idx)) {
        handle = data.get_handle(kernel_idx);
      } else {
        auto new_handle = register_kernel(info);
        data.set_handle(kernel_idx, new_handle);
        handle = new_handle;
      }
      const auto &instance = device_kernels_.at(handle.get_launch_id());
      std::vector<void *> arg_ptrs;
      std::vector<std::unique_ptr<std::vector<uint8_t>>> arg_buffers;

      void *device_result_buffer = nullptr;
      char *host_result_buffer = nullptr;
      if (info.has_result_buffer) {
        TI_ASSERT(ctx.result_buffer_size > 0);
        host_result_buffer =
            reinterpret_cast<char *>(ctx.get_context().result_buffer);
        CUDADriver::get_instance().malloc(&device_result_buffer,
                                          ctx.result_buffer_size);
        CUDADriver::get_instance().memset(device_result_buffer, 0,
                                          ctx.result_buffer_size);
      }
      auto append_pointer_arg = [&](void *ptr) {
        auto buf = std::make_unique<std::vector<uint8_t>>(sizeof(void *));
        std::memcpy(buf->data(), &ptr, sizeof(void *));
        arg_ptrs.push_back(buf->data());
        arg_buffers.push_back(std::move(buf));
      };
      auto append_scalar_arg = [&](const CudaCArgInfo &arg) {
        if (arg.dtype->is_primitive(PrimitiveTypeID::i32)) {
          auto value = ctx.get_struct_arg<int32_t>(arg.arg_id);
          auto buf =
              std::make_unique<std::vector<uint8_t>>(sizeof(int32_t));
          std::memcpy(buf->data(), &value, sizeof(int32_t));
          arg_ptrs.push_back(buf->data());
          arg_buffers.push_back(std::move(buf));
        } else if (arg.dtype->is_primitive(PrimitiveTypeID::f32)) {
          auto value = ctx.get_struct_arg<float>(arg.arg_id);
          auto buf = std::make_unique<std::vector<uint8_t>>(sizeof(float));
          std::memcpy(buf->data(), &value, sizeof(float));
          arg_ptrs.push_back(buf->data());
          arg_buffers.push_back(std::move(buf));
        } else if (arg.dtype->is_primitive(PrimitiveTypeID::u64)) {
          auto value = ctx.get_struct_arg<uint64_t>(arg.arg_id);
          auto buf =
              std::make_unique<std::vector<uint8_t>>(sizeof(uint64_t));
          std::memcpy(buf->data(), &value, sizeof(uint64_t));
          arg_ptrs.push_back(buf->data());
          arg_buffers.push_back(std::move(buf));
        } else {
          TI_ERROR("Unsupported arg dtype");
        }
      };

      for (const auto &param : instance.info.params) {
        switch (param.kind) {
          case CudaCKernelParamKind::kSNode: {
            TI_ASSERT(param.index >= 0 &&
                      param.index < instance.info.snodes.size());
            auto *snode = instance.info.snodes[param.index];
            auto storage = config_.program->lookup_field(snode);
            TI_ASSERT(storage);
            append_pointer_arg(storage->device_ptr);
            break;
          }
          case CudaCKernelParamKind::kArg: {
            TI_ASSERT(param.index >= 0 &&
                      param.index < instance.info.arg_infos.size());
            append_scalar_arg(instance.info.arg_infos[param.index]);
            break;
          }
          case CudaCKernelParamKind::kResultBuffer: {
            TI_ASSERT(info.has_result_buffer);
            append_pointer_arg(device_result_buffer);
            break;
          }
          case CudaCKernelParamKind::kTempBuffer: {
            TI_ASSERT(info.has_temp_buffer);
            auto *temp_buffer = config_.program->temp_buffer();
            TI_ASSERT(temp_buffer);
            append_pointer_arg(temp_buffer);
            break;
          }
        }
      }
      CUDAContext::get_instance().launch(
          instance.function, instance.info.kernel_name, arg_ptrs,
          std::vector<int>(arg_ptrs.size(), 0), instance.info.grid_dim,
          instance.info.block_dim, /*shared_mem=*/0);
      if (device_result_buffer) {
        CUDADriver::get_instance().memcpy_device_to_host(
            host_result_buffer, device_result_buffer, ctx.result_buffer_size);
        CUDADriver::get_instance().mem_free(device_result_buffer);
      }
    }

    for (const auto &kv : external_arrays) {
      auto host_ptr = ctx.array_ptrs.at(kv.first);
      auto size = external_array_sizes.at(kv.first);
      CUDADriver::get_instance().memcpy_device_to_host(host_ptr, kv.second,
                                                       size);
      CUDADriver::get_instance().mem_free(kv.second);
    }
  }

  KernelHandle register_kernel(const CudaCDeviceKernelInfo &info) const {
    auto &driver = CUDADriver::get_instance();
    void *module = nullptr;
    driver.module_load_data_ex(&module, info.cubin.data(), 0, nullptr, nullptr);
    void *function = nullptr;
    driver.module_get_function(&function, module, info.kernel_name.c_str());
    KernelHandle handle;
    handle.set_launch_id(next_id_++);
    DeviceKernelInstance inst;
    inst.module = module;
    inst.function = function;
    inst.info = info;
    device_kernels_.emplace(handle.get_launch_id(), std::move(inst));
    return handle;
  }

  Config config_;
  mutable std::unordered_map<int, DeviceKernelInstance> device_kernels_;
  mutable int next_id_{0};
};

}  // namespace

CudaCProgramImpl::CudaCProgramImpl(CompileConfig &config)
    : ProgramImpl(config) {
  config.use_llvm = false;
  config.real_matrix_scalarize = true;
  config.force_scalarize_matrix = true;
  auto &driver = CUDADriver::get_instance();
  int version = 0;
  driver.driver_get_version(&version);
  int query_max_block_dim = 1024;
  driver.device_get_attribute(&query_max_block_dim,
                              CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, 0);
  if (config.max_block_dim == 0) {
    config.max_block_dim = query_max_block_dim;
  }
  int num_sms = 1;
  driver.device_get_attribute(&num_sms,
                              CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0);
  int query_max_block_per_sm = 16;
  if (version >= 11000) {
    driver.device_get_attribute(&query_max_block_per_sm,
                                CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, 0);
  }
  if (config.saturating_grid_dim == 0) {
    config.saturating_grid_dim = std::max(
        1, num_sms * query_max_block_per_sm * 2);
  }
}

CudaCProgramImpl::~CudaCProgramImpl() {
  if (temp_buffer_) {
    CUDADriver::get_instance().mem_free(temp_buffer_);
    temp_buffer_ = nullptr;
  }
}

void CudaCProgramImpl::materialize_runtime(KernelProfilerBase *profiler,
                                           uint64 **result_buffer_ptr) {
  profiler_ = profiler;
  CUDAContext::get_instance().set_profiler(profiler);
  result_buffer_ = (uint64 *)HostMemoryPool::get_instance().allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);
  if (!temp_buffer_) {
    CUDADriver::get_instance().malloc(
        &temp_buffer_, taichi_global_tmp_buffer_size);
    CUDADriver::get_instance().memset(temp_buffer_, 0,
                                      taichi_global_tmp_buffer_size);
  }
  *result_buffer_ptr = result_buffer_;
}

const CudaCProgramImpl::FieldStorage *CudaCProgramImpl::lookup_field(
    const SNode *snode) const {
  auto it = field_storage_.find(snode);
  if (it == field_storage_.end()) {
    return nullptr;
  }
  return &it->second;
}

void CudaCProgramImpl::materialize_snode_tree(SNodeTree *tree,
                                              uint64 *result_buffer_ptr) {
  tree->root()->set_snode_tree_id(tree->id());
  auto collect = [&](auto &&self, SNode *node) -> void {
    if (node->type == SNodeType::place) {
      allocate_field(node);
    }
    for (auto &ch : node->ch) {
      self(self, ch.get());
    }
  };
  collect(collect, tree->root());
}

void CudaCProgramImpl::destroy_snode_tree(SNodeTree *snode_tree) {
  auto collect = [&](auto &&self, SNode *node) -> void {
    if (node->type == SNodeType::place) {
      free_field(node);
    }
    for (auto &ch : node->ch) {
      self(self, ch.get());
    }
  };
  collect(collect, snode_tree->root());
}

std::size_t CudaCProgramImpl::get_snode_num_dynamically_allocated(
    SNode *snode,
    uint64 *result_buffer) {
  return 0;
}

void CudaCProgramImpl::synchronize() {
  auto *stream = CUDAContext::get_instance().get_stream();
  CUDADriver::get_instance().stream_synchronize(stream);
}

std::unique_ptr<AotModuleBuilder> CudaCProgramImpl::make_aot_module_builder(
    const DeviceCapabilityConfig &caps) {
  TI_ERROR("cuda_c backend does not support AOT yet.");
}

void CudaCProgramImpl::allocate_field(SNode *snode) {
  TI_ASSERT(snode->type == SNodeType::place);
  int64 elements = snode->get_total_num_elements_towards_root();
  size_t bytes = elements * data_type_bytes(snode->dt);
  void *ptr = nullptr;
  CUDADriver::get_instance().malloc(&ptr, bytes);
  CUDADriver::get_instance().memset(ptr, 0, bytes);
  field_storage_[snode] = FieldStorage{ptr, bytes, elements,
                                       snode->dt->get_compute_type()};
}

void CudaCProgramImpl::free_field(SNode *snode) {
  auto it = field_storage_.find(snode);
  if (it != field_storage_.end()) {
    CUDADriver::get_instance().mem_free(it->second.device_ptr);
    field_storage_.erase(it);
  }
}

std::pair<const StructType *, size_t>
CudaCProgramImpl::get_struct_type_with_data_layout(
    const StructType *old_ty,
    const std::string &layout) {
  return pack_struct_type(old_ty);
}

std::unique_ptr<KernelCompiler> CudaCProgramImpl::make_kernel_compiler() {
  return std::make_unique<CudaCKernelCompiler>();
}

std::unique_ptr<KernelLauncher> CudaCProgramImpl::make_kernel_launcher() {
  CudaCKernelLauncher::Config config;
  config.program = this;
  return std::make_unique<CudaCKernelLauncher>(config);
}

}  // namespace taichi::lang
