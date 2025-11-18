#include "taichi/codegen/cuda_c/cuda_c_kernel_compiler.h"

#include "taichi/codegen/cuda_c/codegen_cuda_c.h"
#include "taichi/codegen/cuda_c/cuda_c_compiled_data.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"

namespace taichi::lang {

namespace {

const SNode *extract_accessor_snode(IRNode &ir) {
  auto block = ir.as<Block>();
  TI_ASSERT(block && !block->statements.empty());
  auto off = block->statements[0]->cast<OffloadedStmt>();
  TI_ASSERT(off);
  auto body = off->body.get();
  for (auto &stmt : body->statements) {
    if (auto ptr = stmt->cast<GlobalPtrStmt>()) {
      return ptr->snode;
    }
  }
  return nullptr;
}

}  // namespace

CudaCKernelCompiler::~CudaCKernelCompiler() = default;

KernelCompiler::IRNodePtr CudaCKernelCompiler::compile(
    const CompileConfig &compile_config,
    const Kernel &kernel_def) const {
  auto ir = irpass::analysis::clone(kernel_def.ir.get());
  bool verbose = compile_config.print_ir;
  if (kernel_def.is_accessor && !compile_config.print_accessor_ir) {
    verbose = false;
  }
  irpass::compile_to_offloads(ir.get(), compile_config, &kernel_def, verbose,
                              kernel_def.autodiff_mode,
                              /*ad_use_stack=*/true,
                              /*start_from_ast=*/kernel_def.ir_is_ast());
  return ir;
}

KernelCompiler::CKDPtr CudaCKernelCompiler::compile(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &device_caps [[maybe_unused]],
    const Kernel &kernel_def,
    IRNode &chi_ir) const {
  auto data = std::make_unique<CudaCCompiledKernelData>(
      kernel_def.is_accessor ? (kernel_def.rets.empty()
                                    ? CudaCKernelKind::kAccessorWriter
                                    : CudaCKernelKind::kAccessorReader)
                             : CudaCKernelKind::kDevice);
  if (kernel_def.is_accessor) {
    const SNode *target = extract_accessor_snode(chi_ir);
    data->set_accessor_snode(target);
    return data;
  }
  auto block = chi_ir.as<Block>();
  if (!block || block->statements.empty()) {
    TI_ERROR("Empty kernel body");
  }
  for (auto &stmt : block->statements) {
    auto off = stmt->cast<OffloadedStmt>();
    if (!off) {
      TI_ERROR("Kernel body contains non-offloaded statement");
    }
    CudaCDeviceKernelInfo info;
    TaskCodeGenCudaC builder(off, kernel_def);
    if (!builder.run(&info)) {
      TI_ERROR("Failed to lower kernel {} to CUDA C source",
               kernel_def.get_name());
    }
    info.cubin = compile_cuda_c_with_nvcc(info.cuda_src, info.kernel_name,
                                          compile_config);
    data->add_device_info(std::move(info));
  }
  return data;
}

}  // namespace taichi::lang
