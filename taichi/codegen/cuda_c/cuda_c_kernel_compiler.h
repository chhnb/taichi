#pragma once

#include <memory>

#include "taichi/codegen/kernel_compiler.h"

namespace taichi::lang {

class CudaCKernelCompiler : public KernelCompiler {
 public:
  CudaCKernelCompiler() = default;
  ~CudaCKernelCompiler() override;

  IRNodePtr compile(const CompileConfig &compile_config,
                    const Kernel &kernel_def) const override;

  CKDPtr compile(const CompileConfig &compile_config,
                 const DeviceCapabilityConfig &device_caps,
                 const Kernel &kernel_def,
                 IRNode &chi_ir) const override;
};

}  // namespace taichi::lang
