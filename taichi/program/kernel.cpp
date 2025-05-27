#include "taichi/program/kernel.h"
#include <string>

#include "taichi/codegen/spirv/compiled_kernel_data.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/codegen/codegen.h"
#include "taichi/common/logging.h"
#include "taichi/common/task.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"

#ifdef TI_WITH_LLVM
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#endif

namespace taichi::lang {

class Function;

Kernel::Kernel(Program &program,
               const std::function<void()> &func,
               const std::string &primal_name,
               AutodiffMode autodiff_mode) {
  this->init(program, func, primal_name, autodiff_mode);
}

Kernel::Kernel(Program &program,
               const std::function<void(Kernel *)> &func,
               const std::string &primal_name,
               AutodiffMode autodiff_mode) {
  // due to #6362, we cannot write [func, this] { return func(this); }
  this->init(program, [&] { return func(this); }, primal_name, autodiff_mode);
}

Kernel::Kernel(Program &program,
               std::unique_ptr<IRNode> &&ir,
               const std::string &primal_name,
               AutodiffMode autodiff_mode) {
  this->arch = program.compile_config().arch;
  this->autodiff_mode = autodiff_mode;
  this->ir = std::move(ir);
  this->program = &program;
  is_accessor = false;
  ir_is_ast_ = false;  // CHI IR

  TI_ASSERT(this->ir->is<Block>());
  this->ir->as<Block>()->set_parent_callable(this);

  if (autodiff_mode == AutodiffMode::kNone) {
    name = primal_name;
  } else if (autodiff_mode == AutodiffMode::kForward) {
    name = primal_name + "_forward_grad";
  } else if (autodiff_mode == AutodiffMode::kReverse) {
    name = primal_name + "_reverse_grad";
  } else if (autodiff_mode == AutodiffMode::kCheckAutodiffValid) {
    name = primal_name + "_validate_grad";
  } else {
    TI_ERROR("Unsupported autodiff mode");
  }
}

Kernel::Kernel(Program &program, std::unique_ptr<CompiledKernelData> &&ckd,const std::string& kernel_key, const std::string &name, AutodiffMode autodiff_mode) {
  this->program = &program;
  this->name = name;
  this->autodiff_mode = autodiff_mode;
  this->arch = ckd->arch();
  this->is_accessor = false;
  this->ir_is_ast_ = false;
  this->kernel_key_ = kernel_key;
  
  TI_TRACE("Creating kernel '{}' from CompiledKernelData with arch={}", name, arch_name(this->arch));
  
  // 从 CompiledKernelData 中提取参数和返回值信息
  if (arch_uses_llvm(this->arch)) {
    auto *llvm_ckd = dynamic_cast<LLVM::CompiledKernelData *>(ckd.get());
    if (llvm_ckd) {
      const auto &data = llvm_ckd->get_internal_data();
      // 复制参数信息
      this->args_type = data.args_type;
      this->args_size = data.args_size;
      // 从 vector<pair> 转换到 unordered_map
      for (const auto &arg_pair : data.args) {
        this->nested_parameters[arg_pair.first] = arg_pair.second;
        std::string indices_str = "[";
        for (size_t i = 0; i < arg_pair.first.size(); ++i) {
          indices_str += std::to_string(arg_pair.first[i]);
          if (i < arg_pair.first.size() - 1) {
            indices_str += ", ";
          }
        }
        indices_str += "]";
        TI_TRACE("  LLVM Kernel arg: indices={}, type={}, is_array={}, is_argpack={}, total_dim={}",
                indices_str, arg_pair.second.get_dtype().to_string(),
                arg_pair.second.is_array, arg_pair.second.is_argpack,
                arg_pair.second.total_dim);
      }
      // 复制返回值信息
      this->ret_type = data.ret_type;
      this->ret_size = data.ret_size;
      this->rets = data.rets;
      
      TI_TRACE("  LLVM Kernel args_size={}, ret_size={}, rets_count={}", 
              this->args_size, this->ret_size, this->rets.size());
    }
  } else {
    // 处理 SPIRV 相关后端 (Vulkan/Metal/DirectX/OpenGL)
    auto *spirv_ckd = dynamic_cast<taichi::lang::spirv::CompiledKernelData *>(ckd.get());
    if (spirv_ckd) {
      const auto &ctx_attribs = spirv_ckd->get_internal_data().metadata.kernel_attribs.ctx_attribs;
      
      // 设置参数类型和大小
      this->args_type = ctx_attribs.args_type();
      this->args_size = ctx_attribs.args_bytes();
      
      TI_TRACE("  SPIRV Kernel args_type={}, args_size={}", 
              (this->args_type ? this->args_type->to_string() : "null"), this->args_size);
      
      // 从 arg_attribs_vec_ 转换到 nested_parameters
      for (const auto &arg_pair : ctx_attribs.args()) {
        const auto &indices = arg_pair.first;
        const auto &attrib = arg_pair.second;
        
        // 使用公共构造函数创建 Parameter 对象
        DataType element_type = PrimitiveType::get(attrib.dtype);
        if (!attrib.element_shape.empty()) {
          element_type = TypeFactory::get_instance().create_tensor_type(
              attrib.element_shape, element_type);
        }
        
        Parameter param(
            element_type,                  // dt
            attrib.is_array,               // is_array
            attrib.is_argpack,             // is_argpack
            0,                             // size_unused (不使用)
            attrib.field_dim + attrib.element_shape.size(), // total_dim
            attrib.element_shape,          // element_shape
            BufferFormat::unknown,         // format (默认值)
            false                          // needs_grad (默认值)
        );
        
        param.name = attrib.name;
        param.ptype = attrib.ptype;
        
        this->nested_parameters[indices] = param;
        
        std::string indices_str = "[";
        for (size_t i = 0; i < indices.size(); ++i) {
          indices_str += std::to_string(indices[i]);
          if (i < indices.size() - 1) {
            indices_str += ", ";
          }
        }
        indices_str += "]";
        
        TI_TRACE("  SPIRV Kernel arg: indices={}, name={}, type={}, is_array={}, is_argpack={}, total_dim={}",
                indices_str, param.name, param.get_dtype().to_string(),
                param.is_array, param.is_argpack, param.total_dim);
      }
      
      // 设置返回值信息
      this->ret_type = ctx_attribs.rets_type();
      this->ret_size = ctx_attribs.rets_bytes();
      
      TI_TRACE("  SPIRV Kernel ret_type={}, ret_size={}", 
              (this->ret_type ? this->ret_type->to_string() : "null"), this->ret_size);
      
      // 处理返回值
      if (ctx_attribs.has_rets()) {
        for (const auto &ret_attrib : ctx_attribs.rets()) {
          Ret ret(PrimitiveType::get(ret_attrib.dtype));
          this->rets.push_back(ret);
          
          TI_TRACE("  SPIRV Kernel ret: type={}", 
                  PrimitiveType::get(ret_attrib.dtype).to_string());
        }
      }
      
      // 处理 argpack 类型
      for (const auto &argpack_pair : ctx_attribs.argpack_types()) {
        const auto &indices = argpack_pair.first;
        const auto *type = argpack_pair.second;
        
        // 确保类型是 StructType
        if (type->is<StructType>()) {
          this->argpack_types[indices] = type->as<StructType>();
          
          std::string indices_str = "[";
          for (size_t i = 0; i < indices.size(); ++i) {
            indices_str += std::to_string(indices[i]);
            if (i < indices.size() - 1) {
              indices_str += ", ";
            }
          }
          indices_str += "]";
          
          TI_TRACE("  SPIRV Kernel argpack: indices={}, type={}", 
                  indices_str, type->to_string());
        } else {
          // 如果不是 StructType，可以记录警告或错误
          std::string indices_str = "[";
          for (size_t i = 0; i < indices.size(); ++i) {
            indices_str += std::to_string(indices[i]);
            if (i < indices.size() - 1) {
              indices_str += ", ";
            }
          }
          indices_str += "]";
          
          TI_WARN("Expected StructType for argpack at indices {}, got {}", 
                  indices_str, type->to_string());
        }
      }
    }
  }
  
  TI_TRACE("Kernel '{}' created with {} args and {} rets", 
          name, nested_parameters.size(), rets.size());
}

LaunchContextBuilder Kernel::make_launch_context() {
  return LaunchContextBuilder(this);
}

template <typename T>
T Kernel::fetch_ret(DataType dt, int i) {
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return (T)program->fetch_result<float32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return (T)program->fetch_result<float64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return (T)program->fetch_result<int32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return (T)program->fetch_result<int64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    return (T)program->fetch_result<int8>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return (T)program->fetch_result<int16>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u1)) {
    return (T)program->fetch_result<uint1>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    return (T)program->fetch_result<uint8>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return (T)program->fetch_result<uint16>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return (T)program->fetch_result<uint32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    return (T)program->fetch_result<uint64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
    // use f32 to interact with python
    return (T)program->fetch_result<float32>(i);
  } else {
    TI_NOT_IMPLEMENTED
  }
}

std::string Kernel::get_name() const {
  return name;
}

void Kernel::init(Program &program,
                  const std::function<void()> &func,
                  const std::string &primal_name,
                  AutodiffMode autodiff_mode) {
  this->autodiff_mode = autodiff_mode;
  this->program = &program;

  is_accessor = false;
  context = std::make_unique<FrontendContext>(program.compile_config().arch,
                                              /*is_kernel_=*/true);
  ir = context->get_root();

  TI_ASSERT(ir->is<Block>());
  ir->as<Block>()->set_parent_callable(this);

  ir_is_ast_ = true;
  arch = program.compile_config().arch;

  if (autodiff_mode == AutodiffMode::kNone) {
    name = primal_name;
  } else if (autodiff_mode == AutodiffMode::kCheckAutodiffValid) {
    name = primal_name + "_validate_grad";
  } else if (autodiff_mode == AutodiffMode::kForward) {
    name = primal_name + "_forward_grad";
  } else if (autodiff_mode == AutodiffMode::kReverse) {
    name = primal_name + "_reverse_grad";
  }

  func();
}
}  // namespace taichi::lang
