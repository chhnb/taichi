#include "taichi/codegen/cuda_c/codegen_cuda_c.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <system_error>

#include "taichi/common/core.h"
#include "taichi/codegen/codegen_utils.h"
#include "taichi/ir/type_factory.h"
#include "taichi/ir/type_utils.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/util/hash.h"

namespace taichi::lang {

namespace {

std::string cuda_c_data_type_name(DataType dt) {
  auto primitive = dt->cast<PrimitiveType>();
  if (!primitive) {
    TI_ERROR("Unsupported data type {}", dt->to_string());
  }
  switch (primitive->type) {
    case PrimitiveTypeID::f32:
      return "float";
    case PrimitiveTypeID::f64:
      return "double";
    case PrimitiveTypeID::i32:
      return "int";
    case PrimitiveTypeID::i64:
      return "long long";
    case PrimitiveTypeID::u32:
      return "unsigned int";
    case PrimitiveTypeID::u64:
      return "unsigned long long";
    case PrimitiveTypeID::i16:
      return "short";
    case PrimitiveTypeID::u16:
      return "unsigned short";
    case PrimitiveTypeID::i8:
      return "signed char";
    case PrimitiveTypeID::u8:
      return "unsigned char";
    case PrimitiveTypeID::u1:
      return "bool";
    default:
      TI_ERROR("Unsupported primitive dtype");
      break;
  }
}

std::string make_unique_kernel_name(const std::string &base) {
  static std::atomic<int> counter{0};
  return fmt::format("{}_cuda_c_{}", base, counter.fetch_add(1));
}

std::string replace_all(std::string input,
                        const std::string &from,
                        const std::string &to) {
  if (from.empty()) {
    return input;
  }
  std::size_t pos = 0;
  while ((pos = input.find(from, pos)) != std::string::npos) {
    input.replace(pos, from.size(), to);
    pos += to.size();
  }
  return input;
}

std::string sanitize_filename(const std::string &name) {
  std::string sanitized;
  sanitized.reserve(name.size());
  for (unsigned char ch : name) {
    if (std::isalnum(ch) || ch == '_' || ch == '-') {
      sanitized.push_back(static_cast<char>(ch));
    } else {
      sanitized.push_back('_');
    }
  }
  if (sanitized.empty()) {
    sanitized = "kernel";
  }
  return sanitized;
}

std::string make_cuda_c_cache_key(const std::string &normalized_source,
                                  int compute_capability,
                                  const CompileConfig &config) {
  std::size_t seed = 0;
  hashing::hash_combine(seed, normalized_source);
  hashing::hash_combine(seed, compute_capability);
  hashing::hash_combine(seed, config.max_block_dim);
  hashing::hash_combine(seed, config.default_gpu_block_dim);
  hashing::hash_combine(seed, config.cuda_stack_limit);
  hashing::hash_combine(seed, config.fast_math);
  return fmt::format("{:016x}", seed);
}

void write_text_file(const std::filesystem::path &path,
                     const std::string &text) {
  std::ofstream ofs(path);
  TI_ERROR_IF(!ofs, "Failed to open {} for writing", path.string());
  ofs << text;
}

std::vector<char> read_binary_file(const std::filesystem::path &path) {
  std::ifstream ifs(path, std::ios::binary);
  TI_ERROR_IF(!ifs, "Failed to open {} for reading", path.string());
  return std::vector<char>((std::istreambuf_iterator<char>(ifs)),
                           std::istreambuf_iterator<char>());
}

std::string escape_c_string(const std::string &input) {
  std::string escaped;
  escaped.reserve(input.size());
  for (unsigned char ch : input) {
    switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (std::isprint(ch)) {
          escaped.push_back(static_cast<char>(ch));
        } else {
          escaped += fmt::format("\\x{:02x}", ch);
        }
        break;
    }
  }
  return escaped;
}

std::string zero_literal(DataType dt) {
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return "0.0f";
  }
  if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return "0.0";
  }
  return "0";
}

constexpr const char *kCudaCPreamble = R"(#include <cuda_runtime.h>

__device__ inline unsigned long long ti_cuda_c_random_source() {
  unsigned long long tid = static_cast<unsigned long long>(blockIdx.x) *
                               static_cast<unsigned long long>(blockDim.x) +
                           static_cast<unsigned long long>(threadIdx.x);
  unsigned long long clk = clock64();
  unsigned long long seed = clk ^ (tid + 0x9e3779b97f4a7c15ull);
  seed ^= seed >> 30;
  seed *= 0xbf58476d1ce4e5b9ull;
  seed ^= seed >> 27;
  seed *= 0x94d049bb133111ebull;
  seed ^= seed >> 31;
  return seed;
}

__device__ inline unsigned int ti_cuda_c_rand_u32() {
  return static_cast<unsigned int>(ti_cuda_c_random_source());
}

__device__ inline unsigned long long ti_cuda_c_rand_u64() {
  auto a = ti_cuda_c_random_source();
  auto b = ti_cuda_c_random_source();
  return (a << 32) ^ b;
}

__device__ inline int ti_cuda_c_rand_i32() {
  return static_cast<int>(ti_cuda_c_rand_u32());
}

__device__ inline long long ti_cuda_c_rand_i64() {
  return static_cast<long long>(ti_cuda_c_rand_u64());
}

__device__ inline float ti_cuda_c_rand_f32() {
  return (ti_cuda_c_rand_u32() >> 8) * (1.0f / 16777216.0f);
}

__device__ inline double ti_cuda_c_rand_f64() {
  return (ti_cuda_c_rand_u64() >> 11) * (1.0 / 9007199254740992.0);
}

__device__ inline bool ti_cuda_c_rand_u1() {
  return (ti_cuda_c_rand_u32() & 1u) != 0;
}

)";

}  // namespace

TaskCodeGenCudaC::TaskCodeGenCudaC(OffloadedStmt *offloaded,
                                   const Kernel &kernel)
    : offloaded_(offloaded), kernel_(kernel) {
}

bool TaskCodeGenCudaC::run(CudaCDeviceKernelInfo *info) {
  TI_ASSERT(offloaded_);
  TI_ASSERT(info);
  reset_codegen_state();
  needs_result_buffer_ = (kernel_.ret_size > 0);
  next_result_buffer_index_ = 0;
  switch (offloaded_->task_type) {
    case OffloadedTaskType::range_for:
      return run_range_for(info);
    case OffloadedTaskType::struct_for:
      return run_struct_for(info);
    case OffloadedTaskType::serial:
      return run_serial(info);
    default:
      TI_WARN("Unsupported offloaded task type {} in cuda_c backend",
              static_cast<int>(offloaded_->task_type));
      return false;
  }
}

bool TaskCodeGenCudaC::emit_statements() {
  for (auto &stmt : body_->statements) {
    if (!emit_statement(stmt.get())) {
      return false;
    }
  }
  return true;
}

bool TaskCodeGenCudaC::emit_block(Block *block, int indent_spaces) {
  if (!block || block->statements.empty()) {
    return true;
  }
  Block *old_body = body_;
  body_ = block;
  size_t start = body_lines_.size();
  bool ok = emit_statements();
  body_ = old_body;
  if (!ok) {
    return false;
  }
  if (indent_spaces > 0) {
    std::string extra(indent_spaces, ' ');
    for (size_t i = start; i < body_lines_.size(); ++i) {
      body_lines_[i] = extra + body_lines_[i];
    }
  }
  return true;
}

bool TaskCodeGenCudaC::emit_statement(Stmt *stmt) {
  if (auto alloca = stmt->cast<AllocaStmt>()) {
    auto element = alloca->ret_type.ptr_removed();
    if (!element || !element->is<PrimitiveType>()) {
      TI_WARN("Only primitive local variables supported in cuda_c backend");
      return false;
    }
    std::string name = fmt::format("local{}", local_var_counter_++);
    body_lines_.push_back(fmt::format("    {} {} = {};",
                                      cuda_c_data_type_name(element),
                                      name, zero_literal(element)));
    value_map_[stmt] = name;
    return true;
  }
  if (auto loop = stmt->cast<LoopIndexStmt>()) {
    if (auto offloaded = loop->loop->cast<OffloadedStmt>()) {
      auto it = loop_var_names_.find(offloaded);
      if (it != loop_var_names_.end() &&
          loop->index < static_cast<int>(it->second.size()) &&
          !it->second[loop->index].empty()) {
        value_map_[stmt] = it->second[loop->index];
      } else {
        value_map_[stmt] = "linear";
      }
      return true;
    }
    if (auto range_for = loop->loop->cast<RangeForStmt>()) {
      auto it = loop_var_names_.find(range_for);
      if (it == loop_var_names_.end() ||
          loop->index >= static_cast<int>(it->second.size()) ||
          it->second[loop->index].empty()) {
        TI_WARN("Loop var for RangeForStmt not found");
        return false;
      }
      value_map_[stmt] = it->second[loop->index];
      return true;
    }
    TI_WARN("Unsupported loop type for LoopIndexStmt");
    return false;
  }
  if (auto unary = stmt->cast<UnaryOpStmt>()) {
    if (unary->is_cast()) {
      value_map_[stmt] = fmt::format(
          "({})({})",
          cuda_c_data_type_name(unary->cast_type->get_compute_type()),
          value_of(unary->operand));
      return true;
    }
    switch (unary->op_type) {
      case UnaryOpType::neg:
        value_map_[stmt] = fmt::format("(-{})", value_of(unary->operand));
        return true;
      case UnaryOpType::bit_not:
        value_map_[stmt] = fmt::format("(~{})", value_of(unary->operand));
        return true;
      case UnaryOpType::logic_not:
        value_map_[stmt] = fmt::format("(!{})", value_of(unary->operand));
        return true;
      case UnaryOpType::sqrt:
        value_map_[stmt] =
            fmt::format("std::sqrt({})", value_of(unary->operand));
        return true;
      case UnaryOpType::rsqrt:
        value_map_[stmt] =
            fmt::format("(1.0 / std::sqrt({}))", value_of(unary->operand));
        return true;
      case UnaryOpType::exp:
        value_map_[stmt] =
            fmt::format("std::exp({})", value_of(unary->operand));
        return true;
      case UnaryOpType::log:
        value_map_[stmt] =
            fmt::format("std::log({})", value_of(unary->operand));
        return true;
      case UnaryOpType::sin:
        value_map_[stmt] =
            fmt::format("std::sin({})", value_of(unary->operand));
        return true;
      case UnaryOpType::cos:
        value_map_[stmt] =
            fmt::format("std::cos({})", value_of(unary->operand));
        return true;
      case UnaryOpType::tan:
        value_map_[stmt] =
            fmt::format("std::tan({})", value_of(unary->operand));
        return true;
      case UnaryOpType::tanh:
        value_map_[stmt] =
            fmt::format("std::tanh({})", value_of(unary->operand));
        return true;
      case UnaryOpType::asin:
        value_map_[stmt] =
            fmt::format("std::asin({})", value_of(unary->operand));
        return true;
      case UnaryOpType::acos:
        value_map_[stmt] =
            fmt::format("std::acos({})", value_of(unary->operand));
        return true;
      case UnaryOpType::abs:
        value_map_[stmt] =
            fmt::format("std::abs({})", value_of(unary->operand));
        return true;
      case UnaryOpType::round:
        value_map_[stmt] =
            fmt::format("std::round({})", value_of(unary->operand));
        return true;
      case UnaryOpType::floor:
        value_map_[stmt] =
            fmt::format("std::floor({})", value_of(unary->operand));
        return true;
      case UnaryOpType::ceil:
        value_map_[stmt] =
            fmt::format("std::ceil({})", value_of(unary->operand));
        return true;
      case UnaryOpType::inv:
      case UnaryOpType::rcp:
        value_map_[stmt] =
            fmt::format("(1.0 / ({}))", value_of(unary->operand));
        return true;
      case UnaryOpType::sgn:
        value_map_[stmt] = fmt::format("((({0}) > 0) - (({0}) < 0))",
                                       value_of(unary->operand));
        return true;
      default:
        TI_WARN("Unsupported unary op {}", (int)unary->op_type);
        return false;
    }
  }
  if (auto const_stmt = stmt->cast<ConstStmt>()) {
    value_map_[stmt] = const_literal(const_stmt);
    return true;
  }
  if (auto arg = stmt->cast<ArgLoadStmt>()) {
    auto dtype = arg->ret_type.ptr_removed();
    if (dtype && dtype->is<PrimitiveType>()) {
      value_map_[stmt] =
          get_or_make_scalar_param(arg->arg_id, arg->ret_type);
    } else {
      // Non-primitive args (e.g., ndarray descriptors) are handled via
      // dedicated params elsewhere; no direct kernel argument needed here.
      value_map_[stmt] = "0";
    }
    return true;
  }
  if (auto binary = stmt->cast<BinaryOpStmt>()) {
    auto emit = [&](const std::string &symbol) {
      value_map_[stmt] = fmt::format(
          "({} {} {})", value_of(binary->lhs), symbol, value_of(binary->rhs));
    };
    switch (binary->op_type) {
      case BinaryOpType::add:
        emit("+");
        return true;
      case BinaryOpType::mul:
        emit("*");
        return true;
      case BinaryOpType::sub:
        emit("-");
        return true;
      case BinaryOpType::div:
      case BinaryOpType::truediv:
        emit("/");
        return true;
      case BinaryOpType::floordiv: {
        if (stmt->ret_type->is_primitive(PrimitiveTypeID::f32)) {
          value_map_[stmt] = fmt::format("floorf({} / {})",
                                         value_of(binary->lhs),
                                         value_of(binary->rhs));
        } else if (stmt->ret_type->is_primitive(PrimitiveTypeID::f64)) {
          value_map_[stmt] = fmt::format("floor({} / {})",
                                         value_of(binary->lhs),
                                         value_of(binary->rhs));
        } else {
          emit("/");
        }
        return true;
      }
      case BinaryOpType::mod:
        emit("%");
        return true;
      case BinaryOpType::cmp_lt:
        emit("<");
        return true;
      case BinaryOpType::cmp_le:
        emit("<=");
        return true;
      case BinaryOpType::cmp_gt:
        emit(">");
        return true;
      case BinaryOpType::cmp_ge:
        emit(">=");
        return true;
      case BinaryOpType::cmp_eq:
        emit("==");
        return true;
      case BinaryOpType::cmp_ne:
        emit("!=");
        return true;
      case BinaryOpType::bit_and:
        emit("&");
        return true;
      case BinaryOpType::bit_or:
        emit("|");
        return true;
      case BinaryOpType::bit_xor:
        emit("^");
        return true;
      case BinaryOpType::bit_shl:
        emit("<<");
        return true;
      case BinaryOpType::bit_shr:
        emit(">>");
        return true;
      case BinaryOpType::logical_and:
        emit("&&");
        return true;
      case BinaryOpType::logical_or:
        emit("||");
        return true;
      default:
        TI_WARN("Unsupported binary op {}", (int)binary->op_type);
        return false;
    }
  }
  if (auto ptr = stmt->cast<GlobalPtrStmt>()) {
    auto snode = ptr->snode;
    if (snode->type != SNodeType::place) {
      TI_WARN("Only place snode supported");
      return false;
    }
    const SNode *coord_snode = snode;
    while (coord_snode && coord_snode->type == SNodeType::place) {
      coord_snode = coord_snode->parent;
    }
    bool has_index_metadata =
        coord_snode && coord_snode->num_active_indices > 0;
    std::string buf = get_or_make_buffer_name(snode);
    std::string index_expr = "0";
    if (!ptr->indices.empty()) {
      if (!has_index_metadata) {
        if (ptr->indices.size() != 1) {
          TI_WARN(
              "Unsupported multi-dimensional access on snode {} (type={}, "
              "num_indices={})",
              snode->name, static_cast<int>(snode->type),
              snode->num_active_indices);
          return false;
        }
        index_expr = value_of(ptr->indices[0]);
      } else {
        if ((int)ptr->indices.size() > coord_snode->num_active_indices) {
          TI_WARN(
              "Too many indices for snode {} (type={}, num_indices={}, "
              "provided={})",
              coord_snode->name, static_cast<int>(coord_snode->type),
              coord_snode->num_active_indices, ptr->indices.size());
          return false;
        }
        std::string linear = "0";
        for (int i = 0; i < (int)ptr->indices.size(); ++i) {
          int axis = coord_snode->physical_index_position[i];
          if (axis < 0 || !coord_snode->extractors[axis].active) {
            TI_WARN(
                "Invalid axis {} for snode {} (type={}, num_indices={})", axis,
                coord_snode->name, static_cast<int>(coord_snode->type),
                coord_snode->num_active_indices);
            return false;
          }
          int stride = coord_snode->extractors[axis].acc_shape;
          std::string term = value_of(ptr->indices[i]);
          if (stride != 1) {
            term = fmt::format("({}) * {}", term, stride);
          }
          if (linear == "0") {
            linear = term;
          } else {
            linear = fmt::format("({}) + ({})", linear, term);
          }
        }
        index_expr = linear;
      }
    }
    value_map_[stmt] = fmt::format("{}[({})]", buf, index_expr);
    return true;
  }
  if (auto tmp = stmt->cast<GlobalTemporaryStmt>()) {
    auto element = tmp->ret_type.ptr_removed();
    if (!element || !element->is<PrimitiveType>()) {
      TI_WARN("Only primitive temporaries supported in cuda_c backend");
      return false;
    }
    needs_temp_buffer_ = true;
    auto type_name = cuda_c_data_type_name(element->get_compute_type());
    value_map_[stmt] = fmt::format(
        "*(({0} *)((char *)temp_buffer + {1}ull))", type_name,
        static_cast<unsigned long long>(tmp->offset));
    return true;
  }
  if (auto load = stmt->cast<GlobalLoadStmt>()) {
    value_map_[stmt] = value_of(load->src);
    return true;
  }
  if (auto store = stmt->cast<GlobalStoreStmt>()) {
    body_lines_.push_back(
        fmt::format("    {} = {};", value_of(store->dest),
                    value_of(store->val)));
    return true;
  }
  if (auto store = stmt->cast<LocalStoreStmt>()) {
    body_lines_.push_back(
        fmt::format("    {} = {};", value_of(store->dest),
                    value_of(store->val)));
    return true;
  }
  if (auto load = stmt->cast<LocalLoadStmt>()) {
    value_map_[stmt] = value_of(load->src);
    return true;
  }
  if (auto ret = stmt->cast<ReturnStmt>()) {
    if (!needs_result_buffer_) {
      TI_WARN("Return statement encountered but kernel has no returns");
      return false;
    }
    for (auto *value : ret->values) {
      if (value->ret_type->is<TensorType>()) {
        TI_WARN("Tensor return not supported in cuda_c backend");
        return false;
      }
      int idx = next_result_buffer_index_++;
      auto line = store_return_value(value->ret_type, value_of(value), idx);
      if (line.empty()) {
        return false;
      }
      body_lines_.push_back(fmt::format("    {}", line));
    }
    body_lines_.push_back("    return;");
    return true;
  }
  if (auto print = stmt->cast<PrintStmt>()) {
    std::string format;
    std::vector<std::string> arg_exprs;
    for (int i = 0; i < print->contents.size(); ++i) {
      const auto &content = print->contents[i];
      if (std::holds_alternative<Stmt *>(content)) {
        auto arg_stmt = std::get<Stmt *>(content);
        if (arg_stmt->ret_type->is<TensorType>()) {
          TI_WARN("Tensor print not supported in cuda_c backend");
          return false;
        }
        std::string fmt_spec = merge_printf_specifier(
            print->formats[i],
            data_type_format(arg_stmt->ret_type, Arch::cuda));
        std::replace(fmt_spec.begin(), fmt_spec.end(), 'F', 'f');
        format += fmt_spec;
        arg_exprs.push_back(value_of(arg_stmt));
      } else {
        format += "%s";
        arg_exprs.push_back(fmt::format(
            "\"{}\"", escape_c_string(std::get<std::string>(content))));
      }
    }
    std::string line =
        fmt::format("    printf(\"{}\"", escape_c_string(format));
    for (auto &arg : arg_exprs) {
      line += fmt::format(", {}", arg);
    }
    line += ");";
    body_lines_.push_back(std::move(line));
    return true;
  }
  if (auto shape = stmt->cast<ExternalTensorShapeAlongAxisStmt>()) {
    auto key = shape->arg_id;
    key.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
    key.push_back(shape->axis);
    value_map_[stmt] =
        get_or_make_scalar_param(key, shape->ret_type);
    return true;
  }
  if (auto ext = stmt->cast<ExternalPtrStmt>()) {
    auto *arg_load = ext->base_ptr->cast<ArgLoadStmt>();
    if (!arg_load) {
      TI_WARN("ExternalPtr base must be ArgLoadStmt in cuda_c backend");
      return false;
    }
    auto *struct_type =
        arg_load->ret_type.ptr_removed()->cast<StructType>();
    if (!struct_type) {
      TI_WARN("Unsupported base type for ExternalPtrStmt");
      return false;
    }
    auto ptr_member = struct_type->get_element_type(
        {TypeFactory::DATA_PTR_POS_IN_NDARRAY});
    auto *pointer_type = ptr_member->cast<PointerType>();
    if (!pointer_type) {
      TI_WARN("Ndarray data pointer missing");
      return false;
    }
    DataType operand_dtype(pointer_type->get_pointee_type());
    DataType element_dtype = operand_dtype;
    if (auto tensor = operand_dtype->cast<TensorType>()) {
      element_dtype = DataType(tensor->get_element_type());
    }
    auto primitive = element_dtype->cast<PrimitiveType>();
    if (!primitive) {
      TI_WARN("Only primitive ndarray element supported in cuda_c backend");
      return false;
    }
    auto element_name = cuda_c_data_type_name(element_dtype);
    auto ptr_key = arg_load->arg_id;
    ptr_key.push_back(ext->is_grad ? TypeFactory::GRAD_PTR_POS_IN_NDARRAY
                                   : TypeFactory::DATA_PTR_POS_IN_NDARRAY);
    auto ptr_param =
        get_or_make_scalar_param(ptr_key, PrimitiveType::u64);
    int num_indices = ext->indices.size();
    auto dt = ext->ret_type.ptr_removed();
    int num_element_indices =
        dt->is<TensorType>() ? 0 : (int)ext->element_shape.size();
    int num_array_indices = num_indices - num_element_indices;
    std::vector<std::string> runtime_sizes(num_array_indices);
    for (int i = 0; i < num_array_indices; i++) {
      auto key = arg_load->arg_id;
      key.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
      key.push_back(i);
      runtime_sizes[i] =
          get_or_make_scalar_param(key, PrimitiveType::i32);
    }
    std::string linear = "0";
    size_t size_idx = 0;
    const size_t elem_offset = num_array_indices;
    for (int i = 0; i < num_indices; i++) {
      std::string stride;
      if (i >= elem_offset &&
          i < elem_offset + num_element_indices) {
        stride =
            std::to_string(ext->element_shape[i - elem_offset]);
      } else {
        TI_ASSERT(size_idx < runtime_sizes.size());
        stride = runtime_sizes[size_idx++];
      }
      linear = fmt::format("({} * {} + {})", linear, stride,
                           value_of(ext->indices[i]));
    }
    if (operand_dtype->is<TensorType>() && dt->is<TensorType>()) {
      auto num = dt->cast<TensorType>()->get_num_elements();
      linear = fmt::format("({}) * {}", linear, num);
    }
    auto base_ptr = fmt::format("({0} *)((unsigned long long)({1}))",
                                element_name, ptr_param);
    auto ptr_with_offset = fmt::format("({} + ({}))", base_ptr, linear);
    value_map_[stmt] = fmt::format("*{}", ptr_with_offset);
    return true;
  }
  if (auto atomic = stmt->cast<AtomicOpStmt>()) {
    auto dest = value_of(atomic->dest);
    auto val = value_of(atomic->val);
    auto emit_local = [&](const std::string &expr) {
      body_lines_.push_back(fmt::format("    {} = {};", dest, expr));
    };
    if (atomic->dest->is<AllocaStmt>()) {
      switch (atomic->op_type) {
        case AtomicOpType::add:
          emit_local(fmt::format("({} + {})", dest, val));
          return true;
        case AtomicOpType::sub:
          emit_local(fmt::format("({} - {})", dest, val));
          return true;
        case AtomicOpType::max:
          emit_local(fmt::format("std::max({}, {})", dest, val));
          return true;
        case AtomicOpType::min:
          emit_local(fmt::format("std::min({}, {})", dest, val));
          return true;
        default:
          TI_WARN("Unsupported local atomic op {}", (int)atomic->op_type);
          return false;
      }
    }
    switch (atomic->op_type) {
      case AtomicOpType::add:
        body_lines_.push_back(
            fmt::format("    atomicAdd(&{}, {});", dest, val));
        break;
      case AtomicOpType::sub:
        body_lines_.push_back(
            fmt::format("    atomicAdd(&{}, -({}));", dest, val));
        break;
      case AtomicOpType::max:
        body_lines_.push_back(
            fmt::format("    atomicMax(&{}, {});", dest, val));
        break;
      case AtomicOpType::min:
        body_lines_.push_back(
            fmt::format("    atomicMin(&{}, {});", dest, val));
        break;
      default:
        TI_WARN("Unsupported atomic op {}", (int)atomic->op_type);
        return false;
    }
    return true;
  }
  if (auto if_stmt = stmt->cast<IfStmt>()) {
    body_lines_.push_back(
        fmt::format("    if ({}) {{", value_of(if_stmt->cond)));
    if (!emit_block(if_stmt->true_statements.get(), 4)) {
      return false;
    }
    body_lines_.push_back("    }");
    if (if_stmt->false_statements &&
        !if_stmt->false_statements->statements.empty()) {
      body_lines_.push_back("    else {");
      if (!emit_block(if_stmt->false_statements.get(), 4)) {
        return false;
      }
      body_lines_.push_back("    }");
    }
    return true;
  }
  if (auto ext_shape = stmt->cast<ExternalTensorShapeAlongAxisStmt>()) {
    auto key = ext_shape->arg_id;
    key.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
    key.push_back(ext_shape->axis);
    value_map_[stmt] =
        get_or_make_scalar_param(key, PrimitiveType::i32);
    return true;
  }
  if (auto while_stmt = stmt->cast<WhileStmt>()) {
    body_lines_.push_back("    while (true) {");
    if (!emit_block(while_stmt->body.get(), 4)) {
      return false;
    }
    body_lines_.push_back("    }");
    return true;
  }
  if (auto rand = stmt->cast<RandStmt>()) {
    needs_rand_helpers_ = true;
    auto dt = rand->ret_type->get_compute_type();
    if (dt->is_primitive(PrimitiveTypeID::f32)) {
      value_map_[stmt] = "ti_cuda_c_rand_f32()";
      return true;
    }
    if (dt->is_primitive(PrimitiveTypeID::f64)) {
      value_map_[stmt] = "ti_cuda_c_rand_f64()";
      return true;
    }
    if (dt->is_primitive(PrimitiveTypeID::i32)) {
      value_map_[stmt] = "ti_cuda_c_rand_i32()";
      return true;
    }
    if (dt->is_primitive(PrimitiveTypeID::i64)) {
      value_map_[stmt] = "ti_cuda_c_rand_i64()";
      return true;
    }
    if (dt->is_primitive(PrimitiveTypeID::u32)) {
      value_map_[stmt] = "ti_cuda_c_rand_u32()";
      return true;
    }
    if (dt->is_primitive(PrimitiveTypeID::u64)) {
      value_map_[stmt] = "ti_cuda_c_rand_u64()";
      return true;
    }
    if (dt->is_primitive(PrimitiveTypeID::u1)) {
      value_map_[stmt] = "ti_cuda_c_rand_u1()";
      return true;
    }
    TI_WARN("Unsupported rand dtype {}", dt->to_string());
    return false;
  }
  if (auto range_for = stmt->cast<RangeForStmt>()) {
    if (!range_for->begin || !range_for->end) {
      TI_WARN("RangeForStmt bounds missing");
      return false;
    }
    std::string loop_var =
        fmt::format("range_index{}", local_var_counter_++);
    auto begin_expr = value_of(range_for->begin);
    auto end_expr = value_of(range_for->end);
    std::string init_expr =
        range_for->reversed ? fmt::format("({}) - 1", end_expr) : begin_expr;
    std::string cond_expr =
        range_for->reversed
            ? fmt::format("{} >= ({})", loop_var, begin_expr)
            : fmt::format("{} < ({})", loop_var, end_expr);
    std::string update_expr =
        range_for->reversed ? fmt::format("--{}", loop_var)
                            : fmt::format("++{}", loop_var);
    auto &names = loop_var_names_[range_for];
    if (names.empty()) {
      names.resize(1);
    }
    names[0] = loop_var;
    body_lines_.push_back(
        fmt::format("    for (int {0} = {1}; {2}; {3}) {{", loop_var,
                    init_expr, cond_expr, update_expr));
    if (!emit_block(range_for->body.get(), 4)) {
      return false;
    }
    body_lines_.push_back("    }");
    return true;
  }
  if (auto control = stmt->cast<WhileControlStmt>()) {
    if (!control->cond) {
      TI_WARN("WhileControlStmt cond missing");
      return false;
    }
    body_lines_.push_back(
        fmt::format("    if (!({})) break;", value_of(control->cond)));
    return true;
  }
  if (auto cont = stmt->cast<ContinueStmt>()) {
    body_lines_.push_back("    continue;");
    return true;
  }
  TI_WARN("Unsupported statement id={} type={} in cuda_c backend", stmt->id,
          stmt->type());
  return false;
}

std::string TaskCodeGenCudaC::store_return_value(const DataType &dt,
                                                 const std::string &value,
                                                 int index) {
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return fmt::format(
        "result_buffer[{0}] = (unsigned long long)(unsigned int)__float_as_int({1});",
        index, value);
  }
  if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return fmt::format(
        "result_buffer[{0}] = (unsigned long long)__double_as_longlong({1});",
        index, value);
  }
  if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return fmt::format(
        "result_buffer[{0}] = (unsigned long long)(long long)({1});", index,
        value);
  }
  if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return fmt::format(
        "result_buffer[{0}] = (unsigned long long)(unsigned int)({1});", index,
        value);
  }
  if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return fmt::format(
        "result_buffer[{0}] = (unsigned long long)(long long)({1});", index,
        value);
  }
  if (dt->is_primitive(PrimitiveTypeID::u64)) {
    return fmt::format("result_buffer[{0}] = (unsigned long long)({1});", index,
                       value);
  }
  if (dt->is_primitive(PrimitiveTypeID::u1)) {
    return fmt::format("result_buffer[{0}] = (unsigned long long)({1});", index,
                       value);
  }
  TI_WARN("Unsupported return dtype {}", dt->to_string());
  return "";
}

std::string TaskCodeGenCudaC::get_or_make_scalar_param(
    const std::vector<int> &key,
    DataType dtype) {
  auto it = arg_to_param_.find(key);
  if (it != arg_to_param_.end()) {
    return it->second;
  }
  int idx = arg_to_param_.size();
  auto name = fmt::format("arg{}", idx);
  arg_to_param_[key] = name;
  CudaCArgInfo info;
  info.dtype = dtype;
  info.arg_id = key;
  arg_infos_.push_back(info);
  param_decls_.push_back(fmt::format("{} {}",
                                     cuda_c_data_type_name(
                                         dtype->get_compute_type()),
                                     name));
  CudaCKernelParam param;
  param.kind = CudaCKernelParamKind::kArg;
  param.index = static_cast<int>(arg_infos_.size()) - 1;
  param_order_.push_back(param);
  return name;
}

std::string TaskCodeGenCudaC::const_literal(ConstStmt *stmt) {
  if (stmt->ret_type->is_primitive(PrimitiveTypeID::f32)) {
    return fmt::format("{}", stmt->val.val_f32);
  }
  if (stmt->ret_type->is_primitive(PrimitiveTypeID::i32)) {
    return fmt::format("{}", stmt->val.val_i32);
  }
  TI_ERROR("Unsupported const literal type");
}

std::string TaskCodeGenCudaC::get_or_make_buffer_name(const SNode *snode) {
  auto it = snode_param_names_.find(snode);
  if (it != snode_param_names_.end()) {
    return it->second;
  }
  int idx = snode_param_names_.size();
  auto name = fmt::format(
      "{} *buf{}", cuda_c_data_type_name(snode->dt->get_compute_type()), idx);
  auto short_name = fmt::format("buf{}", idx);
  snode_param_names_[snode] = short_name;
  snode_params_order_.push_back(snode);
  param_decls_.push_back(name);
  CudaCKernelParam param;
  param.kind = CudaCKernelParamKind::kSNode;
  param.index = static_cast<int>(snode_params_order_.size()) - 1;
  param_order_.push_back(param);
  return short_name;
}

std::string TaskCodeGenCudaC::value_of(Stmt *stmt) {
  TI_ASSERT(value_map_.count(stmt));
  return value_map_.at(stmt);
}

std::string TaskCodeGenCudaC::indent(const std::string &content) {
  std::stringstream ss(content);
  std::string line;
  std::string out;
  while (std::getline(ss, line)) {
    out += "    " + line + "\n";
  }
  return out;
}

std::string TaskCodeGenCudaC::block_lines() const {
  std::string result;
  for (const auto &line : body_lines_) {
    result += line + "\n";
  }
  return result;
}

std::string TaskCodeGenCudaC::temp_buffer_value_expr(std::size_t offset,
                                                     DataType dt) {
  needs_temp_buffer_ = true;
  auto type_name = cuda_c_data_type_name(dt);
  return fmt::format("*(({0} *)((char *)temp_buffer + {1}ull))", type_name,
                     static_cast<unsigned long long>(offset));
}

std::string TaskCodeGenCudaC::make_range_bound_decl(
    const std::string &var_name,
    bool is_begin) {
  DataType bound_type = PrimitiveType::i32;
  if ((is_begin && offloaded_->const_begin) ||
      (!is_begin && offloaded_->const_end)) {
    int value = is_begin ? offloaded_->begin_value : offloaded_->end_value;
    return fmt::format("  int {} = {};\n", var_name, value);
  }
  size_t offset = is_begin ? offloaded_->begin_offset : offloaded_->end_offset;
  auto expr = temp_buffer_value_expr(offset, bound_type);
  return fmt::format("  int {} = {};\n", var_name, expr);
}

void TaskCodeGenCudaC::reset_codegen_state() {
  value_map_.clear();
  body_lines_.clear();
  snode_param_names_.clear();
  snode_params_order_.clear();
  param_decls_.clear();
  param_order_.clear();
  arg_to_param_.clear();
  arg_infos_.clear();
  local_var_counter_ = 0;
  needs_temp_buffer_ = false;
  needs_rand_helpers_ = false;
  loop_var_names_.clear();
}

void TaskCodeGenCudaC::append_runtime_params() {
  if (needs_result_buffer_) {
    param_decls_.push_back("unsigned long long *result_buffer");
    CudaCKernelParam param;
    param.kind = CudaCKernelParamKind::kResultBuffer;
    param_order_.push_back(param);
  }
  if (needs_temp_buffer_) {
    param_decls_.push_back("void *temp_buffer");
    CudaCKernelParam param;
    param.kind = CudaCKernelParamKind::kTempBuffer;
    param_order_.push_back(param);
  }
}

bool TaskCodeGenCudaC::run_range_for(CudaCDeviceKernelInfo *info) {
  bool has_const_bounds = offloaded_->const_begin && offloaded_->const_end;
  if (has_const_bounds) {
    begin_ = offloaded_->begin_value;
    end_ = offloaded_->end_value;
  } else {
    begin_ = 0;
    end_ = 0;
  }
  kernel_name_ = make_unique_kernel_name(kernel_.get_name());
  grid_dim_ = offloaded_->grid_dim;
  block_dim_ = offloaded_->block_dim;
  int total_elements = has_const_bounds ? (end_ - begin_) : -1;
  if (grid_dim_ <= 0) {
    if (!has_const_bounds) {
      TI_WARN("Dynamic range_for requires explicit grid dimension");
      grid_dim_ = 1;
    } else {
      if (block_dim_ <= 0) {
        block_dim_ = 128;
      }
      grid_dim_ = std::max(1, (total_elements + block_dim_ - 1) / block_dim_);
    }
  }
  body_ = offloaded_->body.get();
  if (!body_) {
    TI_ERROR("Offloaded body missing");
  }
  if (!emit_statements()) {
    return false;
  }
  auto begin_decl = make_range_bound_decl("range_begin", /*is_begin=*/true);
  auto end_decl = make_range_bound_decl("range_end", /*is_begin=*/false);
  append_runtime_params();
  std::string source = fmt::format(
      "{}extern \"C\" __global__\n"
      "void {}({}) {{\n"
      "{}"
      "  int linear = blockIdx.x * blockDim.x + threadIdx.x;\n"
      "  if (linear < range_end && linear >= range_begin) {{\n"
      "{}"
      "  }}\n"
      "}}\n",
      needs_rand_helpers_ ? kCudaCPreamble : "", kernel_name_,
      fmt::join(param_decls_, ", "),
      begin_decl + end_decl,
      indent(block_lines()));
  info->kernel_name = kernel_name_;
  info->cuda_src = std::move(source);
  info->grid_dim = grid_dim_;
  info->block_dim = block_dim_;
  info->snodes = snode_params_order_;
  info->arg_infos = arg_infos_;
  info->has_result_buffer = needs_result_buffer_;
  info->has_temp_buffer = needs_temp_buffer_;
  info->params = param_order_;
  return true;
}

bool TaskCodeGenCudaC::run_struct_for(CudaCDeviceKernelInfo *info) {
  kernel_name_ = make_unique_kernel_name(kernel_.get_name());
  auto *snode = offloaded_->snode;
  TI_ASSERT(snode);
  int dims = snode->num_active_indices;
  std::vector<int> shapes(dims, 1);
  std::vector<int> strides(dims, 1);
  for (int i = 0; i < dims; ++i) {
    int axis = snode->physical_index_position[i];
    if (axis >= 0) {
      shapes[i] = snode->extractors[axis].shape;
      strides[i] = snode->extractors[axis].acc_shape;
    }
  }
  begin_ = 0;
  end_ = 1;
  for (int s : shapes) {
    end_ *= std::max(1, s);
  }
  grid_dim_ = offloaded_->grid_dim;
  block_dim_ = offloaded_->block_dim;
  if (grid_dim_ <= 0) {
    if (block_dim_ <= 0) {
      block_dim_ = 128;
    }
    grid_dim_ = std::max(1, (end_ + block_dim_ - 1) / block_dim_);
  }
  std::vector<std::string> index_exprs(dims);
  for (int i = 0; i < dims; ++i) {
    int stride = strides[i] <= 0 ? 1 : strides[i];
    int shape = std::max(1, shapes[i]);
    std::string expr =
        fmt::format("((linear / {0}) % {1})", stride, shape);
    if (i < static_cast<int>(offloaded_->index_offsets.size())) {
      int offset = offloaded_->index_offsets[i];
      if (offset != 0) {
        expr = fmt::format("({}) + {}", expr, offset);
      }
    }
    index_exprs[i] = expr;
  }
  loop_var_names_[offloaded_] = index_exprs;
  body_ = offloaded_->body.get();
  if (!body_) {
    TI_ERROR("Offloaded body missing");
  }
  if (!emit_statements()) {
    return false;
  }
  append_runtime_params();
  std::string source = fmt::format(
      "{}extern \"C\" __global__\n"
      "void {}({}) {{\n"
      "  int range_begin = 0;\n"
      "  int range_end = {};\n"
      "  int linear = blockIdx.x * blockDim.x + threadIdx.x;\n"
      "  if (linear < range_end && linear >= range_begin) {{\n"
      "{}"
      "  }}\n"
      "}}\n",
      needs_rand_helpers_ ? kCudaCPreamble : "", kernel_name_,
      fmt::join(param_decls_, ", "),
      end_, indent(block_lines()));
  info->kernel_name = kernel_name_;
  info->cuda_src = std::move(source);
  info->grid_dim = grid_dim_;
  info->block_dim = block_dim_;
  info->snodes = snode_params_order_;
  info->arg_infos = arg_infos_;
  info->has_result_buffer = needs_result_buffer_;
  info->has_temp_buffer = needs_temp_buffer_;
  info->params = param_order_;
  return true;
}

bool TaskCodeGenCudaC::run_serial(CudaCDeviceKernelInfo *info) {
  kernel_name_ = make_unique_kernel_name(kernel_.get_name());
  grid_dim_ = 1;
  block_dim_ = 1;
  body_ = offloaded_->body.get();
  if (!body_) {
    TI_ERROR("Offloaded body missing");
  }
  if (!emit_statements()) {
    return false;
  }
  append_runtime_params();
  std::string source = fmt::format(
      "{}extern \"C\" __global__\n"
      "void {}({}) {{\n"
      "{}"
      "}}\n",
      needs_rand_helpers_ ? kCudaCPreamble : "", kernel_name_,
      fmt::join(param_decls_, ", "),
      indent(block_lines()));
  info->kernel_name = kernel_name_;
  info->cuda_src = std::move(source);
  info->grid_dim = grid_dim_;
  info->block_dim = block_dim_;
  info->snodes = snode_params_order_;
  info->arg_infos = arg_infos_;
  info->has_result_buffer = needs_result_buffer_;
  info->has_temp_buffer = needs_temp_buffer_;
  info->params = param_order_;
  return true;
}

std::vector<char> compile_cuda_c_with_nvcc(const std::string &source,
                                           const std::string &kernel_name,
                                           const CompileConfig &config) {
  namespace fs = std::filesystem;
  const bool dump_generated = !config.offline_cache_file_path.empty();
  const bool cache_enabled = config.offline_cache && dump_generated;
  fs::path cache_dir;
  fs::path cache_cu_path;
  fs::path cache_cubin_path;
  int cc = CUDAContext::get_instance().get_compute_capability();
  std::string arch = fmt::format("sm_{}", cc);
  if (cache_enabled) {
    auto normalized = replace_all(source, kernel_name, "{kernel}");
    cache_dir = fs::path(config.offline_cache_file_path) / "cuda_c";
    std::error_code ec;
    fs::create_directories(cache_dir, ec);
    auto cache_key = make_cuda_c_cache_key(normalized, cc, config);
    auto sanitized_name = sanitize_filename(kernel_name);
    auto cache_base = fmt::format("{}_{}", sanitized_name, cache_key);
    cache_cu_path = cache_dir / (cache_base + ".cu");
    cache_cubin_path = cache_dir / (cache_base + ".cubin");
    if (fs::exists(cache_cubin_path)) {
      return read_binary_file(cache_cubin_path);
    }
  }

  fs::path tmp_dir =
      fs::temp_directory_path() /
      fmt::format("taichi_cuda_c_{}", fmt::ptr(&kernel_name));
  fs::create_directories(tmp_dir);
  fs::path cu_path = tmp_dir / (kernel_name + ".cu");
  fs::path cubin_path = tmp_dir / (kernel_name + ".cubin");
  write_text_file(cu_path, source);
  auto cmd = fmt::format(
      "nvcc -std=c++14 -arch={} --generate-code=arch=compute_{},code=sm_{} "
      "-cubin {} -o {}",
      arch, cc, cc, cu_path.string(), cubin_path.string());
  TI_INFO(cmd);
  int ret = std::system(cmd.c_str());
  if (ret != 0) {
    TI_ERROR("nvcc failed ({}) while compiling {}", ret, kernel_name);
  }
  auto cubin = read_binary_file(cubin_path);

  if (cache_enabled) {
    std::error_code ec;
    fs::copy_file(cu_path, cache_cu_path, fs::copy_options::overwrite_existing,
                  ec);
    ec.clear();
    fs::copy_file(cubin_path, cache_cubin_path,
                  fs::copy_options::overwrite_existing, ec);
  } else if (dump_generated) {
    cache_dir = fs::path(config.offline_cache_file_path) / "cuda_c";
    std::error_code ec;
    fs::create_directories(cache_dir, ec);
    auto cached_cu = cache_dir / (kernel_name + ".cu");
    auto cached_cubin = cache_dir / (kernel_name + ".cubin");
    fs::copy_file(cu_path, cached_cu, fs::copy_options::overwrite_existing,
                  ec);
    ec.clear();
    fs::copy_file(cubin_path, cached_cubin,
                  fs::copy_options::overwrite_existing, ec);
  }

  fs::remove(cu_path);
  fs::remove(cubin_path);
  fs::remove(tmp_dir);
  return cubin;
}

}  // namespace taichi::lang
