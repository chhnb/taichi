#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/program/compile_config.h"
#include "taichi/program/kernel.h"
#include "taichi/util/hash.h"

namespace taichi::lang {

struct CudaCArgInfo {
  DataType dtype{PrimitiveType::unknown};
  std::vector<int> arg_id;
  bool is_return{false};
};

enum class CudaCKernelParamKind {
  kSNode,
  kArg,
  kResultBuffer,
  kTempBuffer
};

struct CudaCKernelParam {
  CudaCKernelParamKind kind{CudaCKernelParamKind::kSNode};
  int index{0};
};

struct CudaCDeviceKernelInfo {
  std::string kernel_name;
  std::string cuda_src;
  std::vector<char> cubin;
  int grid_dim{1};
  int block_dim{1};
  bool has_result_buffer{false};
  bool has_temp_buffer{false};
  std::vector<const SNode *> snodes;
  std::vector<CudaCArgInfo> arg_infos;
  std::vector<CudaCKernelParam> params;
};

class TaskCodeGenCudaC {
 public:
  TaskCodeGenCudaC(OffloadedStmt *offloaded, const Kernel &kernel);

  bool run(CudaCDeviceKernelInfo *info);

 private:
  bool run_range_for(CudaCDeviceKernelInfo *info);
  bool run_struct_for(CudaCDeviceKernelInfo *info);
  bool run_serial(CudaCDeviceKernelInfo *info);
  void reset_codegen_state();
  bool emit_statements();
  bool emit_block(Block *block, int indent_spaces);
  bool emit_statement(Stmt *stmt);
  std::string get_or_make_scalar_param(const std::vector<int> &key,
                                       DataType dtype);
  std::string store_return_value(const DataType &dt,
                                 const std::string &value,
                                 int index);
  void append_runtime_params();
  std::string const_literal(ConstStmt *stmt);
  std::string get_or_make_buffer_name(const SNode *snode);
  std::string value_of(Stmt *stmt);
  std::string indent(const std::string &content);
  std::string block_lines() const;
  std::string temp_buffer_value_expr(std::size_t offset, DataType dt);
  std::string make_range_bound_decl(const std::string &var_name,
                                    bool is_begin);
  OffloadedStmt *offloaded_{nullptr};
  const Kernel &kernel_;
  Block *body_{nullptr};
  std::string kernel_name_;
  int begin_{0};
  int end_{0};
  int grid_dim_{1};
  int block_dim_{1};

  bool needs_result_buffer_{false};
  bool needs_temp_buffer_{false};
  int next_result_buffer_index_{0};
  int local_var_counter_{0};
  bool needs_rand_helpers_{false};

  std::unordered_map<Stmt *, std::string> value_map_;
  std::vector<std::string> body_lines_;
  std::unordered_map<const SNode *, std::string> snode_param_names_;
  std::vector<const SNode *> snode_params_order_;
  std::vector<std::string> param_decls_;
  std::vector<CudaCKernelParam> param_order_;
  std::unordered_map<std::vector<int>,
                     std::string,
                     hashing::Hasher<std::vector<int>>>
      arg_to_param_;
  std::vector<CudaCArgInfo> arg_infos_;
  std::unordered_map<const Stmt *, std::vector<std::string>> loop_var_names_;
};

std::vector<char> compile_cuda_c_with_nvcc(const std::string &source,
                                           const std::string &kernel_name,
                                           const CompileConfig &config);

}  // namespace taichi::lang
