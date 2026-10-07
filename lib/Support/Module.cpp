
#include "Support/Module.h"
#include "Support/MathUtil.h"
#include "Support/ModuleEnum.cpp.inc"
#include <fstream>

namespace im {
namespace module {
struct Attr {
  static constexpr llvm::StringRef STATE = "module.state";
  static constexpr llvm::StringRef CHIP = "module.chip";
  static constexpr llvm::StringRef WEIGHT_FILE = "module.weight_file";
  static constexpr llvm::StringRef FLOPS = "module.FLOPs";
  static constexpr llvm::StringRef CORES = "module.cores";
  static constexpr llvm::StringRef DEVICES = "module.devices";
  static constexpr llvm::StringRef COEFF_ADDR = "module.coeff_addr";
  static constexpr llvm::StringRef COEFF_SIZE = "module.coeff_size";
  static constexpr llvm::StringRef NEURON_ADDR = "module.neuron_addr";
  static constexpr llvm::StringRef NEURON_SIZE = "module.neuron_size";
  static constexpr llvm::StringRef IO_ADDR = "module.io_addr";
  static constexpr llvm::StringRef IO_SIZE = "module.io_size";
  static constexpr llvm::StringRef GMEM_PRIVATE_SIZE = "module.private_size";
  static constexpr llvm::StringRef ASYMMETRIC = "module.asymmetric";
  static constexpr llvm::StringRef MODE = "module.mode";
  static constexpr llvm::StringRef PLATFORM = "module.platform";
  static constexpr llvm::StringRef POSTPROCESS = "module.postprocess";
  static constexpr llvm::StringRef DEVICE_ID = "module.device_id";
  static constexpr llvm::StringRef STEP = "module.step";
  static constexpr llvm::StringRef INPUTS = "module.inputs";
  static constexpr llvm::StringRef OUTPUTS = "module.outputs";
  static constexpr llvm::StringRef TRAIN = "module.train";
  static constexpr llvm::StringRef ADDR_MODE = "module.addr_mode";
  static constexpr llvm::StringRef QUANT_GROUP_SIZE = "module.q_group_size";
  static constexpr llvm::StringRef QUANT_SYMMETRIC = "module.q_symmetric";
  static constexpr llvm::StringRef FRONT_RUN_MODE = "module.front_run_mode";
  static constexpr llvm::StringRef DYNAMIC_COEFF_OFFSET =
      "module.dynamic_coeff_offset";
  static constexpr llvm::StringRef HIGH_PRECISION = "module.high_precision";
  static constexpr llvm::StringRef LORA_RANK = "module.lora_rank";
};

static ModuleOp m = nullptr;
static MLIRContext *ctx = nullptr;
static Chip chip = Chip::ALL;
static Platform platform = Platform::ONNX;
static std::unique_ptr<mlir::TensorFile> wFile = nullptr;
static std::string weightFileName = "";
static bool b_weight_in_mem = false;
static std::string debug_cmd = "";
std::unordered_map<std::string, int> patternMatchCounts;
std::mutex patternMatchCountsMutex;

void init(ModuleOp module) {
  m = module;
  ctx = m.getContext();
  auto chip_ = m->getAttrOfType<StringAttr>(Attr::CHIP);
  chip = symbolizeChip(chip_).value_or(Chip::ALL);
  wFile = nullptr;
  if (m->hasAttrOfType<StringAttr>(Attr::PLATFORM)) {
    auto p = m->getAttrOfType<StringAttr>(Attr::PLATFORM);
    platform = symbolizePlatform(p).value_or(Platform::ONNX);
  } else {
    platform = Platform::ONNX;
  }

  std::ifstream file("/tmp/debug_cmd");
  if (file.is_open()) {
    std::getline(file, debug_cmd);
    file.close();
  }
}

// int32_t cur_log_level = 0;
void init_loglevel(int32_t log_level) { SetLogFlag(log_level); }

void setWeightInMemFlag(bool enable) { b_weight_in_mem = enable; }

bool getWeightInMemFlag() { return b_weight_in_mem; }

front::NoneOp getNoneOp(Operation *op) {
  assert(op != nullptr);
  if (auto noneOp = dyn_cast<front::NoneOp>(op)) {
    return noneOp;
  }
  FuncOp funcOp;
  if (isa<FuncOp>(op)) {
    funcOp = cast<FuncOp>(op);
  } else {
    funcOp = cast<FuncOp>(op->getParentOp());
  }
  auto &block = funcOp.front();
  auto &topOp = block.front();
  if (auto noneOp = dyn_cast<front::NoneOp>(topOp)) {
    return noneOp;
  }
  auto ctx = op->getContext();
  auto builder = OpBuilder(ctx);
  builder.setInsertionPointToStart(&block);
  auto NoneOp = builder.create<front::NoneOp>(builder.getUnknownLoc(),
                                              builder.getNoneType());
  return NoneOp;
}

int getNumUsers(Value v) {
  if (!v) {
    return 0;
  }
  return std::distance(v.getUsers().begin(), v.getUsers().end());
}

static ModuleOp getModuleOp(Value v) {
  auto parent_op = v.getParentBlock()->getParentOp();
  while (parent_op != nullptr && !isa<ModuleOp>(parent_op)) {
    parent_op = parent_op->getParentOp();
  }
  if (parent_op == nullptr) {
    return nullptr;
  }
  return cast<ModuleOp>(parent_op);
}

ModuleOp getModuleOp(Operation *op) {
  while (op != nullptr && !isa<ModuleOp>(op)) {
    op = op->getParentOp();
  }
  if (op == nullptr) {
    return nullptr;
  }
  return cast<ModuleOp>(op);
}

Value getOriValue(Value v) {
  auto s = getModuleOp(v);
  if (!s) {
    return v;
  }
  if (auto block_arg = mlir::dyn_cast_or_null<BlockArgument>(v)) {
    int idx = block_arg.getArgNumber();
    // blockargument have multi-layers nest.
    FuncOp func_op;
    if (isa<FuncOp>(v.getParentBlock()->getParentOp()))
      func_op = cast<FuncOp>(v.getParentBlock()->getParentOp());
    else if (mlir::isa<front::LoopOp, front::IfOp>(
                 v.getParentBlock()->getParentOp())) {
      return getOriValue(v.getParentBlock()->getParentOp()->getOperand(idx));
    } else
      func_op = v.getParentBlock()->getParentOp()->getParentOfType<FuncOp>();

    if (func_op) {
      // cur call op
      auto call_op = getCallOp(func_op);
      // pre call op
      auto operand = call_op.getOperand(idx);
      if (mlir::isa<BlockArgument>(operand)) {
        auto find_root = [](auto &&Me, Value v) -> Value {
          if (mlir::isa<BlockArgument>(v)) {
            int index = mlir::dyn_cast<BlockArgument>(v).getArgNumber();
            FuncOp func_op;
            if (isa<FuncOp>(v.getParentBlock()->getParentOp()))
              func_op = cast<FuncOp>(v.getParentBlock()->getParentOp());
            else
              func_op =
                  v.getParentBlock()->getParentOp()->getParentOfType<FuncOp>();
            auto call_op = getCallOp(func_op);
            return Me(Me, call_op.getOperand(index));
          } else {
            return v;
          }
        };

        Value src_v = find_root(find_root, operand);
        return src_v;
      }
      auto result = mlir::cast<OpResult>(operand);
      auto opd = result.getDefiningOp();
      if (isa<front::InputOp>(opd)) {
        return operand;
      }
      auto pre_call_op = mlir::dyn_cast<func::CallOp>(opd);
      auto pre_func_op = getFuncOp(s, pre_call_op.getCallee());
      auto return_op = mlir::dyn_cast<ReturnOp>(pre_func_op.front().back());
      return return_op.getOperand(result.getResultNumber());
    }
  } else if (auto pre_op = v.getDefiningOp()) {
    if (isa<func::CallOp>(pre_op)) {
      auto call_op = mlir::dyn_cast<func::CallOp>(pre_op);
      int index = mlir::cast<OpResult>(v).getResultNumber();
      for (auto func : s.getOps<FuncOp>()) {
        if (call_op.getCallee() == func.getName()) {
          Block &entryBlock = func.front();
          auto returnOp =
              mlir::dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
          return returnOp->getOperand(index);
        }
      }
    } else {
      return v;
    }
  }

  llvm_unreachable("Failed to get preOperation.FIx me");
}

Operation *getNextOp(Operation *op, int i) {
  Operation *nextOp = nullptr;
  if (op->getResult(i).hasOneUse()) {
    for (auto &use : op->getResult(i).getUses()) {
      nextOp = use.getOwner();
      break;
    }
    assert(nextOp && "nextOp is nullptr");
  } else {
    auto users = op->getUsers();
    if (1 == std::distance(users.begin(), users.end())) {
      nextOp = *users.begin();
    }
  }
  // if not found, will return NULL
  return nextOp;
}

Value getOperand(Operation *op, int i) {
  auto v = op->getOperand(i);
  return getOriValue(v);
}

static void updateModuleTypes(ModuleOp s) {
  Builder builder(ctx);
  // update callee func's return types
  for (auto func : s.getOps<FuncOp>()) {
    if (func.getName() == "main") {
      continue;
    }
    std::vector<Type> returns;

    auto fnType = builder.getFunctionType(func.getArgumentTypes(),
                                          llvm::ArrayRef<Type>{returns});
    func.setType(fnType);
    auto callee = getCallOp(func);
    if (callee) {
      for (auto it : llvm::zip(callee.getResults(), returns)) {
        std::get<0>(it).setType(std::get<1>(it));
      }
    }
  }
  // update callee arg types
  for (auto func : s.getOps<FuncOp>()) {
    if (func.getName() == "main") {
      continue;
    }
    auto callee = getCallOp(func);
    if (!callee) {
      continue;
    }
    std::vector<Type> arguments;
    for (auto it :
         llvm::zip(callee.getOperandTypes(), func.front().getArguments())) {
      arguments.push_back(std::get<0>(it));
      std::get<1>(it).setType(std::get<0>(it));
    }
    auto fnType = builder.getFunctionType(llvm::ArrayRef<Type>(arguments),
                                          func.getResultTypes());
    func.setType(fnType);
  }
  // update main op return types
  auto mainFunc = getMainFuncOp(s);
  Block &entryBlock = mainFunc.front();
  auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
  std::vector<Type> returns;
  for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
    returns.push_back(returnOp->getOperand(i).getType());
  }
  std::vector<Type> inputs;
  auto args = mainFunc.getArguments();
  for (auto arg : args) {
    inputs.push_back(arg.getType());
  }
  auto fnType = builder.getFunctionType(llvm::ArrayRef<Type>{inputs},
                                        llvm::ArrayRef<Type>{returns});
  mainFunc.setType(fnType);
}

void updateModuleTypes() {
  auto modules = getAllModules();
  for (auto s : *modules) {
    updateModuleTypes(s);
  }
}

static void removeUnusedOp(ModuleOp submodule) {
  std::vector<Operation *> all_ops;
  for (auto func : submodule.getOps<FuncOp>()) {
    // for to support nested region's op
    func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (!isa<ReturnOp, FuncOp, front::YieldOp, front::InputOp>(op))
        all_ops.push_back(op);
    });
  }
  for (auto iter = all_ops.rbegin(); iter != all_ops.rend(); iter++) {

    auto op = *iter;
    if (op->hasAttrOfType<mlir::StringAttr>("placeholder") &&
        op->getAttrOfType<mlir::StringAttr>("placeholder").getValue() !=
            "None") {
      continue;
    }
    if (op->use_empty()) {
      op->erase();
    }
  }
}

void removeUnusedOp() {
  auto modules = getAllModules();
  for (auto s : *modules) {
    removeUnusedOp(s);
  }
}

int64_t getAddress(Value v) {
  if (mlir::isa<NoneType>(v.getType())) {
    return 0;
  }
  auto attr = mlir::cast<RankedTensorType>(v.getType()).getEncoding();
  if (attr) {
    return mlir::cast<IntegerAttr>(attr).getInt();
  }
  if (auto block_arg = mlir::dyn_cast_or_null<BlockArgument>(v)) {
    int index = block_arg.getArgNumber();
    auto parent_op = v.getParentBlock()->getParentOp();
    FuncOp funcOp;

    if (mlir::isa<FuncOp>(parent_op))
      funcOp = mlir::cast<FuncOp>(parent_op);
    else
      funcOp = parent_op->getParentOfType<FuncOp>();

    if (funcOp) {
      func::CallOp callee = getCallOp(funcOp);
      return getAddress(callee.getOperand(index));
    }
  }
  return 0;
}

void setAddress(Value v, int64_t addr) {
  auto type = mlir::cast<RankedTensorType>(v.getType());
  Builder builder(v.getContext());
  auto addrAttr = builder.getI64IntegerAttr(addr);
  auto new_type =
      RankedTensorType::get(type.getShape(), type.getElementType(), addrAttr);
  v.setType(new_type);
}

int64_t getRealBytes(Value v) {
  if (mlir::isa<NoneType>(v.getType())) {
    return 0;
  }
  auto type = mlir::cast<RankedTensorType>(v.getType());
  auto elm_count = type.getNumElements();
  auto etype = getStorageType(v);
  int elm_bits = etype.getIntOrFloatBitWidth();
  return align_up(elm_count * elm_bits, (int64_t)8) / 8;
}

size_t getBytes(Value v) {
  auto elm_bytes = getRealBytes(v);
  // ASSERT_OP(elm_bytes <= 0xFFFFFFFF, v.getDefiningOp());
  return (size_t)elm_bytes;
}

double getDtypeSize(Value v) {
  auto etype = getStorageType(v);
  double elm_bytes = (double)etype.getIntOrFloatBitWidth() / 8;
  return elm_bytes;
}

int64_t getNumElements(Value v) {
  if (mlir::isa<RankedTensorType>(v.getType()) == false) {
    return 0;
  }
  auto type = mlir::cast<RankedTensorType>(v.getType());
  return type.getNumElements();
}

llvm::ArrayRef<int64_t> getShape(Value v) {
  if (mlir::isa<NoneType>(v.getType())) {
    v.dump();
    llvm_unreachable("v is none type");
  }
  if (!isUnranked(v)) {
    auto type = mlir::cast<RankedTensorType>(v.getType());
    return type.getShape();
  } else {
    return mlir::cast<UnrankedTensorType>(v.getType()).getShape();
  }
}

std::vector<int64_t> getShapeVec(Value v) {
  llvm::ArrayRef<int64_t> shape = getShape(v);
  std::vector<int64_t> shapeV(shape.begin(), shape.end());
  return shapeV;
}

void setShape(Value v, llvm::ArrayRef<int64_t> shape) {
  auto newType = RankedTensorType::get(shape, getElementType(v));
  v.setType(newType);
}

void getGlobalShape(Value v, int *shape, int dim) {
  for (auto v : llvm::enumerate(getShape(v)))
    shape[v.index()] = (int)v.value();
  for (int i = getShape(v).size(); i < dim; ++i)
    shape[i] = 1;
}

void get128BtyeAlignedStrideForNBit(int *stride, int *shape, int npu_num,
                                    int bit) {
  assert(bit == 8 || bit == 16 || bit == 32);
  int aligned_bit;
  switch (bit) {
  case 8:
    aligned_bit = 128;
    break;
  case 16:
    aligned_bit = 64;
    break;
  case 32:
    aligned_bit = 32;
    break;
  }
  const int cstride = align_up(shape[3] * shape[2], aligned_bit);
  stride[0] = (int)std::ceil((double)shape[1] / npu_num) * cstride;
  stride[1] = cstride;
  stride[2] = shape[3];
  stride[3] = 1;
}

void getCompactStride(int *stride, int *shape, int npu_num) {
  const int cstride = shape[2] * shape[3];
  stride[0] = (int)std::ceil((double)shape[1] / npu_num) * cstride;
  stride[1] = cstride;
  stride[2] = shape[3];
  stride[3] = 1;
}

void getContinousStride(int *stride, int *shape) {
  stride[3] = 1;
  stride[2] = shape[3];
  stride[1] = shape[3] * shape[2];
  stride[0] = stride[1] * shape[1];
}

i32_array_t getI32Array(ArrayAttr arrayAttr) {
  auto data = std::make_shared<std::vector<int32_t>>();
  if (!arrayAttr) {
    return data;
  }
  for (auto en : llvm::enumerate(arrayAttr)) {
    auto attr = mlir::dyn_cast<IntegerAttr>(en.value());
    if (attr) {
      data->push_back(attr.getInt());
    } else {
      arrayAttr.dump();
      llvm_unreachable("not int32_t type");
    }
  }
  return std::move(data);
}

i32_array_t getI32Array(std::optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        int32_t default_value) {
  if (arrayAttr.has_value()) {
    auto arr = getI32Array(arrayAttr.value());
    assert(arr->size() == num_elem);
    return std::move(arr);
  }
  return std::make_shared<std::vector<int32_t>>(num_elem, default_value);
}

i64_array_t getI64Array(ArrayAttr arrayAttr) {
  auto data = std::make_shared<std::vector<int64_t>>();
  if (!arrayAttr) {
    return data;
  }
  for (auto en : llvm::enumerate(arrayAttr)) {
    auto attr = mlir::dyn_cast<IntegerAttr>(en.value());
    if (attr) {
      data->push_back(attr.getInt());
    } else {
      arrayAttr.dump();
      llvm_unreachable("not int64_t type");
    }
  }
  return std::move(data);
}

i64_array_t getI64Array(std::optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        int64_t default_value) {
  if (arrayAttr.has_value()) {
    auto arr = getI64Array(arrayAttr.value());
    assert(arr->size() == num_elem);
    return std::move(arr);
  }
  return std::make_shared<std::vector<int64_t>>(num_elem, default_value);
}

f64_array_t getF64Array(ArrayAttr arrayAttr) {
  auto data = std::make_shared<std::vector<double>>();
  if (!arrayAttr) {
    return data;
  }
  for (auto en : llvm::enumerate(arrayAttr)) {
    auto attr = mlir::dyn_cast<FloatAttr>(en.value());
    data->push_back(attr.getValueAsDouble());
  }
  return std::move(data);
}

f64_array_t getF64Array(std::optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        double default_value) {
  if (arrayAttr.has_value()) {
    auto arr = getF64Array(arrayAttr.value());
    assert(arr->size() == num_elem);
    return std::move(arr);
  }
  return std::make_shared<std::vector<double>>(num_elem, default_value);
}

Type getStorageType(Type type) {
  if (mlir::isa<RankedTensorType>(type)) {
    type = mlir::cast<RankedTensorType>(type).getElementType();
  }
  if (auto qType = mlir::dyn_cast<quant::CalibratedQuantizedType>(type)) {
    return qType.getExpressedType();
  } else if (auto qType = mlir::dyn_cast<quant::UniformQuantizedType>(type)) {
    auto stype = qType.getStorageType();
    bool isSign = qType.isSigned();
    if (stype.isSignlessInteger()) {
      auto bits = stype.getIntOrFloatBitWidth();
      auto sign = isSign ? IntegerType::Signed : IntegerType::Unsigned;
      return IntegerType::get(type.getContext(), bits, sign);
    }
    return stype;
  } else if (auto qType =
                 mlir::dyn_cast<quant::UniformQuantizedPerAxisType>(type)) {
    return qType.getStorageType();
  }
  return type;
}

Type getStorageType(Value v) { return getStorageType(v.getType()); }

Type getElementType(Value v) {
  auto type = v.getType();
  if (mlir::isa<RankedTensorType>(type)) {
    auto rtype = mlir::cast<RankedTensorType>(type);
    return rtype.getElementType();
  } else if (mlir::isa<UnrankedTensorType>(type)) {
    auto rtype = mlir::cast<UnrankedTensorType>(type);
    return rtype.getElementType();
  }
  return type;
}

RankedTensorType getTypeLike(Value v, llvm::ArrayRef<int64_t> shape) {
  return RankedTensorType::get(shape, getElementType(v));
}

static void getNCHW_align_right(llvm::ArrayRef<int64_t> &shape, int64_t &n,
                                int64_t &c, int64_t &h, int64_t &w) {
  int num_dims = shape.size();
  n = 1, c = 1, h = 1, w = 1;
  if (num_dims > 0) {
    w = shape[num_dims - 1];
  }
  if (num_dims > 1) {
    h = shape[num_dims - 2];
  }
  if (num_dims > 2) {
    c = shape[num_dims - 3];
  }
  if (num_dims > 3) {
    n = shape[num_dims - 4];
  }
  for (int i = 4; i < num_dims; i++) {
    n *= shape[num_dims - i - 1];
  }
}

static void getNCHW_align_left(llvm::ArrayRef<int64_t> shape, int64_t &n,
                               int64_t &c, int64_t &h, int64_t &w) {
  int num_dims = shape.size();
  n = 1, c = 1, h = 1, w = 1;
  if (num_dims > 0) {
    n = shape[0];
  }
  if (num_dims > 1) {
    c = shape[1];
  }
  if (num_dims > 2) {
    h = shape[2];
  }
  for (size_t i = 3; i < num_dims; ++i) {
    w *= shape[i];
  }
}

void getNCHW(llvm::ArrayRef<int64_t> shape, int64_t &n, int64_t &c, int64_t &h,
             int64_t &w, bool left_align) {
  if (left_align) {
    getNCHW_align_left(shape, n, c, h, w);
  } else {
    getNCHW_align_right(shape, n, c, h, w);
  }
}

void getNCHW(Value v, int64_t &n, int64_t &c, int64_t &h, int64_t &w,
             bool left_align) {
  auto shape = mlir::cast<RankedTensorType>(v.getType()).getShape();
  getNCHW(shape, n, c, h, w, left_align);
}

void getNCHW(llvm::ArrayRef<int64_t> shape, int64_t &n, int64_t &c, int64_t &h,
             int64_t &w, group_type_t group_type, bool IsHdimIsBatch) {
  module::getNCHW(shape, n, c, h, w, true);
}

void getNCHW(Value v, int64_t &n, int64_t &c, int64_t &h, int64_t &w,
             group_type_t group_type) {
  auto shape = mlir::cast<RankedTensorType>(v.getType()).getShape();
  getNCHW(shape, n, c, h, w, group_type);
}

void getNCDHW(Value v, int64_t &n, int64_t &c, int64_t &d, int64_t &h,
              int64_t &w, group_type_t group_type) {
  d = 1;
  auto shape = mlir::cast<RankedTensorType>(v.getType()).getShape();
  int num_dims = shape.size();
  if (GROUP_3D == group_type) {
    n = num_dims > 0 ? shape[0] : 1;
    c = num_dims > 1 ? shape[1] : 1;
    d = num_dims > 4 ? shape[2] : 1;
    h = num_dims > 4 ? shape[3] : (num_dims > 2 ? shape[2] : 1);
    w = 1;
    for (int i = (num_dims > 4) ? 4 : 3; i < num_dims; i++)
      w *= shape[i];
    return;
  } else if (GROUP_MM_INT4 == group_type) {
    assert(num_dims == 2);
    n = shape[0];
    c = 1;
    h = shape[1];
    w = 1;
  } else {
    getNCHW(shape, n, c, h, w, group_type, false); // TODO
  }
}

bool isValueBlockArgument(Value v) {
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    return true;
  }
  return false;
}

bool isOpInBlock(Operation *op) {
  if (op == nullptr) {
    return false;
  }
  auto parent = op->getParentOp();
  if (parent == nullptr) {
    return false;
  }
  if (isa<func::FuncOp>(parent)) {
    return false;
  }
  return true;
}

bool isOpBlockReturnOp(Operation *op) {
  if (op == nullptr) {
    return false;
  }
  auto block = op->getBlock();
  if (block == nullptr) {
    return false;
  }
  for (auto in : block->getTerminator()->getOperands()) {
    if (module::isSameOp(in.getDefiningOp(), op)) {
      return true;
    }
  }
  return false;
}

FuncOp getFuncOp(ModuleOp mod, StringRef func_name) {
  for (auto func : mod.getOps<FuncOp>()) {
    if (func.getName() == func_name) {
      return func;
    }
  }
  llvm::errs() << "Can't find FuncOp:" << func_name << "\n";
  llvm_unreachable("Error getFuncOp !!\n");
  return nullptr;
}

func::CallOp getCallOp(FuncOp func) {
  auto parent = func->getParentOp();
  auto s = cast<ModuleOp>(parent);
  func::CallOp call = nullptr;
  for (auto each_func : s.getOps<FuncOp>()) {
    WalkResult result =
        each_func.walk<WalkOrder::PreOrder>([&](func::CallOp op) {
          if (!call && op.getCallee() == func.getName()) {
            call = op;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted())
      break;
  }
  return call;
}

FuncOp getMainFuncOp(ModuleOp module) { return getFuncOp(module, "main"); }

bool isSign(Value v) {
  auto stype = getStorageType(v);
  if (stype.isUnsignedInteger()) {
    return false;
  }
  return true;
}

bool isWeight(Value v) {
  auto op = v.getDefiningOp();
  if (op == nullptr) {
    return false;
  }
  if (isa<front::WeightOp>(op)) {
    return true;
  }

  return false;
}

bool isActive(Value v) {
  if (module::isNone(v) || module::isWeight(v)) {
    return false;
  }
  return true;
}

bool isDynWeight(Value v) {
  auto op = v.getDefiningOp();
  if (op == nullptr) {
    return false;
  }
  if (op->hasAttr("dynamic_weight")) {
    // use code below to tag dynamic weight op
    // op->setAttr("dynamic_weight", , rewriter.getBoolAttr(true));
    return true;
  }
  return false;
}

bool isShapeRelatedOp(Value v) {
  auto op = v.getDefiningOp();
  if (op == nullptr) {
    return false;
  }
  if (isa<front::ShapeOp>(op)) {
    return true;
  }
  return false;
}

bool isAllWeight(Operation *op) {
  for (auto in : op->getOperands()) {
    if (isNone(in) || isWeight(in)) {
      continue;
    }
    return false;
  }
  return true;
}

bool isNone(Value v) { return mlir::isa<mlir::NoneType>(v.getType()); }

bool isUnranked(Value v) {
  return mlir::isa<mlir::UnrankedTensorType>(v.getType());
}

bool isDynamicShape(Value v) {
  int ret = false;
  auto tensorTy = mlir::dyn_cast<RankedTensorType>(v.getType());
  if (tensorTy) {
    for (int64_t dim : tensorTy.getShape()) {
      if (ShapedType::isDynamic(dim) || dim == 0)
        ret = true;
    }
  }
  return ret;
}

void setShapeOrVerify(Value v, llvm::ArrayRef<int64_t> shape) {
  if (mlir::isa<NoneType>(v.getType())) {
    return;
  }
  if (isUnranked(v) || isDynamicShape(v)) {
    auto newType = RankedTensorType::get(shape, getElementType(v));
    v.setType(newType);
  } else {
    auto s = getShape(v);
    /* unranked tensor is okay, for example:
       tensor<*xf32>->tensor<1xf32> */
    if ((std::max(s.size(), shape.size()) > 1) && s != shape) {
      v.dump();
      llvm::errs() << "error infer shape:";
      for (auto it : shape)
        llvm::errs() << " " << it;
      llvm::errs() << "\n";
      llvm_unreachable("Shape Verify failed");
    }
  }
}

bool isTypeIndependent(Operation *op) {
  return (isa<front::PermuteOp, front::ReshapeOp, front::SliceOp, front::TileOp,
              front::ConcatOp>(op));
}

int64_t getCoeffSize(ModuleOp s) {
  return s->getAttrOfType<IntegerAttr>(Attr::COEFF_SIZE).getInt();
}

void setCoeffSize(ModuleOp s, int64_t size) {
  s->setAttr(Attr::COEFF_SIZE, Builder(ctx).getI64IntegerAttr(size));
}

int64_t getDynamicOffset(ModuleOp s) {
  return s->getAttrOfType<IntegerAttr>(Attr::DYNAMIC_COEFF_OFFSET).getInt();
}

void setDynamicOffset(ModuleOp s, int64_t size) {
  s->setAttr(Attr::DYNAMIC_COEFF_OFFSET, Builder(ctx).getI64IntegerAttr(size));
}

int64_t getGmemPrivateSize(ModuleOp s) {
  return s->getAttrOfType<IntegerAttr>(Attr::GMEM_PRIVATE_SIZE).getInt();
}

void setGmemPrivateSize(ModuleOp s, int64_t size) {
  s->setAttr(Attr::GMEM_PRIVATE_SIZE, Builder(ctx).getI64IntegerAttr(size));
}

int64_t getCoreNum() {
  if (auto cores = m->getAttrOfType<IntegerAttr>(Attr::CORES))
    return cores.getInt();
  return 1;
}

void setCoreNum(int64_t core_num) {
  m->setAttr(Attr::CORES, Builder(ctx).getI64IntegerAttr(core_num));
}

int64_t getDeviceNum() {
  if (auto devices = m->getAttrOfType<IntegerAttr>(Attr::DEVICES)) {
    return devices.getInt();
  }
  return 1;
}

void setDeviceNum(int64_t device_num) {
  m->setAttr(Attr::DEVICES, Builder(ctx).getI64IntegerAttr(device_num));
}

int64_t getCoeffAddr(ModuleOp s) {
  return s->getAttrOfType<IntegerAttr>(Attr::COEFF_ADDR).getInt();
}

void setCoeffAddr(ModuleOp s, int64_t addr) {
  s->setAttr(Attr::COEFF_ADDR, Builder(ctx).getI64IntegerAttr(addr));
}

int64_t getNeuronSize(ModuleOp s) {
  return s->getAttrOfType<IntegerAttr>(Attr::NEURON_SIZE).getInt();
}

void setNeuronSize(ModuleOp s, int64_t size) {
  s->setAttr(Attr::NEURON_SIZE, Builder(ctx).getI64IntegerAttr(size));
}

int64_t getNeuronAddr(ModuleOp s) {
  return s->getAttrOfType<IntegerAttr>(Attr::NEURON_ADDR).getInt();
}

void setNeuronAddr(ModuleOp s, int64_t addr) {
  s->setAttr(Attr::NEURON_ADDR, Builder(ctx).getI64IntegerAttr(addr));
}

int64_t getIOSize(ModuleOp s) {
  if (s->hasAttrOfType<IntegerAttr>(Attr::IO_SIZE)) {
    return s->getAttrOfType<IntegerAttr>(Attr::IO_SIZE).getInt();
  }
  return 0;
}

void setIOSize(ModuleOp s, int64_t size) {
  s->setAttr(Attr::IO_SIZE, Builder(ctx).getI64IntegerAttr(size));
}

int64_t getIOAddr(ModuleOp s) {
  if (s->hasAttrOfType<IntegerAttr>(Attr::IO_ADDR)) {
    return s->getAttrOfType<IntegerAttr>(Attr::IO_ADDR).getInt();
  }
  return 0;
}

void setIOAddr(ModuleOp s, int64_t addr) {
  s->setAttr(Attr::IO_ADDR, Builder(ctx).getI64IntegerAttr(addr));
}

llvm::StringRef getPostprocess() {
  if (m->hasAttrOfType<StringAttr>(Attr::POSTPROCESS)) {
    return m->getAttrOfType<StringAttr>(Attr::POSTPROCESS).strref();
  }
  return llvm::StringRef("");
}

void setPostprocess(StringRef post) {
  m->setAttr(Attr::POSTPROCESS, Builder(ctx).getStringAttr(post));
}

Chip getChip() { return chip; }

StringRef getChipStr() { return stringifyChip(chip); }

Mode getMode() {
  if (false == m->hasAttrOfType<StringAttr>(Attr::MODE)) {
    return Mode::F32;
  }
  auto s = m->getAttrOfType<StringAttr>(Attr::MODE);
  return symbolizeMode(s).value_or(Mode::F32);
}

std::string toLower(llvm::StringRef str) {
  std::string lower = str.str();
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  return lower;
}

std::string getModeStr() { return toLower(stringifyMode(getMode())); }

bool getTrain() {
  if (m->hasAttrOfType<BoolAttr>(Attr::TRAIN)) {
    return m->getAttrOfType<BoolAttr>(Attr::TRAIN).getValue();
  }
  return false;
}

bool isBF16Modes() {
  auto s = m->getAttrOfType<StringAttr>(Attr::MODE);
  auto mode = symbolizeMode(s).value_or(Mode::F32);
  return mode == Mode::BF16 || mode == Mode::W8BF16 || mode == Mode::W4BF16 ||
         mode == Mode::INT8BF16DYN || mode == Mode::INT4BF16DYN ||
         mode == Mode::F8E4M3BF16DYN || mode == Mode::F4BF16DYN ||
         mode == Mode::MXF4BF16DYN;
}

bool isF16Modes() {
  auto s = m->getAttrOfType<StringAttr>(Attr::MODE);
  auto mode = symbolizeMode(s).value_or(Mode::F32);
  return mode == Mode::F16 || mode == Mode::W8F16 || mode == Mode::W4F16 ||
         mode == Mode::INT8F16DYN || mode == Mode::INT4F16DYN ||
         mode == Mode::F8E4M3F16DYN || mode == Mode::F4F16DYN ||
         mode == Mode::MXF4F16DYN;
}

bool isF8Modes() {
  auto s = m->getAttrOfType<StringAttr>(Attr::MODE);
  auto mode = symbolizeMode(s).value_or(Mode::F32);
  return mode == Mode::F8 || mode == Mode::F8E4M3 || mode == Mode::F8E5M2;
}

void setChip(Chip chip_) {
  chip = chip_;
  auto s = stringifyChip(chip_);
  m->setAttr(Attr::CHIP, StringAttr::get(m.getContext(), s));
}

bool isChip(Chip chip_) { return chip == chip_; }

void setMode(Mode mode) {
  auto s = stringifyMode(mode);
  m->setAttr(Attr::MODE, StringAttr::get(ctx, s));
}

int64_t getFLOPs() {
  if (!m->hasAttrOfType<IntegerAttr>(Attr::FLOPS)) {
    return 0;
  }
  return m->getAttrOfType<IntegerAttr>(Attr::FLOPS).getInt();
}

void setFLOPs(int64_t flops) {
  auto intType = IntegerType::get(ctx, 64);
  m->setAttr(Attr::FLOPS, IntegerAttr::get(intType, flops));
}

std::shared_ptr<std::vector<ModuleOp>> getAllModules() {
  auto modules = std::make_shared<std::vector<ModuleOp>>();
  auto sub = m.getOps<ModuleOp>();
  if (sub.empty()) {
    modules->push_back(m);
  } else {
    modules->assign(sub.begin(), sub.end());
  }
  return std::move(modules);
}

int getNumSubModule() {
  auto sub = m.getOps<ModuleOp>();
  return std::distance(sub.begin(), sub.end());
}

void setSubModuleId(ModuleOp sub, int64_t device_id, int64_t step) {
  sub->setAttr(Attr::DEVICE_ID,
               Builder(sub.getContext()).getI64IntegerAttr(device_id));
  sub->setAttr(Attr::STEP, Builder(sub.getContext()).getI64IntegerAttr(step));
}

void getSubModuleId(ModuleOp sub, int64_t &device_id, int64_t &step) {
  device_id = sub->getAttrOfType<IntegerAttr>(Attr::DEVICE_ID).getInt();
  step = sub->getAttrOfType<IntegerAttr>(Attr::STEP).getInt();
}

bool isAsymmetric() {
  if (m->hasAttrOfType<BoolAttr>(Attr::ASYMMETRIC)) {
    return m->getAttrOfType<BoolAttr>(Attr::ASYMMETRIC).getValue();
  }
  return false;
}

void setAsymmetric(bool is_asymmetric) {
  m->setAttr(Attr::ASYMMETRIC, BoolAttr::get(ctx, is_asymmetric));
}

bool isDynamicQuantize() {
  if (m->hasAttrOfType<StringAttr>(Attr::MODE)) {
    auto mode = m->getAttrOfType<StringAttr>(Attr::MODE).getValue();
    auto imode = symbolizeMode(mode);
    return (imode >= Mode::INT8F16DYN && imode <= Mode::MXF4BF16DYN);
  }
  return false;
}

int getQuantGroupSize() {
  if (m->hasAttrOfType<IntegerAttr>(Attr::QUANT_GROUP_SIZE)) {
    return m->getAttrOfType<IntegerAttr>(Attr::QUANT_GROUP_SIZE)
        .getValue()
        .getSExtValue();
  }
  return 0;
}

bool isQuantSymmetric() {
  if (m->hasAttrOfType<BoolAttr>(Attr::QUANT_SYMMETRIC)) {
    return m->getAttrOfType<BoolAttr>(Attr::QUANT_SYMMETRIC).getValue();
  }
  return false;
}

void setQuantGroupSize(int q_group_size) {
  auto intType = IntegerType::get(ctx, 64);
  m->setAttr(Attr::QUANT_GROUP_SIZE, IntegerAttr::get(intType, q_group_size));
}

void setGroupQuantInfo(int q_group_size, bool q_symmetric) {
  auto intType = IntegerType::get(ctx, 64);
  m->setAttr(Attr::QUANT_GROUP_SIZE, IntegerAttr::get(intType, q_group_size));
  m->setAttr(Attr::QUANT_SYMMETRIC, BoolAttr::get(ctx, q_symmetric));
}

bool isTrain() {
  if (m->hasAttrOfType<BoolAttr>(Attr::TRAIN)) {
    return m->getAttrOfType<BoolAttr>(Attr::TRAIN).getValue();
  }
  return false;
}

void setTrain(bool is_train) {
  m->setAttr(Attr::TRAIN, BoolAttr::get(ctx, is_train));
}

void setAddrMode(AddrMode mode) {
  auto s = stringifyAddrMode(mode);
  m->setAttr(Attr::ADDR_MODE, StringAttr::get(ctx, s));
}

AddrMode getAddrMode() {
  if (m->hasAttrOfType<StringAttr>(Attr::ADDR_MODE)) {
    auto s = m->getAttrOfType<StringAttr>(Attr::ADDR_MODE);
    return symbolizeAddrMode(s).value_or(AddrMode::BASIC);
  }
  return AddrMode::BASIC;
}

bool isAddrMode(AddrMode mode) { return mode == getAddrMode(); }

void setHighPrecision(bool is_high) {
  m->setAttr(Attr::HIGH_PRECISION, BoolAttr::get(ctx, is_high));
}

bool isHighPrecision() {
  if (m->hasAttrOfType<BoolAttr>(Attr::HIGH_PRECISION)) {
    return m->getAttrOfType<BoolAttr>(Attr::HIGH_PRECISION).getValue();
  }
  return false;
}

void setFrontRunMode(FrontRunMode mode) {
  auto s = stringifyFrontRunMode(mode);
  m->setAttr(Attr::FRONT_RUN_MODE, StringAttr::get(ctx, s));
}

FrontRunMode getFrontRunMode() {
  if (m->hasAttrOfType<StringAttr>(Attr::FRONT_RUN_MODE)) {
    auto s = m->getAttrOfType<StringAttr>(Attr::FRONT_RUN_MODE);
    return symbolizeFrontRunMode(s).value_or(FrontRunMode::STATIC);
  }
  return FrontRunMode::STATIC;
}

bool isDynamic() { return getFrontRunMode() == FrontRunMode::DYNAMIC; }

bool isDebugCmdEnable(std::string cmd_str) {
  if (debug_cmd.find(cmd_str) != std::string::npos) {
    return true;
  }
  return false;
}

State getState() {
  auto s = m->getAttrOfType<StringAttr>(Attr::STATE);
  return symbolizeState(s).value_or(State::FRONT_F32);
}

void setState(State state) {
  auto s = stringifyState(state);
  m->setAttr(Attr::STATE, StringAttr::get(ctx, s));
}

int64_t getLoraRank() {
  if (m->hasAttrOfType<IntegerAttr>(Attr::LORA_RANK)) {
    return m->getAttrOfType<IntegerAttr>(Attr::LORA_RANK).getInt();
  }
  return 0;
}

void setLoraRank(int64_t rank) {
  auto intType = IntegerType::get(ctx, 64);
  m->setAttr(Attr::LORA_RANK, IntegerAttr::get(intType, rank));
}

Platform getPlatform() { return platform; }

bool isPlatform(Platform plt) { return platform == plt; }

void setInputs(ArrayRef<StringRef> inputs) {
  m->setAttr(Attr::INPUTS, Builder(ctx).getStrArrayAttr(inputs));
}

std::shared_ptr<std::vector<StringRef>> getInputs() {
  auto inputs = m->getAttrOfType<ArrayAttr>(Attr::INPUTS);
  auto data = std::make_shared<std::vector<StringRef>>();
  for (auto en : llvm::enumerate(inputs)) {
    auto attr = mlir::dyn_cast<StringAttr>(en.value());
    data->push_back(attr.strref());
  }
  return std::move(data);
}

void setOutputs(ArrayRef<StringRef> outputs) {
  m->setAttr(Attr::OUTPUTS, Builder(ctx).getStrArrayAttr(outputs));
}

std::shared_ptr<std::vector<StringRef>> getOutputs() {
  auto outputs = m->getAttrOfType<ArrayAttr>(Attr::OUTPUTS);
  auto data = std::make_shared<std::vector<StringRef>>();
  for (auto en : llvm::enumerate(outputs)) {
    auto attr = mlir::dyn_cast<StringAttr>(en.value());
    data->push_back(attr.strref());
  }
  return std::move(data);
}

void removeAttr(mlir::Operation *op, std::string attr_name) {
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    if (attr.getName() != attr_name) {
      attrs.push_back(attr);
    }
  }
  op->setAttrs(attrs);
}

bool isState(State state) { return state == getState(); }

bool isInt4Op(Operation *op) {
  if (isa<front::ConvOp, front::MatMulOp>(op)) {
    if (auto convOp = dyn_cast<front::ConvOp>(op)) {
      if (convOp.parseParam().is_dw)
        return false;
    }

    return true;
  }

  return false;
}

ModuleOp getModuleOp() { return m; }

Location getLoc() { return m.getLoc(); }

MLIRContext *getCtx() { return ctx; }

double getThreshold(Value v) {
  auto type = getCalibratedType(v);
  return type.getMax();
}

uint32_t getIdx(Value v) {
  uint32_t idx = 0;
  if (auto r = mlir::dyn_cast<OpResult>(v)) {
    idx = r.getResultNumber();
  } else if (auto r = mlir::dyn_cast<BlockArgument>(v)) {
    idx = r.getArgNumber();
  } else {
    v.dump();
    llvm_unreachable("Not Implemented");
  }
  return idx;
}

void setLoc(Value v, NameLoc loc) {
  if (mlir::isa<NameLoc>(v.getLoc())) {
    v.setLoc(loc);
    return;
  }
  if (auto fuse_loc = mlir::dyn_cast<FusedLoc>(v.getLoc())) {
    std::vector<mlir::Location> locs = fuse_loc.getLocations();
    uint32_t idx = getIdx(v);
    locs[idx] = loc;
    auto new_loc = FusedLoc::get(v.getContext(), locs);
    v.setLoc(new_loc);
    return;
  }
  if (auto op = v.getDefiningOp()) {
    auto op_loc = op->getLoc();
    if (mlir::isa<NameLoc>(op_loc)) {
      op->setLoc(loc);
      return;
    }
    if (auto fuse_loc = mlir::dyn_cast<FusedLoc>(op->getLoc())) {
      std::vector<mlir::Location> locs = fuse_loc.getLocations();
      auto idx = getIdx(v);
      locs[idx] = loc;
      auto new_loc = FusedLoc::get(v.getContext(), locs);
      op->setLoc(new_loc);
      return;
    }
  }
  v.dump();
  llvm_unreachable("Not Implemented");
}

NameLoc getLoc(Value v) {
  if (auto loc = mlir::dyn_cast<NameLoc>(v.getLoc())) {
    return loc;
  } else if (auto fuse_loc = mlir::dyn_cast<FusedLoc>(v.getLoc())) {
    auto locs = fuse_loc.getLocations();
    uint32_t idx = getIdx(v);
    if (auto name_loc = mlir::dyn_cast<NameLoc>(locs[idx])) {
      return name_loc;
    }
  } else if (auto op = v.getDefiningOp()) {
    auto loc = op->getLoc();
    if (auto name_loc = mlir::dyn_cast<NameLoc>(loc)) {
      return name_loc;
    }
    if (auto fuse_loc = mlir::dyn_cast<FusedLoc>(loc)) {
      uint32_t idx = getIdx(v);
      auto locs = fuse_loc.getLocations();
      if (auto name_loc = mlir::dyn_cast<NameLoc>(locs[idx])) {
        return name_loc;
      }
    }
  }
  v.dump();
  llvm_unreachable("Not Implemented");
  return nullptr;
}

NameLoc getLocLike(Operation *op, llvm::StringRef suffix) {
  return getLocLike(op->getResult(0), suffix);
}

NameLoc getLocLike(Value v, llvm::StringRef suffix) {
  auto name = getName(v);
  auto new_name = name.str() + "_" + suffix.str();
  Builder builder(v.getContext());
  return NameLoc::get(builder.getStringAttr(new_name));
}

void setLocSuffix(Operation *op, llvm::StringRef suffix) {
  if (op->getNumResults() > 1) {
    std::vector<Location> locs;
    for (auto r : op->getResults()) {
      auto loc = getLocLike(r, suffix);
      locs.push_back(loc);
    }
    auto new_loc = FusedLoc::get(op->getContext(), locs);
    op->setLoc(new_loc);
  } else {
    auto loc = getLocLike(op->getResult(0), suffix);
    op->setLoc(loc);
  }
}

StringRef getName(Operation *op, int index) {
  if (auto module = dyn_cast<ModuleOp>(op)) {
    return module.getName().value_or("Unknown");
  }

  if (auto func = dyn_cast<FuncOp>(op)) {
    return func.getName();
  }
  if (isa<mlir::func::CallOp>(op)) {
    return "func.call";
  }
  if (isa<ReturnOp>(op)) {
    return "func.return";
  }
  if (isa<front::NoneOp>(op)) {
    return "NoneOp";
  }

  if (auto loc = mlir::dyn_cast<NameLoc>(op->getLoc())) {
    return loc.getName();
  }
  if (auto loc = mlir::dyn_cast<FusedLoc>(op->getLoc())) {
    auto locs = loc.getLocations();
    assert(index < locs.size());
    if (auto name_loc = mlir::dyn_cast<NameLoc>(locs[index])) {
      return name_loc.getName();
    }
  }
  op->print(llvm::errs(), OpPrintingFlags().useLocalScope().enableDebugInfo());
  llvm::errs() << "op has no name location!!!\n";
  op->dump();
  llvm_unreachable("op has no name location!!!");
  return "";
}

StringRef getName(Value v) {
  if (isa<NoneType>(v.getType())) {
    return "none";
  }
  return getLoc(v).getName().strref();
}

void getInputsOutputs(ModuleOp s, std::vector<Value> &inputs,
                      std::vector<Value> &outputs) {
  auto main_func = getMainFuncOp(s);
  auto args = main_func.front().getArguments();
  for (auto &arg : args) {
    for (auto user : arg.getUsers()) {
      if (auto op = dyn_cast<front::InputOp>(user)) {
        inputs.push_back(op.getOutput());
      } else {
        llvm_unreachable("arg should only be used for InputOp.");
        user->dump();
      }
    }
  }
  // main_func.walk([&](front::InputOp op) { inputs.push_back(op.getOutput());
  // });
  main_func.walk([&](ReturnOp op) {
    for (auto out : op.getOperands()) {
      auto result = mlir::cast<OpResult>(out);
      auto call_op = result.getDefiningOp<func::CallOp>();
      if (call_op) {
        auto func_op = getFuncOp(s, call_op.getCallee());
        auto return_op = dyn_cast<ReturnOp>(func_op.front().back());
        assert(return_op);
        outputs.push_back(return_op.getOperand(result.getResultNumber()));

        if (module::isDebugCmdEnable("dump_all_global_op_out")) {
          func_op.walk([&](Operation *op) {
            if (!isa<front::NoneOp, front::WeightOp>(op)) {
              for (auto v : op->getResults()) {
                outputs.push_back(v);
              }
            }
          });
        }
      } else {
        outputs.push_back(out);
      }
    }
  });
}

bool isSameOp(Operation *op0, Operation *op1) {
  if (op0 == nullptr || op1 == nullptr) {
    return false;
  }
  if (op0->getName() != op1->getName()) {
    return false;
  }
  if (op0->getNumOperands() != op1->getNumOperands()) {
    return false;
  }
  for (auto it : llvm::zip(op0->getOperands(), op1->getOperands())) {
    if (std::get<0>(it) != std::get<1>(it)) {
      return false;
    }
  }
  if (op0->getNumResults() != op1->getNumResults()) {
    return false;
  }
  for (auto it : llvm::zip(op0->getResultTypes(), op1->getResultTypes())) {
    if (std::get<0>(it) != std::get<1>(it)) {
      return false;
    }
  }
  if (false == op0->getAttrs().equals(op1->getAttrs())) {
    return false;
  }
  return true;
}

void getInputsOutputs(func::CallOp call, std::vector<Value> &inputs,
                      std::vector<Value> &outputs) {
  for (auto opd : call.getOperands()) {
    inputs.emplace_back(module::getOriValue(opd));
  }
  auto md = getModuleOp(call);
  auto func = getFuncOp(md, call.getCallee());
  func.walk([&](ReturnOp op) {
    for (auto output : op.getOperands()) {
      outputs.push_back(output);
    }
  });
  if (module::isDebugCmdEnable("dump_all_global_op_out")) {
    func.walk([&](Operation *op) {
      if (!isa<front::NoneOp, front::WeightOp>(op)) {
        for (auto v : op->getResults()) {
          outputs.push_back(v);
        }
      }
    });
  }
}

void getScaleAndZeroPoint(double rmin, double rmax, double &scale,
                          int64_t &zeroPoint, int bitwidth) {
  int qmin = rmin < 0 ? -128 : 0;
  int qmax = rmin < 0 ? 127 : 255;
  if (bitwidth == 4) {
    qmin = rmin < 0 ? -8 : 0;
    qmax = rmin < 0 ? 7 : 15;
  } else if (bitwidth == 16) {
    qmin = rmin < 0 ? -32768 : 0;
    qmax = rmin < 0 ? 32767 : 65535;
  }
  // Determine the scale.
  double qminDouble = qmin;
  double qmaxDouble = qmax;
  scale = (rmax - rmin) / (qmaxDouble - qminDouble);
  double zeroPointFromMin = qminDouble - rmin / scale;

  // Now nudge the zero point to be an integer.
  zeroPoint = round(zeroPointFromMin);
  if (zeroPointFromMin < qminDouble) {
    zeroPoint = qmin;
    scale = rmax / (qmaxDouble - zeroPoint);
  } else if (zeroPointFromMin > qmaxDouble) {
    zeroPoint = qmax;
    scale = rmin / (qminDouble - zeroPoint);
  }
}

double getScale(double threshold, bool sign, int bitwidth) {
  if (bitwidth == 8) {
    if (sign) {
      return threshold / 127.0;
    } else {
      return threshold / 255.0;
    }
  } else if (bitwidth == 4) {
    if (sign) {
      return threshold / 7.0;
    } else {
      return threshold / 15.0;
    }
  } else if (bitwidth == 16) {
    if (sign) {
      return threshold / 32767.0;
    } else {
      return threshold / 65535.0;
    }
  } else {
    llvm_unreachable("not support");
    return 0;
  }
}

void getScaleAndZeroPoint(Value v, double &scale, int64_t &zeropoint,
                          bool asymmetric, int bitwidth) {
  bool sign;
  getScaleAndZeroPoint(v, scale, zeropoint, sign, asymmetric, bitwidth);
}

void getScaleAndZeroPoint(Value v, double &scale, int64_t &zeropoint,
                          bool &sign, bool asymmetric, int bitwidth) {
  if (isCalibratedType(v)) {
    if (bitwidth == 8) {
      auto pre_op = v.getDefiningOp();
      if (module::isInt4Op(pre_op)) {
        if (auto convOp = dyn_cast<front::ConvOp>(pre_op)) {
          if (convOp.getOutInt8Scale().has_value()) {
            scale = convOp.getOutInt8Scale()
                        .value_or(APFloat(1.0))
                        .convertToDouble();
            zeropoint = int64_t(
                convOp.getOutInt8Zp().value_or(APFloat(0.0)).convertToDouble());
            return;
          }
        } else {
          if (auto matmulOp = dyn_cast<front::MatMulOp>(pre_op)) {
            // break; //todo matmul need support int4
            if (matmulOp.getOutInt8Scale().has_value()) {
              scale = matmulOp.getOutInt8Scale()
                          .value_or(APFloat(1.0))
                          .convertToDouble();
              zeropoint = int64_t(matmulOp.getOutInt8Zp()
                                      .value_or(APFloat(0.0))
                                      .convertToDouble());
              return;
            }
          }
        }
      }
    }

    auto qtype = getCalibratedType(v);
    auto max = qtype.getMax();
    auto min = qtype.getMin();
    sign = min < 0;
    if (asymmetric) {
      getScaleAndZeroPoint(min, max, scale, zeropoint, bitwidth);
    } else {
      zeropoint = 0;
      auto th = std::max(std::abs(max), std::abs(min));
      scale = getScale(th, sign, bitwidth);
      if (isa<front::CompareOp, front::CompareConstOp>(v.getDefiningOp())) {
        scale = 1.0;
      }
    }
  } else if (isUniformQuantized(v)) {
    auto qtype = getUniformQuantizedType(v);
    scale = qtype.getScale();
    zeropoint = qtype.getZeroPoint();
    sign = qtype.isSigned();
  } else if (auto ccop = dyn_cast<front::CompareConstOp>(v.getDefiningOp())) {
    if (ccop.getMode().str() == "And")
      llvm_unreachable("calibration info not set for compareconst And");
    scale = 1.0;
    zeropoint = 0;
    sign = 0;
  } else if (auto ccop = dyn_cast<front::CompareOp>(v.getDefiningOp())) {
    scale = 1.0;
    zeropoint = 0;
    sign = 0;
  } else {
    v.dump();
    llvm_unreachable("can't get scale and zeropoint");
  }
}

bool isScalar(mlir::Operation *op) {
  if (op->hasTrait<trait::ScalarProducer>()) {
    auto is_scalar = mlir::cast<BoolAttr>(op->getAttr("is_scalar")).getValue();
    return is_scalar;
  }
  return false;
}

bool isCalibratedType(Type type) {
  return mlir::isa<quant::CalibratedQuantizedType>(
      mlir::cast<RankedTensorType>(type).getElementType());
}

bool isCalibratedType(Value v) { return isCalibratedType(v.getType()); }

bool isUniformQuantized(Type type) {
  if (mlir::isa<RankedTensorType>(type) == false) {
    return false;
  }
  return mlir::isa<quant::UniformQuantizedType>(
      mlir::cast<RankedTensorType>(type).getElementType());
}

bool isUniformQuantized(Value v) { return isUniformQuantized(v.getType()); }

quant::CalibratedQuantizedType getCalibratedType(Value v) {
  return mlir::cast<quant::CalibratedQuantizedType>(
      mlir::cast<RankedTensorType>(v.getType()).getElementType());
}

quant::CalibratedQuantizedType getCalibratedType(Type t) {
  return mlir::cast<quant::CalibratedQuantizedType>(
      mlir::cast<RankedTensorType>(t).getElementType());
}

quant::UniformQuantizedType getUniformQuantizedType(Value v) {
  return mlir::cast<quant::UniformQuantizedType>(
      mlir::cast<RankedTensorType>(v.getType()).getElementType());
}

quant::UniformQuantizedType getUniformQuantizedType(Type t) {
  return mlir::cast<quant::UniformQuantizedType>(
      mlir::cast<RankedTensorType>(t).getElementType());
}

//-----------------------------------------------------------------
// Helper Functions for weight
//-----------------------------------------------------------------
static std::string genWeightFileName(bool &same_name) {
  auto name = getName(m);
  auto state = getState();
  auto chip_ = getChip();
  auto chip = stringifyChip(chip_);
  auto old_name = m->getAttrOfType<StringAttr>(Attr::WEIGHT_FILE).getValue();
  std::string file_name = name.lower() + std::string("_") +
                          stringifyState(state).lower() + std::string("_") +
                          chip.lower();
  if (!isChip(Chip::ALL)) {
    auto mode = getMode();
    std::string sym = "";
    if (mode == Mode::INT8) {
      sym = isAsymmetric() ? "_asym" : "_sym";
    }
    auto mode_ = stringifyMode(mode);
    file_name += std::string("_") + mode_.lower() + sym;
  }
  auto new_name = file_name + "_weight.npz";
  same_name = (old_name == new_name);
  if (same_name) {
    new_name = file_name + "_weight_fix.npz";
  }
  return new_name;
}

void saveWeight() {
  // check name conflict
  std::set<StringRef> all_names;
  auto modules = getAllModules();
  for (auto s : *modules) {
    for (auto func : s.getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
        if (mlir::dyn_cast<NameLoc>(op->getLoc()) && !module::isOpInBlock(op) &&
            !isa<func::ReturnOp, func::CallOp, func::FuncOp, front::InputOp>(
                op)) {
          auto name = module::getName(op);
          // if op have more than two regions, it can have the same op Name
          ASSERT_OP(all_names.find(name) == all_names.end(), op);
          all_names.insert(name);
        }
      });
    }
  }
  bool same_name = true;
  std::string filename_;
  if (weightFileName == "") {
    filename_ = module::genWeightFileName(same_name);
  } else {
    same_name = false;
    filename_ = weightFileName;
  }
  // weight remove unused in npz
  if (wFile == nullptr) {
    if (!same_name) {
      weightFile().save(filename_);
      setWeightFileAttr(filename_);
    }
    return;
  }
  if (wFile->changed() == false && same_name) {
    return;
  }
  std::set<StringRef> weight_names;
  for (auto s : *modules) {
    for (auto func : s.getOps<FuncOp>()) {
      func.walk([&](front::WeightOp op) {
        weight_names.insert(module::getName(op.getOperation()));
      });
    }
  }
  std::set<StringRef> npz_names;
  wFile->getAllNames(npz_names);
  std::set<StringRef> dif_names;
  for (auto name : npz_names) {
    if (weight_names.find(name) == weight_names.end()) {
      dif_names.insert(name);
    }
  }
  for (auto &name : dif_names) {
    (void)(wFile->deleteTensor(name));
  }
  if (wFile->changed() == false && same_name) {
    return;
  }
  wFile->save(filename_);
  setWeightFileAttr(filename_);
}

void setWeightFileName(const std::string &name) { weightFileName = name; }
void detachWeightFile() { wFile = nullptr; }
void setWeightFileAttr(const std::string &name) {
  m->setAttr(Attr::WEIGHT_FILE, StringAttr::get(ctx, name));
}
llvm::StringRef getWeightFileAttr() {
  return m->getAttrOfType<StringAttr>(Attr::WEIGHT_FILE).getValue();
}

mlir::TensorFile &weightFile() {
  if (wFile == nullptr) {
    auto name = getWeightFileAttr();
    wFile = std::make_unique<mlir::TensorFile>(name, false);
  }
  return *wFile;
}

//-----------------------------------------------------------------
// Helper for shape op inference
//-----------------------------------------------------------------
void ShapeHelper::bindShapeInfo(const Value &v,
                                const std::vector<int64_t> &shape) {
  _shape_info[v] = shape;
}

std::vector<int64_t> ShapeHelper::getShapeInfo(const Value &v) {
  return _shape_info.at(v);
}

bool ShapeHelper::isShape(const Value &v) {
  return _shape_info.find(v) != _shape_info.end();
}

void bindShapeTensorValue(const Value &v, const std::vector<int64_t> &shape) {
  ShapeHelper::getInstance().bindShapeInfo(v, shape);
}

std::vector<int64_t> getShapeTensorValue(const Value &v) {
  if (module::isWeight(v)) {
    auto weight_op = dyn_cast<front::WeightOp>(v.getDefiningOp());
    auto weight_data = weight_op.read_as_float();
    std::vector<int64_t> int_data;
    int_data.reserve(weight_data->size());
    for (float f : *weight_data) {
      int_data.push_back(static_cast<int64_t>(f));
    }
    return int_data;
  } else {
    return ShapeHelper::getInstance().getShapeInfo(v);
  }
}

bool isShape(const Value &v) { return ShapeHelper::getInstance().isShape(v); }

std::vector<int64_t>
commonShapeValInfer(mlir::Operation *op,
                    const std::vector<std::vector<int64_t>> &in_shapes_v,
                    const std::vector<int64_t> &out_shape) {
  // support scalar
  // assert(out_shape.size() == 1 || out_shape.size() == 0);
  // auto real_out_size = out_shape.size() == 0 ? 1 : out_shape[0];
  int64_t real_out_size = 1;
  for (auto dim : out_shape) {
    real_out_size *= dim;
  }
  InferenceParameter p;
  std::vector<std::vector<float_t>> input_datas;
  for (auto &in_shape_v : in_shapes_v) {
    std::vector<float_t> input_data(in_shape_v.size());
    std::transform(in_shape_v.begin(), in_shape_v.end(), input_data.begin(),
                   [](auto &i) { return static_cast<float_t>(i); });
    input_datas.push_back(input_data);
  }
  std::transform(input_datas.begin(), input_datas.end(),
                 std::back_inserter(p.inputs),
                 [](auto &i) { return i.data(); });
  std::vector<float_t> output_data(real_out_size);
  p.outputs.push_back(output_data.data());
  auto inf_op = dyn_cast<InferenceInterface>(op);
  assert(inf_op);
  (void)(inf_op.init(p));
  auto ret = inf_op.inference(p);
  assert(mlir::succeeded(ret));
  inf_op.deinit(p);
  std::vector<int64_t> output_shape_v(real_out_size);
  std::transform(output_data.begin(), output_data.end(), output_shape_v.begin(),
                 [](float_t i) { return static_cast<int64_t>(i); });
  return output_shape_v;
}

void assert_with_dump(bool cond, Operation *op, const char *info,
                      const char *file, unsigned line) {
  if (cond) {
    return;
  }
  unreachable(info, op, file, line);
}

void unreachable(const char *info, Operation *op, const char *file,
                 unsigned line) {
  if (op != nullptr) {
    auto inputs = op->getOperands();
    if (!inputs.empty()) {
      for (auto input : inputs) {
        input.dump();
      }
    }
    std::cerr << "-> ";
    op->dump();
    for (auto out : op->getResults()) {
      for (auto user : out.getUsers())
        user->dump();
    }
  }
  std::cerr << "ASSERT executed at" << file << ":" << line << std::endl;
  std::cerr << "ASSERT INFO:" << info << std::endl << "Operation:" << std::endl;
  exit(-1);
}

bool startsWith(const std::string &fullString,
                const std::string &startingSubstring) {
  if (fullString.length() >= startingSubstring.length()) {
    return (0 == fullString.compare(0, startingSubstring.length(),
                                    startingSubstring));
  } else {
    return false;
  }
}

bool endsWith(const std::string &fullString, const std::string &suffix) {
  return fullString.rfind(suffix) == fullString.length() - suffix.length();
}

bool isOpSameCalc(Operation *op0, Operation *op1) {
  auto compare = [&](mlir::ValueRange left, mlir::ValueRange right) -> bool {
    for (auto it : llvm::zip(left, right)) {
      auto left = std::get<0>(it);
      auto right = std::get<1>(it);
      if (module::isNone(left) || module::isNone(right)) {
        continue;
      }
      auto l_s = module::getShape(left);
      auto r_s = module::getShape(right);
      if (l_s != r_s) {
        return false;
      }
    }
    return true;
  };
  if (op0 == op1) {
    // can't be the same op
    return false;
  }
  if (op0->getName() != op1->getName()) {
    return false;
  }
  if (false == compare(op0->getOperands(), op1->getOperands())) {
    return false;
  }
  if (false == compare(op0->getResults(), op1->getResults())) {
    return false;
  }
  return true;
}

bool isOpSameCalc(const std::vector<Operation *> &ops) {
  if (ops.size() < 2) {
    return false;
  }
  for (int i = 1; i < ops.size(); i++) {
    if (!isOpSameCalc(ops[0], ops[i])) {
      return false;
    }
  }
  return true;
}

bool areAttributesEqual(mlir::Operation *op1, mlir::Operation *op2) {
  auto attrs1 = op1->getAttrs();
  auto attrs2 = op2->getAttrs();

  if (attrs1.size() != attrs2.size()) {
    return false;
  }

  llvm::DenseMap<mlir::StringRef, mlir::Attribute> attrsMap;
  for (const auto &attr : attrs2) {
    attrsMap[attr.getName()] = attr.getValue();
  }

  for (const auto &attr : attrs1) {
    auto it = attrsMap.find(attr.getName());
    if (it == attrsMap.end() || it->second != attr.getValue()) {
      return false;
    }
  }
  return true;
}

} // namespace module
} // namespace im
