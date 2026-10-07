

#pragma once

#include "Support/Dnnl/Binary.h"
#include "Support/Dnnl/Conv.h"
#include "Support/Dnnl/ConvBwd.h"
#include "Support/Dnnl/Deconv.h"
#include "Support/Dnnl/LRN.h"
#include "Support/Dnnl/MatMul.h"
#include "Support/Dnnl/PRelu.h"
#include "Support/Dnnl/Pool.h"
#include "mlir/IR/OpDefinition.h"
namespace im {

dnnl::memory::data_type getDnnlType(mlir::Value v);

}
