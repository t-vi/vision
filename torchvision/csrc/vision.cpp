#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/variable.h>
#include "torch/script.h"
#include "ROIAlign.h"
#include "ROIPool.h"
#include "nms.h"

#ifdef WITH_CUDA
#include <cuda.h>
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
#ifdef WITH_CUDA
  m.attr("CUDA_VERSION") = CUDA_VERSION;
#endif
}

using torch::Tensor;
using torch::autograd::variable_list;

struct ROIAlignBackward : public torch::autograd::TraceableFunction {
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ROIAlignBackward";
  }
  void release_variables() override {
    rois_.reset_data();
    rois_.reset_grad_function();
  }
  torch::autograd::SavedVariable rois_;
  double spatial_scale;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t sampling_ratio;
  int64_t batch_size, channels, height, width;
};

variable_list ROIAlignBackward::apply(variable_list&& grads) {
  variable_list grad_inputs(1);
  auto& grad = grads[0];
  auto rois = rois_.unpack();
  if (should_compute_output(0)) {
    grad_inputs[0] = ROIAlign_backward(
        grad,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        batch_size,
        channels,
        height,
        width,
        sampling_ratio);
  }
  return grad_inputs;
}

Tensor roi_align(
    const Tensor& input,
    const Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t sampling_ratio) {
  TORCH_CHECK(input.defined() && input.is_variable(), "invalid argument input");
  TORCH_CHECK(rois.defined() && rois.is_variable(), "invalid argument rois");
  // we might error out if rois requires grad...
  auto& input_ = torch::autograd::as_variable_ref(input);
  auto& rois_ = torch::autograd::as_variable_ref(rois);
  std::shared_ptr<ROIAlignBackward> grad_fn;
  if (torch::autograd::compute_requires_grad(input, rois)) {
    grad_fn = std::shared_ptr<ROIAlignBackward>(
        new ROIAlignBackward(), torch::autograd::deleteFunction);
    grad_fn->set_next_edges(torch::autograd::collect_next_edges(input));
    grad_fn->rois_ = torch::autograd::SavedVariable(rois, false);
    grad_fn->spatial_scale = spatial_scale;
    grad_fn->pooled_height = pooled_height;
    grad_fn->pooled_width = pooled_width;
    grad_fn->sampling_ratio = sampling_ratio;
    grad_fn->batch_size = input.size(0);
    grad_fn->channels = input.size(1);
    grad_fn->height = input.size(2);
    grad_fn->width = input.size(3);
  }
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return ROIAlign_forward(
        input_,
        rois_,
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio);
	      })();
  auto result = torch::autograd::as_variable(tmp);
  if (grad_fn) {
    set_history(torch::autograd::flatten_tensor_args(result), grad_fn);
  }
  return result;
}

static auto registry = torch::RegisterOperators()
  .op("torchvision::roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio) -> Tensor", &roi_align);
