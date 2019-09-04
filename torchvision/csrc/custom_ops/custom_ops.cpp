#include <torch/script.h>

#include "nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"

using namespace at;

#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/variable.h>
#include "torch/script.h"

using torch::Tensor;
using torch::autograd::variable_list;

struct ROIAlignBackward : public torch::autograd::Node {
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
  // checks from VariableType::unpack
  TORCH_CHECK(input.defined() && input.is_variable(), "invalid argument input");
  TORCH_CHECK(rois.defined() && rois.is_variable(), "invalid argument rois");
  // we might error if rois requires grad...
  auto& input_ = torch::autograd::as_variable_ref(input);
  auto& rois_ = torch::autograd::as_variable_ref(rois);
  std::shared_ptr<ROIAlignBackward> grad_fn;
  if (torch::autograd::compute_requires_grad(input, rois)) {
    grad_fn = std::shared_ptr<ROIAlignBackward>(
        new ROIAlignBackward(), torch::autograd::deleteNode);
    grad_fn->set_next_edges(
        torch::autograd::collect_next_edges(input)); // note, only input!
    grad_fn->rois_ = torch::autograd::SavedVariable(rois, false);
    // extra bookkeeping
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

struct ROIPoolBackward : public torch::autograd::Node {
  variable_list apply(variable_list&& grads) override;
  std::string name() const override {
    return "ROIPoolBackward";
  }
  void release_variables() override {
    rois_.reset_data();
    rois_.reset_grad_function();
    argmax_.reset_data();
    argmax_.reset_grad_function();
  }
  torch::autograd::SavedVariable rois_;
  torch::autograd::SavedVariable argmax_;
  double spatial_scale;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t batch_size, channels, height, width;
};

variable_list ROIPoolBackward::apply(variable_list&& grads) {
  variable_list grad_inputs(1);
  auto& grad = grads[0];
  auto rois = rois_.unpack();
  auto argmax = argmax_.unpack();
  if (should_compute_output(0)) {
    grad_inputs[0] = ROIPool_backward(
        grad,
        rois,
        argmax,
        spatial_scale,
        pooled_height,
        pooled_width,
        batch_size,
        channels,
        height,
        width);
  }
  return grad_inputs;
}

at::Tensor roi_pool(
    const at::Tensor& input,
    const at::Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width) {
  // checks from VariableType::unpack
  TORCH_CHECK(input.defined() && input.is_variable(), "invalid argument input");
  TORCH_CHECK(rois.defined() && rois.is_variable(), "invalid argument rois");
  // we might error if rois requires grad...
  auto& input_ = torch::autograd::as_variable_ref(input);
  auto& rois_ = torch::autograd::as_variable_ref(rois);
  std::shared_ptr<ROIPoolBackward> grad_fn;
  if (torch::autograd::compute_requires_grad(input, rois)) {
    grad_fn = std::shared_ptr<ROIPoolBackward>(
        new ROIPoolBackward(), torch::autograd::deleteNode);
    grad_fn->set_next_edges(
        torch::autograd::collect_next_edges(input)); // note, only input!
    grad_fn->rois_ = torch::autograd::SavedVariable(rois, false);
    // extra bookkeeping
    grad_fn->spatial_scale = spatial_scale;
    grad_fn->pooled_height = pooled_height;
    grad_fn->pooled_width = pooled_width;
    grad_fn->batch_size = input.size(0);
    grad_fn->channels = input.size(1);
    grad_fn->height = input.size(2);
    grad_fn->width = input.size(3);
  }
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return ROIPool_forward(
        input_, rois_, spatial_scale, pooled_height, pooled_width);
  })();
  auto result = torch::autograd::as_variable(std::get<0>(tmp));
  if (grad_fn) {
    grad_fn->argmax_ = torch::autograd::SavedVariable(
        torch::autograd::as_variable(std::get<1>(tmp)), false);
    set_history(torch::autograd::flatten_tensor_args(result), grad_fn);
  }
  return result;
}

static auto registry =
    torch::jit::RegisterOperators()
        .op("torchvision::nms", &nms)
        .op("torchvision::roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio) -> Tensor",
            &roi_align)
        .op("torchvision::roi_pool", &roi_pool);
