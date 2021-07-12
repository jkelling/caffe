#ifdef USE_CUDNN
#include <vector>
#include "caffe/layers/cudnn_deconv_layer.hpp"

namespace caffe {

__global__ void sync_deconv_groups() {}

template <typename Dtype>
void CuDNNDeconvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  VLOG(1) << "  " << this->layer_param_.name() << " Forward GPU" << std::endl;
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    for (int n = 0; n < this->num_; ++n) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* top_data = top[i]->mutable_gpu_data();
      // Forward through cuDNN in parallel over groups.
      for (int g = 0; g < this->group_; g++) {
        // Filters.
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight + this->weight_offset_ * g,
              bottom_descs_[i], bottom_data + bottom_offset_ * g + n * this->bottom_dim_,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[g],
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              top_descs_[i], top_data + top_offset_ * g + n * this->top_dim_));
        // Bias.
        if (this->bias_term_) {
            const Dtype* bias_data = this->blobs_[1]->gpu_data();
            CUDNN_CHECK(cudnnAddTensor(handle_[g],
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g + n * this->top_dim_));
        }
      }
      // Synchronize the work across groups, each of which went into its own
      // stream, by launching an empty kernel into the default (null) stream.
      // NOLINT_NEXT_LINE(whitespace/operators)
      sync_deconv_groups<<<1, 1>>>();
    }
  }
}

template <typename Dtype>
void CuDNNDeconvolutionLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  VLOG(1) << "  " << this->layer_param_.name() << " Backward GPU"
          << std::endl;
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  for (int i = 0; i < top.size(); ++i) {
    if (this->param_propagate_down_[0]) {
      weight = this->blobs_[0]->gpu_data();
      weight_diff = this->blobs_[0]->mutable_gpu_diff();
    }
    Dtype* bias_diff = NULL;
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      bias_diff = this->blobs_[1]->mutable_gpu_diff();
    }
    const Dtype* top_diff = top[i]->gpu_diff();
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
      for (int n = 0; n < this->num_; ++n) {
          CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
            cudnn::dataType<Dtype>::one,
            top_descs_[i],  top_diff + top_offset_ * g + n * this->top_dim_,
            cudnn::dataType<Dtype>::one,
            bias_desc_, bias_diff + bias_offset_ * g));
          }
      }

      // Gradient w.r.t. weights. Note that we will accumulate diffs.
      if (this->param_propagate_down_[0] || propagate_down[i]) {
        for (int n = 0; n < this->num_; ++n) {
          if (this->param_propagate_down_[0]) {
            CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],    top_diff + top_offset_ * g + n * this->top_dim_,
              bottom_descs_[i], bottom_data + bottom_offset_ * g +
              n * this->bottom_dim_, conv_descs_[i],
              bwd_filter_algo_[i], workspace[1*this->group_ + g],
              workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + this->weight_offset_ * g));
          }
          // Gradient w.r.t. bottom data.
          if (propagate_down[i]) {
            if (weight == NULL) {
              weight = this->blobs_[0]->gpu_data();
            }
            CUDNN_CHECK(cudnnConvolutionForward(handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_diff + top_offset_ * g + n * this->top_dim_,
              filter_desc_, weight + this->weight_offset_ * g,
              conv_descs_[i],
              fwd_algo_[i], workspace[2*this->group_ + g],
              workspace_fwd_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g +
              n * this->bottom_dim_));
          }
        }
      }
    }
    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_deconv_groups<<<1, 1>>>();
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNDeconvolutionLayer);

}  // namespace caffe
#endif
