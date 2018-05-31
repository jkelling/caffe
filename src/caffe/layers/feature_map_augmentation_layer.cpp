#include <string>
#include <vector>
#include <array>
#include <algorithm>

#include <opencv2/imgproc.hpp>

#if 0
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <iostream>
#endif

#include "caffe/layers/feature_map_augmentation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
FeatureMapAugmentationLayer<Dtype>::~FeatureMapAugmentationLayer<Dtype>() { }

template <typename Dtype>
void FeatureMapAugmentationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param_.has_augmentation_param()) <<
      this->type() << " missing augmentation params.";

  Reshape(bottom, top);
  amp_min_ = this->layer_param_.augmentation_param().amp_min();
  scale_max_ = this->layer_param_.augmentation_param().scale_max();
}

template <typename Dtype>
void FeatureMapAugmentationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[0]->num_axes() == 4 && bottom[1]->num_axes() == 4) <<
      this->type() << " requires (2+2)d bottom.";
  for(int a = 0; a < 1; ++a)
    CHECK(bottom[a]->shape(3) == bottom[a]->shape(2)) <<
        this->type() << " bottom[" << a <<"] must be square ( " << bottom[a]->shape(2) << " != " << bottom[a]->shape(3) << ")";
  CHECK(bottom[0]->shape(3) == bottom[1]->shape(3)) <<
      this->type() << " shape of bottom[0] must be equal bottom[1]";

  // Reshape blobs.
  vector<int> top_shape = bottom[0]->shape();
  if(this->layer_param_.augmentation_param().crop_size() > 0)
  {
    for(int a = 2; a < top_shape.size(); ++a)
      top_shape[a] = this->layer_param_.augmentation_param().crop_size();
  }
  rand_vec_.Reshape(std::vector<int>({7*top_shape[0]}));

  top[0]->Reshape(top_shape);
  top[1]->Reshape(top_shape);

}

template <typename Dtype>
void FeatureMapAugmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = bottom[0]->shape(0);

  auto rnd = rand_vec_.mutable_cpu_data();
  caffe_rng_uniform(rand_vec_.count(), 0.f,1.f, rnd);

  const unsigned int shapeBottom = bottom[0]->shape(3);
  const unsigned int shapeTop = top[0]->shape(3);
  const size_t imgSizeBottom = bottom[0]->count(1);
  const size_t imgSizeTop = top[0]->count(1);
  size_t rndi = 0;
  for (int i = 0; i < batch_size; ++i) {
    const unsigned int tflags = rnd[rndi++]*(1<<4); // 2 flips + 4 angles
    const float amp = amp_min_ + rnd[rndi++]*(1.-amp_min_);
    const float scaleX = 1.f+ (rnd[rndi++]*2-1)*scale_max_;
    const float scaleY = 1.f+ (rnd[rndi++]*2-1)*scale_max_;
    const float bottomX = std::min(shapeTop*scaleX, (float)shapeBottom);
    const float bottomY = std::min(shapeTop*scaleY, (float)shapeBottom);
    const float shiftX = rnd[rndi++]*(shapeBottom-bottomX);
    const float shiftY = rnd[rndi++]*(shapeBottom-bottomY);
    cv::Mat trans = (cv::Mat_<float>(2,2) << 1, 0, 0, 1 );
    cv::Mat shift = cv::Mat::zeros(2,1, CV_32F);
    cv::Rect roi = cv::Rect(shiftX, shiftY, bottomX, bottomY);
    if(tflags & 1) // flip horizontally
    {
      trans *= (cv::Mat_<float>(2,2) << -1, 0, 0, 1 );
      shift.at<float>(0) = ((int)shift.at<float>(0)+1)&1;
    }
    if(tflags & 2) // flip vertically
    {
      trans *= (cv::Mat_<float>(2,2) << 1, 0, 0, -1 );
      shift.at<float>(1) = ((int)shift.at<float>(1)+1)&1;
    }
    switch((tflags>>2)&3)
    {
      // case 0: would be no rotation
      case 1:
        trans *= (cv::Mat_<float>(2,2) << 0, -1, 1, 0 );
        shift.at<float>(0) = ((int)shift.at<float>(0)+1)&1;
        break;
      case 2:
        trans *= (cv::Mat_<float>(2,2) << 0, 1, -1, 0 );
        shift.at<float>(1) = ((int)shift.at<float>(1)+1)&1;
      // case 3: both flips yield 180 degrees rotation
    }
    // std::cout << "augment_" << i << " tflags= " << tflags << std::endl;

    trans *= (cv::Mat_<float>(2,2) << shapeTop/bottomX, 0, 0, shapeTop/bottomY );
    cv::Mat imgBottom(shapeBottom, shapeBottom, CV_32F, bottom[0]->mutable_cpu_data()+i*imgSizeBottom);
    cv::Mat imgTop(shapeTop, shapeTop, CV_32F, top[0]->mutable_cpu_data()+i*imgSizeTop);
    cv::Mat mapBottom(shapeBottom, shapeBottom, CV_32F, bottom[1]->mutable_cpu_data()+i*imgSizeBottom);
    cv::Mat mapTop(shapeTop, shapeTop, CV_32F, top[1]->mutable_cpu_data()+i*imgSizeTop);
    cv::Mat T;
    cv::hconcat(trans, shift*shapeTop, T);

    cv::warpAffine(imgBottom(roi)*amp, imgTop, T, imgTop.size());
    cv::warpAffine(mapBottom(roi), mapTop, T, mapTop.size());

#if 0
    std::cout << "augment_" << i << " roi= " << roi << std::endl;
    cv::Mat dbgTmp = cv::Mat::ones(2*shapeBottom, 2*shapeBottom, CV_32F);
    dbgTmp(cv::Rect(0,0,shapeBottom, shapeBottom)) = (imgBottom*128)+128;
    dbgTmp(cv::Rect(shapeBottom,0,shapeTop, shapeTop)) = (imgTop*128)+128;
    dbgTmp(cv::Rect(0,shapeBottom,shapeBottom, shapeBottom)) = (mapBottom)*128;
    dbgTmp(cv::Rect(shapeBottom,shapeBottom,shapeTop, shapeTop)) = (mapTop)*128;
    cv::Mat dbgOut;
    dbgTmp.convertTo(dbgOut, CV_8UC1);

    std::ostringstream os;
    os << "augment_" << i << ".png";
    cv::imwrite(os.str().c_str(), dbgOut);
#endif
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(FeatureMapAugmentationLayer, Forward);
#endif

INSTANTIATE_CLASS(FeatureMapAugmentationLayer);
REGISTER_LAYER_CLASS(FeatureMapAugmentation);

}  // namespace caffe
