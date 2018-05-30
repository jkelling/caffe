#include <string>
#include <vector>
#include <array>

#include <opencv2/imgproc.hpp>

#include <opencv2/imgcodecs.hpp>
#include <sstream>

#include "caffe/layers/feature_map_augmentation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
FeatureMapAugmentationLayer<Dtype>::~FeatureMapAugmentationLayer<Dtype>() { }

template <typename Dtype>
void FeatureMapAugmentationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Refuse transformation parameters since HDF5 is totally generic.
  CHECK(this->layer_param_.has_transform_param()) <<
      this->type() << " missing transform params.";
  Reshape(bottom, top);
}

template <typename Dtype>
void FeatureMapAugmentationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[0]->num_axes() == 4 || bottom[1]->num_axes() == 4) <<
      this->type() << " requires 2d bottom.";
  for(int a = 0; a < 1; ++a)
    CHECK(bottom[a]->shape(1) == bottom[a]->shape(2)) <<
        this->type() << " bottom[" << a <<"] must be square";
  CHECK(bottom[0]->shape(1) >= bottom[1]->shape(1)) <<
      this->type() << " bottom[0] must larger or equal bottom[1]";

  // Reshape blobs.
  vector<int> top_shape = bottom[0]->shape();
  if(this->layer_param_.transform_param().crop_size() > 0)
  {
    for(int a = 1; a < top_shape.size()-1; ++a)
      top_shape[a] = this->layer_param_.transform_param().crop_size();
    rand_vec_.Reshape(std::vector<int>({4*top_shape[0]}));
    map_cut_size_ = top[0]->shape(1)*bottom[1]->shape(1)/bottom[0]->shape(1);
  }
  else
  {
    rand_vec_.Reshape(std::vector<int>({2*top_shape[0]}));
    map_cut_size_ = bottom[1]->shape(1);
  }

  top[0]->Reshape(top_shape);
  top[1]->Reshape(bottom[1]->shape());

}

template <typename Dtype>
void FeatureMapAugmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = bottom[0]->shape(0);

  auto rnd = rand_vec_.mutable_cpu_data();
  caffe_rng_uniform(rand_vec_.count(), 0.f,1.f, rnd);

  const unsigned int shapeBottom = bottom[0]->shape(1);
  const unsigned int shapeTop = top[0]->shape(1);
  const unsigned int shapeMap = bottom[1]->shape(1);
  const unsigned int diff = bottom[0]->shape(1)-top[0]->shape(1);
  const size_t imgSizeBottom = bottom[0]->count(1);
  const size_t mapSize = bottom[1]->count(1);
  const size_t imgSizeTop = top[0]->count(1);
  size_t rndi = 0;
  for (int i = 0; i < batch_size; ++i) {
    const unsigned int tflags = rnd[rndi++]*(1<<6); // 2 flips + 4 angles
    const float amp = rnd[rndi++]+.2;
    cv::Mat trans = (cv::Mat_<float>(2,2) << 1, 0, 0, 1 );
    cv::Mat shift = cv::Mat::zeros(2,1, CV_32F);
    if(tflags & 1) // flip horizontally
      trans *= (cv::Mat_<float>(2,2) << -1, 0, 0, 1 );
      shift.at<float>(0) = ((int)shift.at<float>(0)+1)&1;
    if(tflags & 2) // flip vertically
      trans *= (cv::Mat_<float>(2,2) << 1, 0, 0, -1 );
      shift.at<float>(1) = ((int)shift.at<float>(1)+1)&1;
    switch((tflags>>2)&3)
    {
      // case 0: would be no rotation
      case 1:
        trans *= (cv::Mat_<float>(2,2) << 0, -1, 1, 0 );
        shift.at<float>(0) = ((int)shift.at<float>(0)+1)&1;
        break;
      case 2:
        trans *= (cv::Mat_<float>(2,2) << -1, 0, 0, -1 );
        shift.at<float>(1) = ((int)shift.at<float>(1)+1)&1;
      // case 3: both flips yield 180 degrees rotation
    }
    cv::Mat imgBottom(shapeBottom, shapeBottom, CV_32F, bottom[0]->mutable_cpu_data()+i*imgSizeBottom);
    cv::Mat imgTop(shapeTop, shapeTop, CV_32F, top[0]->mutable_cpu_data()+i*imgSizeTop);
    cv::Mat mapBottom(shapeMap, shapeMap, CV_32F, bottom[1]->mutable_cpu_data()+i*mapSize);
    cv::Mat mapTop(shapeMap, shapeMap, CV_32F, top[1]->mutable_cpu_data()+i*mapSize);
    cv::Mat T;
    cv::hconcat(trans, shift*shapeTop, T);
    if(diff > 0)
    {
      const float shiftX = rnd[rndi++];
      const float shiftY = rnd[rndi++];
      cv::warpAffine(imgBottom(cv::Rect(diff*shiftX, diff*shiftY, shapeTop, shapeTop))*amp, imgTop, T, imgTop.size());
      cv::hconcat(trans, shift*map_cut_size_, T);
      T *= shapeMap/(float)map_cut_size_;
      cv::warpAffine(mapBottom(cv::Rect(diff*shiftX, diff*shiftY, map_cut_size_, map_cut_size_)), mapTop, T, mapTop.size());
    }
    else
    {
      cv::warpAffine(imgBottom*amp, imgTop, T, imgTop.size());
      cv::hconcat(trans, shift*shapeMap, T);
      cv::warpAffine(mapBottom, mapTop, T, mapTop.size());
    }

    cv::Mat dbgTmp = cv::Mat::ones(2*shapeBottom, 2*shapeBottom, CV_32F);
    dbgTmp(cv::Rect(0,0,shapeBottom, shapeBottom)) = (imgBottom+128)*128;
    dbgTmp(cv::Rect(shapeBottom,0,shapeBottom, shapeBottom)) = (imgTop+128)*128;
    dbgTmp(cv::Rect(0,shapeBottom,shapeBottom, shapeBottom)) = (mapBottom)*128;
    dbgTmp(cv::Rect(shapeBottom,shapeBottom,shapeBottom, shapeBottom)) = (mapTop)*128;
    cv::Mat dbgOut;
    dbgTmp.convertTo(dbgOut, CV_8UC1);

    std::ostringstream os;
    os << "augment_" << i << ".png";
    cv::imwrite(os.str().c_str(), dbgOut);
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(FeatureMapAugmentationLayer, Forward);
#endif

INSTANTIATE_CLASS(FeatureMapAugmentationLayer);
REGISTER_LAYER_CLASS(FeatureMapAugmentation);

}  // namespace caffe
