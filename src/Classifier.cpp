#define CPU_ONLY
#include "Classifier.h"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

std::vector<Prediction> Classifier::Classify(const std::vector<cv::Mat>& img) {
  std::vector< std::vector<float> > output = Predict(img, img.size());
  
  std::vector<Prediction> predictions;
  for ( int i = 0 ; i < output.size(); i++ ) {
    std::vector<int> maxN = Argmax(output[i], 1);
    int idx = maxN[0];
    predictions.push_back(std::make_pair(labels_[idx], output[i][idx]));
  }
  return predictions;
}

std::vector< std::vector<float> > Classifier::Predict(const std::vector<cv::Mat>& img, int nImages) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(nImages, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels, nImages);

  Preprocess(img, &input_channels, nImages);

  net_->ForwardPrefilled();

  /* Copy the output layer to a std::vector */
  
  Blob<float>* output_layer = net_->output_blobs()[0];
  std::vector <std::vector<float> > ret;
  for (int i = 0; i < nImages; i++) {
    const float* begin = output_layer->cpu_data() + i*output_layer->channels();
    const float* end = begin + output_layer->channels();
    ret.push_back( std::vector<float>(begin, end) );
  }
  return ret;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels, int nImages) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels()* nImages; ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const std::vector<cv::Mat>& img,
                            std::vector<cv::Mat>* input_channels, int nImages) {
  for (int i = 0; i < nImages; i++) {
      vector<cv::Mat> channels;
      cv::split(img[i], channels);
      for (int j = 0; j < channels.size(); j++){
           channels[j].copyTo((*input_channels)[i*num_channels_+j]);
      }
  }
}

