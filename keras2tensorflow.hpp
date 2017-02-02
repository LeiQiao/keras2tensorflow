#ifndef __KERAS2TENSORFLOW_HPP__
#define __KERAS2TENSORFLOW_HPP__

#include <fstream>
#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message_lite.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace keras2tensorflow {

// define layers
enum {
    kLayerUnknown = 0,
    kLayerConv2D,
    kLayerActivation,
    kLayerMaxPool,
    kLayerFlatten,
    kLayerDense,
    kLayerCount
};

// define activation types
enum {
    kActivationUnknown = 0,
    kActivationLinear,
    kActivationRelu,
    kActivationSoftmax,
    kActivationCount
};

/////////////////////////////////////////////////////////////////////
// keras's base layer
class KerasLayer
{
public:
    KerasLayer(){}
    virtual ~KerasLayer(){}

public:
    bool readUnsignedInt(std::ifstream* file, unsigned int* i);
    bool readFloat(std::ifstream* file, float* f);
    bool readFloats(std::ifstream* file, float* f, size_t n);

public:
    virtual bool loadFromFile(std::ifstream* file) = 0;
};

/////////////////////////////////////////////////////////////////////
// keras's Convolution2D layer
class KerasConv2D : public KerasLayer
{
public:
    KerasConv2D();
    virtual ~KerasConv2D();

public:
    virtual bool loadFromFile(std::ifstream* file);
    
private:
    tensorflow::Tensor mWeights;
    tensorflow::Tensor mBiases;
};

/////////////////////////////////////////////////////////////////////
// keras's Activation layer
class KerasActivation : public KerasLayer
{
public:
    KerasActivation();
    virtual ~KerasActivation();

public:
    virtual bool loadFromFile(std::ifstream* file);
    
private:
    unsigned int mActivationType;
};

/////////////////////////////////////////////////////////////////////
// keras's MaxPooling2D layer
class KerasMaxPool : public KerasLayer
{
public:
    KerasMaxPool();
    virtual ~KerasMaxPool();

public:
    virtual bool loadFromFile(std::ifstream* file);
    
private:
    unsigned int mPoolWidth;
    unsigned int mPoolHeight;
};

/////////////////////////////////////////////////////////////////////
// keras's Flatten layer
class KerasFlatten : public KerasLayer
{
public:
    KerasFlatten(){}
    virtual ~KerasFlatten(){}

public:
    virtual bool loadFromFile(std::ifstream* file){return true;}
};

/////////////////////////////////////////////////////////////////////
// keras's Dense layer
class KerasDense : public KerasLayer
{
public:
    KerasDense();
    virtual ~KerasDense();

public:
    virtual bool loadFromFile(std::ifstream* file);
    
private:
    tensorflow::Tensor mWeights;
    tensorflow::Tensor mBiases;
};

/////////////////////////////////////////////////////////////////////
// use this to convert from keras's model to tensorflow's layers
class Keras2Tensorflow : public KerasLayer
{
public:
    Keras2Tensorflow();
    virtual ~Keras2Tensorflow();

public:
    bool loadFromFile(const char* filename);
    bool loadFromProtobuf(const char* filename);
    
    virtual bool prediction(tensorflow::Tensor& input, std::vector<float>& outputs);

private:
    virtual bool loadFromFile(std::ifstream* file);
    
    std::vector<KerasLayer*> mLayers;
    tensorflow::GraphDef mGraph;
    tensorflow::Session* pSessionPointer;
    std::unique_ptr<tensorflow::Session> pSession;
};

};

#endif // __KERAS2TENSORFLOW_HPP__
