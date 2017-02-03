#include "keras2tensorflow.hpp"

using namespace keras2tensorflow;

#define KASSERT(x, ...) \
    if (!(x)) { \
        printf("KASSERT: %s(%d): ", __FILE__, __LINE__); \
        printf(__VA_ARGS__); \
        printf("\n"); \
        return false; \
    }


/////////////////////////////////////////////////////////////////////
// keras's base layer
bool KerasLayer::readUnsignedInt(std::ifstream* file, unsigned int* i)
{
    KASSERT(file, "Invalid file stream");
    KASSERT(i, "Invalid pointer");

    file->read((char *) i, sizeof(unsigned int));
    KASSERT(file->gcount() == sizeof(unsigned int), "Expected unsigned int");

    return true;
}

bool KerasLayer::readFloat(std::ifstream* file, float* f)
{
    KASSERT(file, "Invalid file stream");
    KASSERT(f, "Invalid pointer");

    file->read((char *) f, sizeof(float));
    KASSERT(file->gcount() == sizeof(float), "Expected float");

    return true;
}

bool KerasLayer::readFloats(std::ifstream* file, float* f, size_t n)
{
    KASSERT(file, "Invalid file stream");
    KASSERT(f, "Invalid pointer");

    file->read((char *) f, sizeof(float) * n);
    KASSERT(((unsigned int) file->gcount()) == sizeof(float) * n, "Expected floats");

    return true;
}


/////////////////////////////////////////////////////////////////////
// keras's Convolution2D layer
KerasConv2D::KerasConv2D()
{
}

KerasConv2D::~KerasConv2D()
{
}

bool KerasConv2D::loadFromFile(std::ifstream* file)
{
    KASSERT(file, "Invalid file stream");
    
    unsigned int weights_rows = 0;
    KASSERT(readUnsignedInt(file, &weights_rows), "Expected weights_rows");
    KASSERT(weights_rows > 0, "Invalid weights # rows");
    
    unsigned int weights_cols = 0;
    KASSERT(readUnsignedInt(file, &weights_cols), "Expected weights_cols");
    KASSERT(weights_cols > 0, "Invalid weights # cols");
    
    unsigned int weights_depth = 0;
    KASSERT(readUnsignedInt(file, &weights_depth), "Expected weights_depth");
    KASSERT(weights_depth > 0, "Invalid weights # depth");
    
    unsigned int weights_count = 0;
    KASSERT(readUnsignedInt(file, &weights_count), "Expected weights_count");
    KASSERT(weights_count > 0, "Invalid weights # count");
    
    unsigned int biases_count = 0;
    KASSERT(readUnsignedInt(file, &biases_count), "Expected biases shape");
    
    tensorflow::TensorShape weightShape({weights_rows, weights_cols, weights_depth, weights_count});
    tensorflow::TensorShape biasShape({biases_count});
    
    mWeights = tensorflow::Tensor(tensorflow::DT_FLOAT, weightShape);
    mBiases = tensorflow::Tensor(tensorflow::DT_FLOAT, biasShape);
    
    float* weights = mWeights.tensor<float, 4>().data();
    float* biases = mBiases.tensor<float, 1>().data();
    
    KASSERT(readFloats(file, weights, weights_rows*weights_cols*weights_depth*weights_count), "Expected weights");
    KASSERT(readFloats(file, biases, biases_count), "Expected biases");
    

    return true;
}


/////////////////////////////////////////////////////////////////////
// keras's Activation layer
    
KerasActivation::KerasActivation()
{
    mActivationType = kActivationUnknown;
}
    
KerasActivation::~KerasActivation()
{
}
    
bool KerasActivation::loadFromFile(std::ifstream *file)
{
    KASSERT(file, "Invalid file stream");
    
    KASSERT(readUnsignedInt(file, &mActivationType), "Expected activation type");
    
    KASSERT((mActivationType > kActivationUnknown && mActivationType < kActivationCount),
            "Unsupported activation type: %d", mActivationType);
    
    return true;
}

/////////////////////////////////////////////////////////////////////
// keras's MaxPooling2D layer

KerasMaxPool::KerasMaxPool()
{
    mPoolWidth = mPoolHeight = 0;
}

KerasMaxPool::~KerasMaxPool()
{
}

bool KerasMaxPool::loadFromFile(std::ifstream *file)
{
    KASSERT(file, "Invalid file stream");
    
    KASSERT(readUnsignedInt(file, &mPoolWidth), "Expected pool width");
    KASSERT(readUnsignedInt(file, &mPoolHeight), "Expected pool height");
    
    return true;
}

/////////////////////////////////////////////////////////////////////
// keras's Flatten layer

/////////////////////////////////////////////////////////////////////
// keras's Dense layer

KerasDense::KerasDense()
{
}

KerasDense::~KerasDense()
{
}

bool KerasDense::loadFromFile(std::ifstream *file)
{
    KASSERT(file, "Invalid file stream");
    
    unsigned int weights_depth = 0;
    KASSERT(readUnsignedInt(file, &weights_depth), "Expected weights_depth");
    KASSERT(weights_depth > 0, "Invalid weights # depth");
    
    unsigned int weights_count = 0;
    KASSERT(readUnsignedInt(file, &weights_count), "Expected weights_count");
    KASSERT(weights_count > 0, "Invalid weights # count");
    
    unsigned int biases_count = 0;
    KASSERT(readUnsignedInt(file, &biases_count), "Expected biases shape");
    
    
    tensorflow::TensorShape weightShape({weights_depth, weights_count});
    tensorflow::TensorShape biasShape({biases_count});
    
    mWeights = tensorflow::Tensor(tensorflow::DT_FLOAT, weightShape);
    mBiases = tensorflow::Tensor(tensorflow::DT_FLOAT, biasShape);
    
    float* weights = mWeights.tensor<float, 2>().data();
    float* biases = mBiases.tensor<float, 1>().data();
    
    KASSERT(readFloats(file, weights, weights_depth*weights_count), "Expected weights");
    KASSERT(readFloats(file, biases, biases_count), "Expected biases");
    
    return true;
}

/////////////////////////////////////////////////////////////////////
// keras's Dropout layer

KerasDropout::KerasDropout()
{
}

KerasDropout::~KerasDropout()
{
}

bool KerasDropout::loadFromFile(std::ifstream *file)
{
    KASSERT(file, "Invalid file stream");
    
    KASSERT(readFloat(file, &mProb), "Expected prob");
    
    return true;
}

/////////////////////////////////////////////////////////////////////
// protobuf input stream

class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream
{
public:
    explicit IfstreamInputStream(const std::string& file_name)
    : ifs_(file_name.c_str(), std::ios::in | std::ios::binary) {}
    ~IfstreamInputStream() { ifs_.close(); }
    
    int Read(void* buffer, int size)
    {
        if (!ifs_) return -1;
        ifs_.read(static_cast<char*>(buffer), size);
        return (int)ifs_.gcount();
    }
    
private:
    std::ifstream ifs_;
};

/////////////////////////////////////////////////////////////////////
// use this to convert from keras's model to tensorflow's layers
Keras2Tensorflow::Keras2Tensorflow()
{
}

Keras2Tensorflow::~Keras2Tensorflow()
{
    while( !this->mLayers.empty() )
    {
        KerasLayer* oldLayer = this->mLayers.back();
        delete oldLayer;
        this->mLayers.pop_back();
    }
    
    if( this->pSession )
    {
        this->pSession->Close();
        this->pSession = NULL;
    }
}

bool Keras2Tensorflow::loadFromFile(const char* filename)
{
    // open model file
    std::ifstream file(filename, std::ios::binary);
    KASSERT(file.is_open(), "Unable to open file %s", filename);
    
    return this->loadFromFile(&file);
}

bool Keras2Tensorflow::loadFromProtobuf(const char* filename)
{
    // close last tensorflow session
    if( pSession )
    {
        pSession->Close();
        pSession = NULL;
    }
    
    // load protobuf from file
    IfstreamInputStream* stream = new IfstreamInputStream(filename);
    ::google::protobuf::io::CopyingInputStreamAdaptor sAdaptor(stream);
    sAdaptor.SetOwnsCopyingStream(true);
    ::google::protobuf::io::CodedInputStream coded_stream(&sAdaptor);
    coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
    
    bool bSucc = mGraph.ParseFromCodedStream(&coded_stream);
    if( !bSucc ) return false;
    
    // create new tensorflow session
    tensorflow::SessionOptions options;
    pSession = std::unique_ptr<tensorflow::Session>(tensorflow::NewSession(options));
    tensorflow::Status s = pSession->Create(mGraph);
    return s.ok();
}

bool Keras2Tensorflow::prediction(tensorflow::Tensor& input, std::vector<float>& outputs)
{
    if( !pSession ) return false;
    
    // run
    std::string input_layer = "input";
    std::string output_layer = "output";
    std::vector<tensorflow::Tensor> output_tensors;
    tensorflow::Status s = this->pSession->Run({{input_layer, input}},
                                               {output_layer},
                                               {},
                                               &output_tensors);
    
    if( !s.ok() )
    {
        tensorflow::LogAllRegisteredKernels();
        return false;
    }
    
    // output
    auto predictions = (&output_tensors[0])->flat<float>();
    long count = predictions.size();
    
    outputs.clear();
    for( int i=0; i<count; i++ )
    {
        outputs.push_back(predictions(i));
    }
    
    return true;
}

bool Keras2Tensorflow::loadFromFile(std::ifstream* file)
{
    // read layer count
    unsigned int layerCount = 0;
    KASSERT(readUnsignedInt(file, &layerCount), "Expected number of layers");

    // delete all old layers
    while( !this->mLayers.empty() )
    {
        KerasLayer* oldLayer = this->mLayers.back();
        delete oldLayer;
        this->mLayers.pop_back();
    }

    // load layers from file
    for( unsigned int iLayer=0; iLayer<layerCount; iLayer++ )
    {
        unsigned int layerType = kLayerUnknown;
        KASSERT(readUnsignedInt(file, &layerType), "Expected layer type");

        KerasLayer* newLayer = NULL;
        switch( layerType )
        {
            case kLayerConv2D:
                newLayer = new KerasConv2D();
            break;
            case kLayerActivation:
                newLayer = new KerasActivation();
            break;
            case kLayerMaxPool:
                newLayer = new KerasMaxPool();
            break;
            case kLayerFlatten:
                newLayer = new KerasFlatten();
            break;
            case kLayerDense:
                newLayer = new KerasDense();
            break;
            case kLayerDropout:
                newLayer = new KerasDropout();
            break;
            default:
                KASSERT(false, "Unsupported layer type: %d", layerType);
            break;
        }

        newLayer->loadFromFile(file);
        this->mLayers.push_back(newLayer);
    }
    
    return true;
}




