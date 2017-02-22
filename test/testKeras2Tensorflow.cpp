//
//  main.cpp
//  testKeras2Tensorflow
//
//  Created by lei.qiao on 17/1/30.
//  Copyright © 2017年 LoveC. All rights reserved.
//

#include <iostream>
#include "keras2tensorflow.hpp"

using namespace keras2tensorflow;

#include <opencv2/opencv.hpp>


int main(int argc, const char * argv[])
{
    // load protobuf
    const char* filename = "mnist.pb";
    
    // read model
    Keras2Tensorflow k2tf;
    k2tf.loadFromProtobuf(filename);
    
    for( int i=0; i<10; i++ )
    {
        // open image
        char filename[1024];
        sprintf(filename, "%d.png", i);
        
        cv::Mat image = cv::imread(filename);
        
        if( image.data == NULL )
        {
            std::cout << filename << " cannot read." << std::endl;
            continue;
        }
        cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
        
        // convert image to tensor
        const int wanted_width = image.rows;
        const int wanted_height = image.cols;
        tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT,
                                        tensorflow::TensorShape({
            1, wanted_height, wanted_width, 1}));
        
        auto image_tensor_mapped = image_tensor.tensor<float, 4>();
        float* tensor_pointer = image_tensor_mapped.data();
        
        for( int r = 0; r < image.rows; r++ )
        {
            for( int c = 0; c < image.cols; c++ )
            {
                int i = r * image.cols + c;
                tensor_pointer[i] = image.data[i];
            }
        }
        
        // prediction
        std::vector<float> outputs;
        bool s = k2tf.prediction(image_tensor, outputs);
        if( !s )
        {
            std::cout << filename << " prediction failed." << std::endl;
            continue;
        }
        
        // print results
        printf("[");
        for( int i=0; i<outputs.size(); i++ )
        {
            printf(" %.0f", outputs[i]);
        }
        printf(" ]\n");
    }
    
    return 0;
}
