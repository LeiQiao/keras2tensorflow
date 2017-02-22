//
//  ViewController.m
//  testKeras2Tensorflow_iOS
//
//  Created by lei.qiao on 2017/2/21.
//  Copyright © 2017年 LoveC. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include "keras2tensorflow.hpp"

#include "sys/time.h"

#import "ViewController.h"

double getTimeFromLastOp()
{
    static struct timeval lastTime;
    static bool haveLastTime = false;
    if( !haveLastTime )
    {
        gettimeofday(&lastTime, NULL);
        haveLastTime = true;
    }
    
    struct timeval curTime;
    gettimeofday(&curTime, NULL);
    double timeuse = curTime.tv_sec - lastTime.tv_sec + (curTime.tv_usec - lastTime.tv_usec) / 1000000.0;
    
    gettimeofday(&lastTime, NULL);
    return timeuse;
}

@interface ViewController ()

@property(nonatomic, strong) IBOutlet UITextView* textView;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    [self testKeras2Tensorflow];
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

-(void) testKeras2Tensorflow
{
    // load protobuf
    NSString* filename = [[NSBundle mainBundle] pathForResource:@"mnist.pb" ofType:nil];
    
    // read model
    keras2tensorflow::Keras2Tensorflow k2tf;
    k2tf.loadFromProtobuf([filename UTF8String]);
    
    // initialize log message
    NSString* logMessage = @"";
    
    double aveTimeCost = 0;
    int aveCount = 0;
    
    for(int runrepeat=0; runrepeat < 100; runrepeat ++)
    for( int i=0; i<10; i++ )
    {
        // open image
        filename = [NSString stringWithFormat:@"%d.png", i];
        filename = [[NSBundle mainBundle] pathForResource:filename ofType:nil];
        
        cv::Mat image = cv::imread([filename UTF8String]);
        
        if( image.data == NULL )
        {
            logMessage = [logMessage stringByAppendingFormat:@"%@ cannot read.\n", filename];
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
        getTimeFromLastOp();
        bool s = k2tf.prediction(image_tensor, outputs);
        double predictionTime = getTimeFromLastOp();
        aveTimeCost += predictionTime;
        aveCount++;
        if( !s )
        {
            logMessage = [logMessage stringByAppendingFormat:@"%@ prediction failed.\n", filename];
            continue;
        }
        
        // print results
        logMessage = [logMessage stringByAppendingString:@"["];
        for( int i=0; i<outputs.size(); i++ )
        {
            logMessage = [logMessage stringByAppendingFormat:@" %.0f", outputs[i]];
        }
        logMessage = [logMessage stringByAppendingString:@"]\n"];
        logMessage = [logMessage stringByAppendingFormat:@"prediction cost: %.6fms\n", predictionTime];
        logMessage = [logMessage stringByAppendingFormat:@"repeat time: %d average cost: %.6fms\n", aveCount, aveTimeCost / aveCount];
        printf("repeat time: %d average cost: %.6fms\n", aveCount, aveTimeCost / aveCount);
        
        self.textView.text = logMessage;
    }
    
    self.textView.text = logMessage;
}


@end
