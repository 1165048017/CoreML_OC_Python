//
//  main.m
//  testCoreML
//
//  Created by 周兵 on 2020/7/29.
//  Copyright © 2020 周兵. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "tanh_adam.h"

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        float input[48]={0.687976,-0.725733,0,0.102204,-0.965447,0.239723,0.00265387,-0.934145,-0.356883,-0.678969,-0.734167,0,-0.0574909,-0.965108,0.255465,-0.184422,-0.89568,-0.404656,-0.0146139,0.982024,-0.18819,-0.0146139,0.982024,-0.18819,0.941489,0.327004,-0.0816485,0.670676,-0.739907,0.0522592,0.83285,-0.115006,0.541419,-0.532807,-0.43154,0.727936,-0.895597,0.4379,-0.0784221,-0.660519,-0.736813,0.144298,-0.844917,-0.115421,0.522295,0.529348,-0.320712,0.785452
        };
        // convert to MLMultiArray format
        NSError *error = NULL;
        NSArray *shape = [NSArray arrayWithObjects:[NSNumber numberWithInt:1],[NSNumber numberWithInt:48], nil];
        NSArray *stride = [NSArray arrayWithObjects:[NSNumber numberWithInt:1],[NSNumber numberWithInt:1], nil];
        MLMultiArray *mma = [[MLMultiArray alloc] initWithDataPointer:input
                                                                         shape:shape
                                                                      dataType:MLMultiArrayDataTypeFloat32
                                                                       strides:stride
                                                                   deallocator:^(void * _Nonnull bytes) { }
                                                                         error:&error];
     
//        for(int i = 0;i<48;i++){
//            NSLog(@"%f ",((float*)mma.dataPointer)[i]);
//        }
        tanh_adam *model = [[tanh_adam alloc] init];
        tanh_adamOutput *output = [model predictionFromInput_1:mma error:nil];
        
        float *result = (float*) output.Identity.dataPointer;
        for(int i = 0;i<39;i++){
            NSLog(@"%f ",result[i]);
        }
    }
    return 0;
}
