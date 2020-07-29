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
        float input[48]={0.726929,-0.686713,0,0.158151,-0.861633,0.482261,-0.257199,-0.78799,-0.559393,-0.663442,-0.748228,0,0.0205787,-0.849843,0.526634,0.0523806,-0.792685,-0.607377,-0.0797524,0.972289,-0.219758,-0.0797524,0.972289,-0.219758,0.941548,0.331547,-0.059699,0.512718,-0.844069,-0.157061,0.660604,-0.681821,0.314201,-0.7416,-0.527576,0.414357,-0.893561,0.448883,0.00731473,-0.461029,-0.885892,-0.0514584,-0.646183,-0.688946,0.328332,0.719832,-0.407261,0.562121
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
