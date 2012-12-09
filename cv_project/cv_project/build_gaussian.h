//
//  build_gaussian.h
//  cv_project
//
//  Created by Alex Haefner on 12/8/12.
//  Copyright (c) 2012 Alex Haefner. All rights reserved.
//

#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#ifndef cv_project_build_gaussian_h
#define cv_project_build_gaussian_h

class Gaussian3D
{
public:
    Gaussian3D();
    Gaussian3D(cv::Mat*, int samples);
    ~Gaussian3D();
    cv::Mat mu;
    cv::Mat sigma;
};

extern Gaussian3D build_gaussian(std::string image_names[], int length);

extern double KL_Distance(Gaussian3D, Gaussian3D);

#endif
