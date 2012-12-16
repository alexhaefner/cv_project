//
//  build_gaussian.h
//  cv_project
//
//  Created by Alex Haefner on 12/8/12.
//  Copyright (c) 2012 Alex Haefner. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <pthread.h>
#include <vector>
#include <string>

#ifndef cv_project_build_gaussian_h
#define cv_project_build_gaussian_h

class Gaussian3D
{
public:
    Gaussian3D();
    Gaussian3D(double **, int samples);
    ~Gaussian3D();
    double mu[3];
    double sigma[9]; //this was cv::Mat but replaced with double array for speed
};

extern Gaussian3D build_gaussian(std::string image_names[], int length, double **pixels);

extern double KL_Distance(const Gaussian3D &,const Gaussian3D &, double [], double [], double []);

#endif
