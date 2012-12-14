//
//  main.cpp
//  cv_project
//
//  Created by Alex Haefner on 12/8/12.
//  Copyright (c) 2012 Alex Haefner. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>
#include "build_gaussian.h"

#define NUM_THREADS 2

std::string hand_files[] = {
    "/Users/ahaef/cv_project/hands/0.png",
    "/Users/ahaef/cv_project/hands/1.png",
    "/Users/ahaef/cv_project/hands/2.png",
    "/Users/ahaef/cv_project/hands/3.png",
    "/Users/ahaef/cv_project/hands/4.png",
    "/Users/ahaef/cv_project/hands/5.png",
    //"/Users/ahaef/cv_project/hands/6.png",
    //"/Users/ahaef/cv_project/hands/7.png",
    //"/Users/ahaef/cv_project/hands/8.png",
    //"/Users/ahaef/cv_project/hands/9.png",
    //"/Users/ahaef/cv_project/hands/10.png",
    //"/Users/ahaef/cv_project/hands/11.png",
    //"/Users/ahaef/cv_project/hands/12.png",
};
std::string background_files[] = {
    "/Users/ahaef/cv_project/background/0.png",
    "/Users/ahaef/cv_project/background/1.png",
    "/Users/ahaef/cv_project/background/2.png",
    "/Users/ahaef/cv_project/background/3.png",
    "/Users/ahaef/cv_project/background/4.png",
    "/Users/ahaef/cv_project/background/5.png",
    //"/Users/ahaef/cv_project/background/6.png",
    //"/Users/ahaef/cv_project/background/7.png",
    //"/Users/ahaef/cv_project/background/8.png",
    //"/Users/ahaef/cv_project/background/9.png",
    //"/Users/ahaef/cv_project/background/10.png",
    //"/Users/ahaef/cv_project/background/11.png",
    //"/Users/ahaef/cv_project/background/12.png",
};

struct gaussians {
    Gaussian3D *a;
    Gaussian3D *b;
};

void *callKLDistance(void* pass)
{
    gaussians *calls = (gaussians *)pass;
    double result = KL_Distance(*(calls->a), *(calls->b));
    return (void*)&result;
}

int calculate_image_probabilities(cv::Mat &image, cv::Mat &prob, Gaussian3D &N_o, Gaussian3D &N_b)
{
    int cstart = 0;
    int rstart = 0;
    int rheight = 3;
    int cwidth = 3;
    double probability;
    int count = 0;
    cv::Mat *pixels = new cv::Mat[rheight*cwidth];
    for (int j=0; j<image.cols-2; j++) {
        for (int i=0; i<image.rows-2; i++) {
            if (j==0)
                cstart = 0;
            else if (j == image.cols -2)
                cstart = image.cols -2;
            else
                cstart = j - 1;
            if (i==0)
                rstart=0;
            else if (i==image.rows -2)
                rstart = i - 2;
            else
                rstart = i - 1;
            count = 0;
            for (int cindex = cstart; cindex < cstart + cwidth; cindex++)
            {
                for (int rindex =rstart; rindex< rstart+rheight; rindex++)
                {
                    cv::Vec3b bgrpixel = image.at<cv::Vec3b>(rindex, cindex);
                    cv::Vec3d bgrpixel_float = cv::Vec3d(0.0, 0.0, 0.0);
                    bgrpixel_float[0] = (double)bgrpixel[0] / 255;
                    bgrpixel_float[1] = (double)bgrpixel[1] / 255;
                    bgrpixel_float[2] = (double)bgrpixel[2] / 255;
                    //std::cout << bgrpixel_float << std::endl;
                    pixels[count] = cv::Mat(bgrpixel_float, true);
                    count++;
                }
            }
            Gaussian3D N_iw = Gaussian3D(pixels, count);
            //Trying threads to make it faster
            /*pthread_t threads[NUM_THREADS];
            
            for (int index=0; index<NUM_THREADS; index++)
            {
                gaussians pass;
                pass.a = &N_iw;
                if (index==0)
                {
                    pass.b = &N_o;
                } else
                {
                    pass.b = &N_b;
                }
                pthread_create(&threads[index], NULL, callKLDistance, (void*)&pass);
            }
            void *N_iw_No, *N_iw_Nb;
            pthread_join(threads[0], &N_iw_No);
            pthread_join(threads[1], &N_iw_Nb);
            double N_iw_No = (double)N_iw_No;
            double N_iw_Nb = (double)N_iw_Nb;*/
            try {
                double N_iw_No = KL_Distance(N_iw, N_o);
                double N_iw_Nb = KL_Distance(N_iw, N_b);
                probability = exp(-N_iw_No / (N_iw_No + N_iw_Nb));
                if (probability != probability) // nan again
                {
                    probability = 0.0;
                }
                //std::cout << probability << std::endl;
            } catch (int e) {
                printf("FAIL");
            }
            if (probability >= .3914)
            {
                image.at<cv::Vec3b>(i,j) = cv::Vec3b(255, 255, 255);
            }
            //std::cout << prob << std::endl;
        }
    }
    return 0;
}


int main(int argc, const char * argv[])
{
    cv::Mat image;
    int imagecount = 6; // pre loaded images of hands and background
    // Generate gaussians of hands and background
    Gaussian3D hand_gaussian = build_gaussian(hand_files, imagecount);
    Gaussian3D background_gaussian = build_gaussian(background_files, imagecount);
    //start video capture
    cv::VideoCapture capture =  cv::VideoCapture(0);
    cv::namedWindow("image");
    cv::Mat prob;
    // optimization to stop prob from being created every frame
    int count = 0;
    while(true) {
        capture.read(image);
        cv::Size new_img_dims = cv::Size(320, 240);
        cv::Mat newImage;
        cv::resize(image, image, new_img_dims);
        if (count ==0) {
            count += 1;
            prob = image.clone();
        }
        calculate_image_probabilities(image, prob, hand_gaussian, background_gaussian);
        cv::imshow("image", image);
        int key = cv::waitKey(10);
        if (key != -1) {
            std::cout << "key: " << key << std::endl;
        }
        if (key == 115) { //key 's'
            break;
        }
    }
    // insert code here...
    std::cout << "Hello, World!\n";
    return 0;
}

