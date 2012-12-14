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

std::string hand_files[] = {
    "/Users/ahaef/cv_project/hands/0.png",
    "/Users/ahaef/cv_project/hands/1.png",
    "/Users/ahaef/cv_project/hands/2.png",
    "/Users/ahaef/cv_project/hands/3.png",
    "/Users/ahaef/cv_project/hands/4.png",
    "/Users/ahaef/cv_project/hands/5.png",
    "/Users/ahaef/cv_project/hands/6.png",
    "/Users/ahaef/cv_project/hands/7.png",
    "/Users/ahaef/cv_project/hands/8.png",
    "/Users/ahaef/cv_project/hands/9.png",
    "/Users/ahaef/cv_project/hands/10.png",
    "/Users/ahaef/cv_project/hands/11.png",
    "/Users/ahaef/cv_project/hands/12.png",
};
std::string background_files[] = {
    "/Users/ahaef/cv_project/background/0.png",
    "/Users/ahaef/cv_project/background/1.png",
    "/Users/ahaef/cv_project/background/2.png",
    "/Users/ahaef/cv_project/background/3.png",
    "/Users/ahaef/cv_project/background/4.png",
    "/Users/ahaef/cv_project/background/5.png",
    "/Users/ahaef/cv_project/background/6.png",
    "/Users/ahaef/cv_project/background/7.png",
    "/Users/ahaef/cv_project/background/8.png",
    "/Users/ahaef/cv_project/background/9.png",
    "/Users/ahaef/cv_project/background/10.png",
    "/Users/ahaef/cv_project/background/11.png",
    "/Users/ahaef/cv_project/background/12.png",
};

int calculate_image_probabilities(cv::Mat &image, cv::Mat &prob, Gaussian3D &N_o, Gaussian3D &N_b)
{
    int cstart = 0;
    int rstart = 0;
    int rheight = 3;
    int cwidth = 3;
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
            int count = 0;
            cv::Mat *pixels = new cv::Mat[rheight*cwidth];
            for (int cindex = cstart; cindex < cstart + cwidth; cindex++)
            {
                for (int rindex =rstart; rindex< rstart+rheight; rindex++)
                {
                    cv::Vec3b bgrpixel = image.at<cv::Vec3b>(rindex, cindex);
                    cv::Vec3f bgrpixel_float = cv::Vec3f(0.0, 0.0, 0.0);
                    bgrpixel_float[0] = (float)bgrpixel[0] / 255;
                    bgrpixel_float[1] = (float)bgrpixel[1] / 255;
                    bgrpixel_float[2] = (float)bgrpixel[2] / 255;
                    //std::cout << bgrpixel_float << std::endl;
                    pixels[count] = cv::Mat(bgrpixel_float, true);
                    count++;
                }
            }
            Gaussian3D N_iw = Gaussian3D(pixels, count);
            try {
                double prob = exp(-KL_Distance(N_iw, N_o) / (KL_Distance(N_iw, N_o) + KL_Distance(N_iw, N_b)));
                std::cout << prob << std::endl;
            } catch (int e) {
                printf("FAIL");
            }
            //std::cout << prob << std::endl;
        }
    }
    return 0;
}


int main(int argc, const char * argv[])
{
    cv::Mat image;
    Gaussian3D hand_gaussian = build_gaussian(hand_files, 13);
    Gaussian3D background_gaussian = build_gaussian(background_files, 13);
    cv::VideoCapture capture =  cv::VideoCapture(0);
    cv::namedWindow("image");
    while(true) {
        capture.read(image);
        cv::Size new_img_dims = cv::Size(320, 240);
        cv::Mat newImage;
        cv::resize(image, image, new_img_dims);
        cv::Mat prob = cv::Mat(image.rows, image.cols, CV_64F);
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

