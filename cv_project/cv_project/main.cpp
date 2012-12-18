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
#include <vector>
#include <cmath>
#include "build_gaussian.h"
#include <opencv2/features2d/features2d.hpp>

#define NUM_THREADS 2
#define NUM_EXTRA_IMAGE_SAMPLES 10

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

int rheight = 3;
int cwidth = 3;

void *callKLDistance(void* pass)
{
    gaussians *calls = (gaussians *)pass;
    //double result = KL_Distance(*(calls->a), *(calls->b));
    //return (void*)&result;
}
double new_maxprob = 0.0;
static double prob_threshold = 0.001;
static double search_threshold = 0.01;

int calculate_image_probabilities(cv::Mat &image, cv::Mat &prob_image, double *prob, double **pixels, Gaussian3D &N_o, Gaussian3D &N_b, std::vector<cv::Rect> &ROI)
{
    int cstart = 0;
    int rstart = 0;
    double probability;
    cv::Vec3b maxprob_bgrpixel;
    int count = 0;
    double mu1_minus_m2[3];
    double gauss2invert_and_trace[9];
    double resulting[3];
    new_maxprob = 0.0;
    int roisize = (int)ROI.size();
    for (int j=0; j<image.cols-2; j++) {
        for (int i=0; i<image.rows-2; i++) {
            /*bool insideRange = false;
            if(roisize != 0) {
                for (int k=0; k<roisize; k++)
                {
                    cv::Rect boundingrect = ROI.at(k);
                    if (boundingrect.contains(cv::Point(j, i)))
                    {
                        insideRange = true;
                        break;
                    }
                }
            } else
            {
                insideRange = true;
            }
            if (!insideRange)
            {
                // the point is not within any of our regions of interest and is not worth considering
                continue;
            }*/
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
                    double *bgrpixel_float = pixels[count];
                    bgrpixel_float[0] = (double)bgrpixel[0] / 255;
                    bgrpixel_float[1] = (double)bgrpixel[1] / 255;
                    bgrpixel_float[2] = (double)bgrpixel[2] / 255;
                    //std::cout << bgrpixel_float << std::endl;
                    pixels[count] = bgrpixel_float;
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
                double N_iw_No = KL_Distance(N_iw, N_o, mu1_minus_m2, gauss2invert_and_trace, resulting);
                double N_iw_Nb = KL_Distance(N_iw, N_b, mu1_minus_m2, gauss2invert_and_trace, resulting);
                probability = exp(-N_iw_No / (N_iw_No + N_iw_Nb));
                if (probability != probability) // nan again
                {
                    probability = 0.0;
                }
                //std::cout << probability << std::endl;
            } catch (int e) {
                printf("FAIL");
            }
            new_maxprob = std::max(probability, new_maxprob);
            prob[j*image.rows + i] = probability;
            //std::cout << prob << std::endl;
        }
    }
    // I cycle the max probability pixel from each frame into the color of the hand gaussian
    // This keeps the gaussian up to date with changing lighting conditions
    /*for (int j=0; j<image.cols-2; j++) {
        for (int i=0; i<image.rows-2; i++) {
            if (prob[j*image.rows + i] > new_maxprob - prob_threshold)
                image.at<cv::Vec3b>(i,j) = cv::Vec3b(255, 255, 255);
        }
    }*/
    //if (new_maxprob > old_maxprob - 0.1)
    //{
    //}
    for (int j=0; j<image.cols-2; j++) {
        for (int i=0; i<image.rows-2; i++) {
            //std::cout << prob_image.at<cv::Vec3f>(j,i) << std::endl;
            
            uchar res = pow(((prob[j*image.rows + i] -  new_maxprob) + 1.0), 2550) * 255;
            //std::cout << prob[j*image.rows + i] << std::endl;
            prob_image.at<uchar>(i,j) = res;
        }
    }
    //std::cout << new_maxprob << std::endl;
    return 0;
}

void resetProbabilityMapToZero(double *prob)
{
    for (int i=0; i<360*240; i++)
    {
        prob[i] = 0.0;
    }
}


int main(int argc, const char * argv[])
{
    cv::Mat image;
    int imagecount = 6; // pre loaded images of hands and background
    // Generate gaussians of hands and background
    int min_samples = 6;
    int min_bg_samples = 6;
    int num_bg_samples = 0;
    int num_samples = 0;
    int total_samples = min_samples;
    int total_bg_samples = min_samples;
    double **hand_pixels = new double*[imagecount+NUM_EXTRA_IMAGE_SAMPLES];
    double **background_pixels = new double*[imagecount+NUM_EXTRA_IMAGE_SAMPLES];
    Gaussian3D hand_gaussian = build_gaussian(hand_files, imagecount, hand_pixels);
    Gaussian3D background_gaussian = build_gaussian(background_files, imagecount, background_pixels);
    
    //Setup the rest of the hand pixels which will be filled in later by high probability samples
    for (int i=imagecount; i<imagecount+NUM_EXTRA_IMAGE_SAMPLES; i++)
    {
        double *sum = new double[3];
        sum[0] = 0.0;
        sum[1] = 0.0;
        sum[2] = 0.0;
        hand_pixels[i] = sum;
        double *sum2 = new double[3];
        sum2[0] = 0.0;
        sum2[1] = 0.0;
        sum2[2] = 0.0;
        background_pixels[i] = sum2;
    }
    
    //start video capture
    cv::VideoCapture capture =  cv::VideoCapture(0);
    cv::namedWindow("image");
    cv::namedWindow("probability");
    cv::namedWindow("subregion");
    double prob[360*240];
    int theta = 0; //Theta as described in MSER tracking, is the interval on which we will back-check the entire image for new ROI
    // optimization to stop prob from being created every frame
    //pixels are the window of pixels for a covariance calculation in the image
    double **pixels = new double*[rheight*cwidth];
    for (int i=0; i<rheight*cwidth; i++) {
        pixels[i] = new double[3];
    }
    cv::Mat imageClone;
    cv::Mat prob_image = cv::Mat(360, 240, CV_8UC1, 1);
    std::vector<cv::Rect> ROI;
    while(true) {
        resetProbabilityMapToZero(prob);
        capture.read(image);
        cv::Size new_img_dims = cv::Size(360, 240);
        cv::resize(image, image, new_img_dims);
        imageClone = image.clone();
        cv::cvtColor(image, prob_image, CV_BGR2GRAY, 1);
        //cv::GaussianBlur(image, image, cv::Size(3,3), 0.0);
        calculate_image_probabilities(image, prob_image, prob, pixels, hand_gaussian, background_gaussian, ROI);
        std::vector<std::vector<cv::Point> > contours;
        cv::GaussianBlur(prob_image, imageClone, cv::Size(3, 3), 0.0);
        cv::MSER()(prob_image, contours);
        for (int i=(int)contours.size()-1; i>=0; i--)
        {
            const std::vector<cv::Point>& r = contours[i];
            double totalprob = 0.0;
            for (int j=0; j< (int)r.size(); j++)
            {
                cv::Point pt = r[j];
                int index = pt.x *image.rows + pt.y;
                totalprob += prob[index];
                //Sum the probability in a region
            }
            double average_probability = totalprob/(int)r.size();
            // IF the average probability in that region is above a threshold mark this as region to track
            if (average_probability > new_maxprob - 0.01 && theta == 0 ) // if theta is 0 it's time to reconsider our ROIs
            {
                cv::Rect boundingRect = cv::boundingRect(contours[i]);
                ROI.push_back(boundingRect);
            }
            if (average_probability > new_maxprob - 0.0002)
            {
                std::cout << average_probability << std::endl;
                //cv::drawContours(image, contours, i, cv::Scalar(0, 255, 255), 0.1);
                //std::cout << cv::contourArea(contours[i]) << std::endl;
                cv::Rect rect = cv::boundingRect(r);
                cv::RotatedRect rotRect = fitEllipse(r);
                rotRect.angle = (double)CV_PI/2 - rotRect.angle;
                cv::ellipse(image, rotRect, cv::Scalar(196, 255, 255));
                cv::rectangle(image, rect.tl(), rect.br(), cv::Scalar(255, 0, 0), 0.1);
                //newRects.push_back(rect);
                for (int j=0; j< (int)r.size(); j++)
                {
                    cv::Point pt = r[j];
                    //previous_mser[index] += 1;
                    prob_image.at<uchar>(pt) = (uchar)255;
                }
                cv::Rect boundingRect = cv::boundingRect(contours[i]);
                cv::Mat Subregion = image(boundingRect).clone();
                cv::threshold(Subregion, Subregion, 150, 150, CV_THRESH_BINARY_INV);
                //cv::imshow("subregion", Subregion);
                double minprob = 1.0;
                double maxprob = 0.0;
                cv::Vec3b maxprob_bgrpixel;
                cv::Vec3b minprob_bgrpixel;
                int topyval, bottomy, topx, bottomx;
                topyval = boundingRect.tl().y;
                bottomy = boundingRect.br().y;
                topx = boundingRect.tl().x;
                bottomx = boundingRect.br().x;
                //std::cout << topyval << std::endl;
                for (int x=topx; x<bottomx; x++)
                {
                    for (int y=topyval; y<bottomy; y++)
                    {
                        int idx = x*image.rows +y;
                        double probability = prob[idx];
                        //std::cout << minprob << std::endl;
                        minprob = std::min(probability, minprob);
                        if (probability == minprob)
                            minprob_bgrpixel = image.at<cv::Vec3b>(y, x);
                        
                        maxprob = std::max(probability, maxprob);
                       
                        if (probability == maxprob)
                            maxprob_bgrpixel = image.at<cv::Vec3b>(y,x);
                    }
                }
                
                if(minprob - maxprob <= 0.1)
                    continue;
                
                double *px = hand_pixels[min_samples + num_samples];
                //std::cout << *min_samples + *sampled_count << std::endl;
                px[0] = (double)maxprob_bgrpixel[0] / 255;
                px[1] = (double)maxprob_bgrpixel[1] / 255;
                px[2] = (double)maxprob_bgrpixel[2] / 255;
                num_samples = (num_samples + 1) % NUM_EXTRA_IMAGE_SAMPLES; // window from 0 to 6
                total_samples = std::min(total_samples + 1, min_samples + NUM_EXTRA_IMAGE_SAMPLES);
                
                double *px2 = background_pixels[min_bg_samples + num_bg_samples];
                //std::cout << *min_samples + *sampled_count << std::endl;
                px2[0] = (double)minprob_bgrpixel[0] / 255;
                px2[1] = (double)minprob_bgrpixel[1] / 255;
                px2[2] = (double)minprob_bgrpixel[2] / 255;
                num_bg_samples = (num_bg_samples + 1) % NUM_EXTRA_IMAGE_SAMPLES; // window from 0 to 6
                total_bg_samples = std::min(total_bg_samples + 1, min_bg_samples + NUM_EXTRA_IMAGE_SAMPLES);
            }
            // search_threshold is a larger threshold to know which pixels should be searched next time
            /*else if (average_probability > new_maxprob - search_threshold)
            {
                for (int j=0; j< (int)r.size(); j++)
                {
                    cv::Point pt = r[j];
                    //previous_mser[index] += 1;
                    int origindex = pt.x *image.rows + pt.y;
                    previous_mser[origindex] = true;
                    int index = (origindex +1 >= image.rows *image.cols) ? origindex : origindex + 1;
                    previous_mser[index] = true;
                    index = (origindex -1 < 0) ? origindex : origindex - 1;
                    previous_mser[origindex] = true;
                    index = (origindex + image.rows >= image.rows *image.cols) ? origindex : origindex + image.rows;
                    previous_mser[origindex] = true;
                }
            }*/
        }
        theta++;
        if (theta == 5)
        {
            theta = 0;
            ROI.clear();
        }
        
        // regenerate hand/bg gaussians based on interest regions
        hand_gaussian = Gaussian3D::Gaussian3D(hand_pixels, total_samples);
        background_gaussian = Gaussian3D::Gaussian3D(background_pixels, total_bg_samples);
        cv::imshow("image", image);
        cv::imshow("prob_image", imageClone);
        int key = cv::waitKey(10);
        if (key != -1) {
            std::cout << "key: " << key << std::endl;
        }
        if (key == 115) { //key 's'
            break;
        }
    }
    delete [] hand_pixels;
    delete [] pixels;
    // insert code here...
    std::cout << "Hello, World!\n";
    return 0;
}

