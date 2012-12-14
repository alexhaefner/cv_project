#include "build_gaussian.h"

Gaussian3D::Gaussian3D()
{
    
}

Gaussian3D::Gaussian3D(cv::Mat *bgr_averages, int samples)
{
    cv::Vec3d sum = cv::Vec3d(0, 0, 0);
    for (int i=0; i<samples; i++) {
        cv::Vec3d bgrPixel = bgr_averages[i].at<cv::Vec3d>(0, 0);
        sum[0] += (double)bgrPixel[0];
        sum[1] += (double)bgrPixel[1];
        sum[2] += (double)bgrPixel[2];
    }
    //std::cout << sum << std::endl;
    cv::Vec3d averages = sum / samples;
    mu = cv::Mat(averages, CV_64FC1);
    //std::cout << mu << std::endl;
    //calcCovarMatrix doesn't works so I did it myself
    //double vals[9];
    sigma = cv::Mat(3,3, CV_64FC1);
    for (int i = 0; i<3; i++)
    {
        for (int j=0; j<3; j++)
        {
            sigma.at<double>(i,j) = 0.0;
        }
    }
    for (int index = 0; index< samples; index++)
    {
        cv::Vec3d bgrPixel = bgr_averages[index].at<cv::Vec3d>(0, 0);
        for (int i = 0; i<3; i++)
        {
            //double value = 0.0;
            int j =0;
            for (j=0; j<3; j++)
            {
                double subtraction = (bgrPixel[j] - mu.at<double>(0, j))*(bgrPixel[i] - mu.at<double>(0,i));
                
                sigma.at<double>(i,j) = sigma.at<double>(i,j) + subtraction;
                //printf("%f\n", subtraction);
            }
            //vals[i*3 + j] = value;
            //std::cout << sigma << std::endl;
        }
    }
    sigma = sigma / (samples - 1);
    //std::cout << sigma << std::endl;
    //cv::calcCovarMatrix(bgr_averages, samples, sigma, mu, 0, CV_COVAR_SCRAMBLED+CV_COVAR_SCALE);
}

Gaussian3D::~Gaussian3D()
{
    
}

Gaussian3D build_gaussian(std::string image_names[], int length)
{
    cv::Mat *pixels = new cv::Mat[length];
    int total_averaged_rgbs = 0;
    for (int index =0; index<length; index++) {
        std::string image = image_names[index];
        cv::Mat img;
        img = cv::imread(image);
        int count = 0;
        cv::Vec3d sum = cv::Vec3d(0.0, 0.0, 0.0);
        for (int i=0; i<img.rows; i++) {
            for (int j=0; j<img.cols; j++) {
                cv::Vec3b bgrPixel = img.at<cv::Vec3b>(i,j);
                sum[0] += (double)bgrPixel[0]/255;
                sum[1] += (double)bgrPixel[1]/255;
                sum[2] += (double)bgrPixel[2]/255;
                count++;
            }
            
        }
        sum = sum / count;
        pixels[index] = cv::Mat(sum, true);
        total_averaged_rgbs++;

    }
    Gaussian3D result = Gaussian3D(pixels, total_averaged_rgbs);
    delete [] pixels;
    return result;
}

double KL_Distance(Gaussian3D &gauss1, Gaussian3D &gauss2, cv::Mat &mu1_minus_m2, cv::Mat &gauss2invert_and_trace, cv::Mat &resulting)
{
    //This function has been optimized to use as few cv::Mat 3x3 as possible because they have slow constructors.
    double result;
    try {
        mu1_minus_m2 = (gauss1.mu-gauss2.mu);
        //std::cout << gauss1.mu << std::endl;
        // = cv::Mat(3,3,CV_64FC1);
        cv::invert(gauss2.sigma, gauss2invert_and_trace);
        //Right now it's gauss2invert
        resulting = gauss2invert_and_trace * mu1_minus_m2;
        cv::add(gauss2invert_and_trace, gauss1.sigma, gauss2invert_and_trace);
        double det_gauss2 = cv::determinant(gauss2.sigma);
        double det_gauss1 = cv::determinant(gauss1.sigma);
        //and now it's gauss2trace
        double gauss_trace_res = cv::trace(gauss2invert_and_trace)[0];
        result = 0.5*log(det_gauss2/det_gauss1) + gauss_trace_res - 3.0;
        
        // gauss2invert_and_trace is now used as transpose;
        cv::transpose(mu1_minus_m2, gauss2invert_and_trace);
        //mu1_minus_m2 is the result
        mu1_minus_m2 = resulting * gauss2invert_and_trace;
        result += mu1_minus_m2.at<double>(0,0);
        if (result != result)  // nan case
            return 0.0;
    } catch (int e)
    {
        return 0.0;
    }
    return result;
}
