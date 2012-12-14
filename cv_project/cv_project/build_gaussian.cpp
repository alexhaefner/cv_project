#include "build_gaussian.h"

Gaussian3D::Gaussian3D()
{
    
}

Gaussian3D::Gaussian3D(cv::Mat *bgr_averages, int samples)
{
    cv::Vec3f sum = cv::Vec3f(0, 0, 0);
    for (int i=0; i<samples; i++) {
        cv::Vec3f bgrPixel = bgr_averages[i].at<cv::Vec3f>(0, 0);
        sum[0] += (float)bgrPixel[0];
        sum[1] += (float)bgrPixel[1];
        sum[2] += (float)bgrPixel[2];
    }
    cv::Vec3f averages = sum / samples;
    mu = cv::Mat(averages, CV_64FC1);
    sigma = cv::Mat(3,3, CV_64FC1);
    //calcCovarMatrix doesn't works so I did it myself
    for (int index = 0; index< samples; index++)
    {
        cv::Vec3f bgrPixel = bgr_averages[index].at<cv::Vec3f>(0, 0);
        for (int i = 0; i<3; i++)
        {
            for (int j=0; j<3; j++)
            {
                double subtraction = (bgrPixel[j] - mu.at<double>(0, j))*(bgrPixel[i] - mu.at<double>(0,i));
                sigma.at<double>(i,j) = sigma.at<double>(i,j) + subtraction;
                //printf("%f\n", bgrPixel[j]);
            }
        }
    }
    //std::cout << sigma << std::endl;
    sigma = sigma / (samples - 1);
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
        cv::Vec3f sum = cv::Vec3f(0.0, 0.0, 0.0);
        for (int i=0; i<img.rows; i++) {
            for (int j=0; j<img.cols; j++) {
                cv::Vec3b bgrPixel = img.at<cv::Vec3b>(i,j);
                sum[0] += (float)bgrPixel[0]/255;
                sum[1] += (float)bgrPixel[1]/255;
                sum[2] += (float)bgrPixel[2]/255;
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

double KL_Distance(Gaussian3D gauss1, Gaussian3D gauss2)
{
    double result;
    try {
        cv::Mat transposed = cv::Mat(3,1,CV_64FC1);
        cv::transpose((gauss1.mu-gauss2.mu), transposed);
        cv::Mat gauss2invert = cv::Mat(3,3,CV_64FC1);
        cv::invert(gauss2.sigma, gauss2invert);
        cv::Mat gauss_trace = cv::Mat(3,3, CV_64FC1);
        cv::add(gauss2invert, gauss1.sigma, gauss_trace);
        //std::cout << gauss2.sigma << std::endl;
        //std::cout << gauss1.sigma << std::endl;
        double det_gauss2 = cv::determinant(gauss2.sigma);
        double det_gauss1 = cv::determinant(gauss1.sigma);
        double gauss_trace_res = cv::trace(gauss_trace)[0];
        result = 0.5*log(det_gauss2/det_gauss1) + gauss_trace_res - 3.0;
        cv::Mat mu1_minus_m2 = cv::Mat(1,3,CV_64FC1);
        cv::subtract(gauss1.mu, gauss2.mu, mu1_minus_m2);
        mu1_minus_m2.assignTo(mu1_minus_m2, CV_64FC1);
        //std::cout << mu1_minus_m2 << std::endl;
        //std::cout << gauss2invert << std::endl;
        //cv::Mat resulting = cv::Mat(3,1,CV_64FC1);
        cv::Mat resulting = gauss2invert * mu1_minus_m2;
        resulting.assignTo(resulting, CV_64FC1);
        transposed.assignTo(transposed, CV_64FC1);
        cv::Mat final = resulting * transposed;
        result += final.at<double>(0,0);
        if (result != result)  // nan case
            return 0.0;
    } catch (int e)
    {
        return 0.0;
    }
    return result;
}
