#include "build_gaussian.h"

Gaussian3D::Gaussian3D()
{
    
}

Gaussian3D::Gaussian3D(cv::Mat *bgr_averages, int samples)
{
    cv::Vec3f sum = cv::Vec3f(0, 0, 0);
    for (int i=0; i<samples; i++) {
        cv::Vec3f bgrPixel = bgr_averages[i].at<cv::Vec3f>(0, 0);
        sum[0] += (float)bgrPixel[0] / samples;
        sum[1] += (float)bgrPixel[1] / samples;
        sum[2] += (float)bgrPixel[2] / samples;
    }
    cv::Vec3f averages = sum / samples;
    mu = cv::Mat(averages);
    sigma = cv::Mat(3,3, CV_64F);
    cv::calcCovarMatrix(bgr_averages, samples, sigma, mu, 0);
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
        pixels[total_averaged_rgbs] = cv::Mat(sum);
        total_averaged_rgbs++;

    }
    Gaussian3D result = Gaussian3D(pixels, total_averaged_rgbs);
    delete [] pixels;
    return result;
}

double KL_Distance(Gaussian3D gauss1, Gaussian3D gauss2)
{
    try {
        cv::Mat transposed = cv::Mat(3,1,CV_64F);
        cv::transpose((gauss1.mu-gauss2.mu), transposed);
        cv::Mat gauss2invert = cv::Mat(3,3,CV_64F);
        cv::invert(gauss2.sigma, gauss2invert);
        double result = 0.5*log(cv::determinant(gauss2.sigma)/cv::determinant(gauss1.sigma)) + cv::trace(gauss2invert * gauss1.sigma)[0] - 3.0;
        cv::Mat mu1_minus_m2 = cv::Mat(1,3,CV_64F);
        cv::subtract(gauss1.mu, gauss2.mu, mu1_minus_m2);
        cv::Mat resulting = cv::Mat(3,1, CV_64F);
        cv::multiply(mu1_minus_m2, gauss2invert, resulting);
        cv::Mat final = cv::Mat(1,1, CV_64F);
        cv::multiply(resulting, transposed, final);
        result += final.at<double>(0,0);
    } catch (int e)
    {
        return 0;
    }
}
