#include "build_gaussian.h"

// README:
// My hand detection appearance based approach.
// Originally based off of realtime appearance based hand tracking
// This file contains everything that generates rgb gaussians, and then calculates the distances between those gaussians using the Kullback-Lieber distance (KLDistance function)
// Note: Originally I wrote this with cv::Mat and then replaced all cv::Mats with arrays of doubles and got a 500% speed increase!

Gaussian3D::Gaussian3D()
{
    
}

Gaussian3D::Gaussian3D(double **bgr_averages, int samples)
// Generate a gaussian in RGB or BGR space (3D) given an input of pixels as doubles.
{
    cv::Vec3d sum = cv::Vec3d(0, 0, 0);
    for (int i=0; i<samples; i++) {
        double *bgrPixel = bgr_averages[i];
        sum[0] += (double)bgrPixel[0];
        sum[1] += (double)bgrPixel[1];
        sum[2] += (double)bgrPixel[2];
    }
    cv::Vec3d averages = sum / samples;
    mu[0] = averages[0];
    mu[1] = averages[1];
    mu[2] = averages[2];
    //I rolled my own 3x3 covariance matrix calculator
    for (int i = 0; i<3; i++)
    {
        for (int j=0; j<3; j++)
        {
            sigma[i*3 +j] = 0.0;
        }
    }
    double subtraction;
    for (int index = 0; index< samples; index++)
    {
        double *bgrPixel = bgr_averages[index];
        for (int i = 0; i<3; i++)
        {
            //double value = 0.0;
            int j =0;
            for (j=0; j<3; j++)
            {
                subtraction = (bgrPixel[j] - mu[j])*(bgrPixel[i] - mu[i]);
                
                sigma[i*3 +j] = sigma[i*3 +j] + subtraction;
            }
        }
    }
    for (int i=0; i<9; i++)
        sigma[i] = sigma[i] / (samples - 1);
    //cv::calcCovarMatrix(bgr_averages, samples, sigma, mu, 0, CV_COVAR_SCRAMBLED+CV_COVAR_SCALE); TOO SLOW
}

Gaussian3D::~Gaussian3D()
{
    
}

Gaussian3D build_gaussian(std::string image_names[], int length, double **pixels)
// Builds the initial hand and background gaussians from files.
{
    int total_averaged_rgbs = 0;
    for (int index =0; index<length; index++) {
        std::string image = image_names[index];
        cv::Mat img;
        img = cv::imread(image);
        int count = 0;
        double *sum = new double[3];
        sum[0] = 0.0;
        sum[1] = 0.0;
        sum[2] = 0.0;
        for (int i=0; i<img.rows; i++) {
            for (int j=0; j<img.cols; j++) {
                cv::Vec3b bgrPixel = img.at<cv::Vec3b>(i,j);
                sum[0] += (double)bgrPixel[0]/255;
                sum[1] += (double)bgrPixel[1]/255;
                sum[2] += (double)bgrPixel[2]/255;
                count++;
            }
            
        }
        sum[0] = sum[0] / count;
        sum[1] = sum[1] / count;
        sum[2] = sum[2] / count;
        pixels[index] = sum;
        total_averaged_rgbs++;

    }
    Gaussian3D result = Gaussian3D(pixels, total_averaged_rgbs);
    return result;
}

double determinant3_3(const double sigma[])
//cals determininant of a 3x3 matrix
{
    const double ek = sigma[4]*sigma[8];
    const double fh = sigma[5]*sigma[7];
    const double kd = sigma[8]*sigma[3];
    const double fg = sigma[5]*sigma[6];
    const double dh = sigma[3]*sigma[7];
    const double eg = sigma[4]*sigma[6];
    return (1.0/(sigma[0]*(ek-fh) - sigma[1]*(kd-fg)+ sigma[2]*(dh-eg)));
}

void  inverse3_3(const double sigma[], double result[])
//inverse of a 3x3 matrix
{
    const double ek = sigma[4]*sigma[8];
    const double fh = sigma[5]*sigma[7];
    const double kd = sigma[8]*sigma[3];
    const double fg = sigma[5]*sigma[6];
    const double dh = sigma[3]*sigma[7];
    const double eg = sigma[4]*sigma[6];
    const double ch = sigma[2]*sigma[7];
    const double bk = sigma[1]*sigma[8];
    const double ak = sigma[0]*sigma[8];
    const double cg = sigma[2]*sigma[6];
    const double bg = sigma[1]*sigma[6];
    const double ah = sigma[0]*sigma[7];
    const double bf = sigma[1]*sigma[5];
    const double ce = sigma[2]*sigma[4];
    const double cd = sigma[2]*sigma[3];
    const double af = sigma[0]*sigma[5];
    const double ae = sigma[0]*sigma[4];
    const double bd = sigma[1]*sigma[3];
    const double detA = (1.0/(sigma[0]*(ek-fh) - sigma[1]*(kd-fg)+ sigma[2]*(dh-eg)));
    result[0] = detA*(ek-fh);
    result[3] = detA*(fg-kd);
    result[6] = detA*(dh-eg);
    result[1] = detA*(ch-bk);
    result[4] = detA*(ak-cg);
    result[7] = detA*(bg-ah);
    result[2] = detA*(bf-ce);
    result[5] = detA*(cd-af);
    result[8] = detA*(ae-bd);
}
void subtract3_1(const double vec1[], const double vec2[], double result[])
//3x1 vector subtraction
{
    result[0] = vec1[0] - vec2[0];
    result[1] = vec1[1] - vec2[1];
    result[2] = vec1[2] - vec2[2];
}

void add3_3(const double mat1[], const double mat2[], double result[])
// mat1 and mat are 3x3 result is 3x3
{
    for (int i=0; i<9; i++)
    {
        result[i] = mat1[i] + mat2[i];
    }
}

void multiply3_3by3_1(const double mat[], const double vec[], double result[])
//mat is 3x3 matrix, vec is 3x1 vector, result is 3x1 vector
{
    for (int i=0; i<3; i++)
    {
        result[i] = 0.0;
        for (int j=0; j<3; j++)
        {
            result[i] += vec[i]* mat[i*3 + j];
        }
    }
}
double trace(const double mat[])
// mat is 3x3
{
    return mat[0] + mat[4] + mat[8];
}

double KL_Distance(const Gaussian3D &gauss1, const Gaussian3D &gauss2, double mu1_minus_m2[], double gauss2invert_and_trace[], double resulting[])
// mu1_minus_mu2 is 3x1, gauss2_invert_and_trace is 3x3, resulting is 3x1
// computs Kullback-Leiber distance between two gaussians and returns the result (scalar double)
{
    //This function has been optimized to use as few double arrays as possible because this function was slow
    double result;
    try {
        subtract3_1(gauss1.mu, gauss2.mu, mu1_minus_m2);
        inverse3_3(gauss2.sigma, gauss2invert_and_trace);
        multiply3_3by3_1(gauss2invert_and_trace, mu1_minus_m2, resulting);
        add3_3(gauss2invert_and_trace, gauss1.sigma, gauss2invert_and_trace);
        double det_gauss2 = determinant3_3(gauss2.sigma);
        double det_gauss1 = determinant3_3(gauss1.sigma);
        double gauss_trace_res = trace(gauss2invert_and_trace);
        result = 0.5*log(det_gauss2/det_gauss1) + gauss_trace_res - 3.0;
        
        gauss_trace_res = resulting[0] * mu1_minus_m2[0] + resulting[1] * mu1_minus_m2[1] + resulting[2] * mu1_minus_m2[2];
        result += gauss_trace_res;
        if (result != result)  // nan case
            return 0.0;
    } catch (int e)
    {
        return 0.0;
    }
    return result;
}
