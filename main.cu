// C++ imports
#include <iostream>
#include<cstdio>
#include <ctime>
#include <cmath>
#include "bits/time.h"

//#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include "opencv2/core/cuda.hpp"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include<cuda_runtime.h>

// namespaces
using namespace std;
using namespace cv;
#define PI 3.1415926


// int frameWidth = 640;
// int frameHeight = 480;
#define FRAMEWIDTH  640
#define FRAMEHEIGHT 480


// ---- GLOBAL VAR ----
int alpha_ = 90, beta_ = 90, gamma_ = 90;
int f_ = 500, dist_ = 500;



string type2str(int type) {
	string r;
  
	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);
  
	switch ( depth ) {
	  case CV_8U:  r = "8U"; break;
	  case CV_8S:  r = "8S"; break;
	  case CV_16U: r = "16U"; break;
	  case CV_16S: r = "16S"; break;
	  case CV_32S: r = "32S"; break;
	  case CV_32F: r = "32F"; break;
	  case CV_64F: r = "64F"; break;
	  default:     r = "User"; break;
	}
  
	r += "C";
	r += (chans+'0');
  
	return r;
}


__global__ void rotation_multiply_kernel(float *d_RX,float *d_RY,float *d_R, int N){
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += d_RX[ROW * N + i] * d_RY[i * N + COL];
        }
    }
    d_R[ROW * N + COL] = tmpSum;

}


void cuda_wrapper(const Mat &input, Mat &output){
    double focalLength, dist, alpha, beta, gamma; 

    alpha =((double)alpha_ -90) * PI/180;
    beta =((double)beta_ -90) * PI/180;
    gamma =((double)gamma_ -90) * PI/180;
    focalLength = (double)f_;
    dist = (double)dist_;

    Size input_size = input.size();
    double w = (double)input_size.width, h = (double)input_size.height;


    // Projecion matrix 2D -> 3D
    
    Mat A1 = (Mat_<float>(4, 3)<< 
        1, 0, -w/2,
        0, 1, -h/2,
        0, 0, 0,
        0, 0, 1 );
    
    
    // Rotation matrices Rx, Ry, Rz

    Mat RX = (Mat_<float>(4, 4) << 
        1, 0, 0, 0,
        0, cos(alpha), -sin(alpha), 0,
        0, sin(alpha), cos(alpha), 0,
        0, 0, 0, 1 );

    Mat RY = (Mat_<float>(4, 4) << 
        cos(beta), 0, -sin(beta), 0,
        0, 1, 0, 0,
        sin(beta), 0, cos(beta), 0,
        0, 0, 0, 1	);

    Mat RZ = (Mat_<float>(4, 4) << 
        cos(gamma), -sin(gamma), 0, 0,
        sin(gamma), cos(gamma), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1	);

    // R - rotation matrix
    Mat R = RX * RY * RZ;

    cout<< " R : "<< R << endl;


    // T - translation matrix
    Mat T = (Mat_<float>(4, 4) << 
        1, 0, 0, 0,  
        0, 1, 0, 0,  
        0, 0, 1, dist,  
        0, 0, 0, 1); 
    
    // K - intrinsic matrix 
    Mat K = (Mat_<float>(3, 4) << 
        focalLength, 0, w/2, 0,
        0, focalLength, h/2, 0,
        0, 0, 1, 0
        ); 


    Mat transformationMat = K * (T * (R * A1));
    //cout<< "transformationMat.rows : " << transformationMat.rows << "\ttransformationMat.cols : " << transformationMat.cols << endl;
    //cout << "tipo matrice di transformazione : "<< "CV_" + type2str( transformationMat.type()) << endl;

    warpPerspective(input, output, transformationMat, input_size, INTER_CUBIC | WARP_INVERSE_MAP);

    return;



}





void cuda_wrapper_new(const Mat &input, Mat &output){

    double focalLength, dist, alpha, beta, gamma; 

    alpha =((double)alpha_ -90) * PI/180;
    beta =((double)beta_ -90) * PI/180;
    gamma =((double)gamma_ -90) * PI/180;
    focalLength = (double)f_;
    dist = (double)dist_;

    Size input_size = input.size();
    double w = (double)input_size.width, h = (double)input_size.height;


    // Projecion matrix 2D -> 3D
    /*
    Mat A1 = (Mat_<float>(4, 3)<< 
        1, 0, -w/2,
        0, 1, -h/2,
        0, 0, 0,
        0, 0, 1 );
    */
   
    //DEFINIZIONE A VETTORE :
    float A1[12] {1, 0, -w/2, 0, 1, -h/2, 0, 0, 0, 0, 0, 1};
    
    float RX[16] = { 1, 0, 0, 0, 0, cos(alpha), -sin(alpha), 0 , 0, sin(alpha), cos(alpha), 0, 0, 0, 0, 1};

    float RY[16] = {cos(beta), 0, -sin(beta), 0,0, 1, 0, 0,sin(beta), 0, cos(beta), 0,0, 0, 0, 1};

    float RZ[16] = {cos(gamma), -sin(gamma), 0, 0,sin(gamma), cos(gamma), 0, 0,0, 0, 1, 0,0, 0, 0, 1};

    float R[16];

    float *d_RX, *d_RY, *d_RZ, *d_R, *d_XY;
    cudaMalloc((void **) &d_RX, sizeof(float)*4*4);
    cudaMalloc((void **) &d_RY, sizeof(float)*4*4);
    cudaMalloc((void **) &d_RZ, sizeof(float)*4*4);
    //alloco il vettore risultato
    cudaMalloc((void **) &d_R, sizeof(float)*4*4);
    cudaMalloc((void **) &d_XY, sizeof(float)*4*4);
    //copio i vettori
    cudaMemcpy(d_RX,RX,sizeof(float)*4*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_RY,RY,sizeof(float)*4*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_RY,RY,sizeof(float)*4*4,cudaMemcpyHostToDevice);

    //Setup the execution configuration
    // dim3 dimBlock(4,4);
    // dim3 dimGrid(1,1);
    dim3 threadsPerBlock(4, 4);
    dim3 blocksPerGrid(1, 1);

    //Launch the device computation threads!
    rotation_multiply_kernel<<<blocksPerGrid,threadsPerBlock>>>(d_RX,d_RY,d_XY,4);
    rotation_multiply_kernel<<<blocksPerGrid,threadsPerBlock>>>(d_XY,d_RZ,d_R,4);
    
    cudaDeviceSynchronize();

    // R - rotation matrix
    //Mat R = RX * RY * RZ;
    cudaMemcpy(R,d_R,sizeof(float)*4*4,cudaMemcpyDeviceToHost);

    Mat R_mat =(Mat_<float>(4,4));
    memcpy(R_mat.data, R, 16 *sizeof(float));
    /*
    Mat m = Mat(rows, cols, CV_8UC1); // initialize matrix of uchar of 1-channel where you will store vec data
    //copy vector to mat
    memcpy(m.data, vec.data(), vec.size()*sizeof(uchar)); // change uchar to any type of data values that you want to use instead
    */
    for(int i=0; i<16;i++){
        cout<<"R : " << R[i] << endl;
    }
     // Projecion matrix 2D -> 3D
    /*
    Mat A1 = (Mat_<float>(4, 3)<< 
        1, 0, -w/2,
        0, 1, -h/2,
        0, 0, 0,
        0, 0, 1 );

    // T - translation matrix
    Mat T = (Mat_<float>(4, 4) << 
        1, 0, 0, 0,  
        0, 1, 0, 0,  
        0, 0, 1, dist,  
        0, 0, 0, 1); 
    
    // K - intrinsic matrix 
    Mat K = (Mat_<float>(3, 4) << 
        focalLength, 0, w/2, 0,
        0, focalLength, h/2, 0,
        0, 0, 1, 0
        ); 
    */

    //Mat transformationMat = K * (T * (R * A1));
    //cout<< "transformationMat.rows : " << transformationMat.rows << "\ttransformationMat.cols : " << transformationMat.cols << endl;
    //cout << "tipo matrice di transformazione : "<< "CV_" + type2str( transformationMat.type()) << endl;

    //warpPerspective(input, output, transformationMat, input_size, INTER_CUBIC | WARP_INVERSE_MAP);
    

    output=input.clone();
    return;


}




int main(int argc, char const *argv[]) {
	
	if(argc > 2) {
      cerr << "Usage: " << argv[0] << " ' /path/to/video/ '  or nothing " << endl;
      cout << "Exiting...." << endl;
      return -1;
    }
    int flag=0;
    Mat image,output;
    

    VideoCapture capture;

    if (argc == 1){
        capture.open(0);
    }else{
        string filename = argv[1];
        capture.open(filename);
    }

    if(!capture.isOpened()) throw "Error reading video";

    

    /*
        definisco i parametri e le trackbar
    */

    namedWindow("Result", 1);

	createTrackbar("Alpha", "Result", &alpha_, 180);
	createTrackbar("Beta", "Result", &beta_, 180);
	createTrackbar("Gamma", "Result", &gamma_, 180);
	createTrackbar("f", "Result", &f_, 2000);
	createTrackbar("Distance", "Result", &dist_, 2000);




    cout << "Capture is opened" << endl;
    for(;;)
    {
        capture >> image;
        //stampo il tipo di immagine
        if(flag == 0){
            string ty = "CV_" + type2str( image.type() );
            cout << "tipo matrice :" << ty.c_str() <<endl;
            flag = 1;
        }
        resize(image, image,Size(FRAMEWIDTH, FRAMEHEIGHT));



		          
        cuda_wrapper_new(image, output);
        
        
        //per la visualizzazione 
        if(output.empty())
            break;
        //drawText(image);
        imshow("Result", output);
        if(waitKey(10) >= 0)
            break;
    }
    
    
    return 0;
}