// C++ imports
#include <iostream>
#include<cstdio>
#include <ctime>
#include <cmath>
#include <vector>
#include <array>
#include <chrono>
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


std::string type2str(int type);

void stampaMatrice(float *matrice, int rig, int col);

/**
__global__ void generic_mat_mul(float *A, float *B, float *C, int numARows,int numAColumns, int numBRows, int numBColumns);
*/

/**
    A * B = C
    N = numero di colonne
*/

cudaError_t matrixMultiplication(float *A, float *B, float *C, int numARows,int numAColumns, int numBRows, int numBColumns);

/*
void arrayToMat(cv::Mat &mat, float *array, int numElem);

void matToArray(float *array, cv::Mat &mat, int numElem);
*/


/**
    * converte un vettore in un oggetto Mat
    * src : array
    * dst : Mat
*/
void arrayToMat(cv::Mat &mat, const float *array, int numElem);
/**
    * converte un oggetto Mat in un array
    * src : Mat
    * dst : array
*/
// void matToArray(float *array, const cv::Mat &mat, int numElem);
void matToArray(float *array, const cv::Mat &mat, int rig, int col);



cv::Mat warpPerspectiveCPU(cv::Mat A, cv::Mat H);

cudaError_t warpPerspectiveCUDA(cv::Mat input, cv::Mat &output, const cv::Mat H);

cudaError_t warpPerspectiveRemappingCUDA(cv::Mat input, cv::Mat &output, const cv::Mat H);

cudaError_t calculateTransferArray(cv::Mat H, int *TransArry, int rows, int cols);
