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


void stampaMatrice(float *matrice, int rig, int col);

/**

*/
__global__ void generic_mat_mul(float *A, float *B, float *C, int numARows,int numAColumns, int numBRows, int numBColumns);

/**
    A * B = C
    N = numero di colonne
*/

cudaError_t matrixMultiplication(float *A, float *B, float *C, int numARows,int numAColumns, int numBRows, int numBColumns);