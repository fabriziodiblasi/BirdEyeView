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

void stampaMatrice(float *matrice){
    int idx;  
    //stampa a matrice
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            if (i == 0){
                idx = j;
            }else{
                idx = i * 4 + j;
            }
            cout << matrice[idx] << "\t";
        }
        cout<<"\n";
    }
    cout<<"\n\n";
    
}



__global__ void _matrix_Multiplication_Kernel_(float* A, float* B, float* C, int N) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < N && COL < N) {
        float tmpSum = 0;
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
        C[ROW * N + COL] = tmpSum;
    }
    
}


void matrixMultiplication(float *A, float *B, float *C, int N){

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block

    //stampaMatrice(A);
    //stampaMatrice(B);
    

    //@@ Initialize the grid and block dimensions here
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((float)N) / blockDim.x), ceil(((float)N) / blockDim.y));
    
    cudaMemset(C, 0, N * N * sizeof(float));
    _matrix_Multiplication_Kernel_<<<gridDim, blockDim>>>(A, B, C, N);
    //ceil(n/256.0),256
    //_matrix_Multiplication_Kernel_<<<ceil(N/256.0),256>>>(A, B, C, N);
}

int main(int argc, const char *argv[]) {
    float RX[16] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7};

    float RY[16] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7};

    float RZ[16] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7};

    
    float *d_RX, *d_RY, *d_RZ, *d_R, *d_XY;

    float *ris;

    cudaMalloc((void **) &d_RX, sizeof(float)*4*4);
    cudaMalloc((void **) &d_RY, sizeof(float)*4*4);
    cudaMalloc((void **) &d_RZ, sizeof(float)*4*4);
    //alloco il vettore risultato
    cudaMalloc((void **) &d_R, sizeof(float)*4*4);
    cudaMalloc((void **) &d_XY, sizeof(float)*4*4);
    //copio i vettori
    cudaMemcpy(d_RX,RX,sizeof(float)*4*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_RY,RY,sizeof(float)*4*4,cudaMemcpyHostToDevice);
    

    matrixMultiplication(d_RX,d_RY,d_XY,4);
    
    //in d_XY ottengo il risultato della moltiplicazione

    cudaMemcpy(ris,d_XY,sizeof(float)*4*4,cudaMemcpyHostToDevice);

    // for (int i=0; i<16;i++){
    //     cout << RX[i] << " ";
    // }

    // cout << "\n\n\n";

    
    stampaMatrice(ris);
    
}