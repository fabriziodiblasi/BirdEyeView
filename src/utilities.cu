#include "../lib/utilities.h"

using namespace std;
using namespace cv;

void stampaMatrice(float *matrice, int rig, int col){
    int idx;  
    //stampa a matrice
    for(int i = 0; i < rig; i++){
        for(int j = 0; j < col; j++){
            if (i == 0){
                idx = j;
            }else{
                idx = i * col + j;
            }
            cout << matrice[idx] << "\t";
        }
        cout<<"\n";
    }
    cout<<"\n\n";
    
}

/**

*/
__global__ void generic_mat_mul(float *A, float *B, float *C, int numARows,int numAColumns, int numBRows, int numBColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numARows && col < numBColumns) {
        float sum = 0;
        for (int ii = 0; ii < numAColumns; ii++) {
            sum += A[row * numAColumns + ii] * B[ii * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
    }
}

/**
    A * B = C
    N = numero di colonne
*/

cudaError_t matrixMultiplication(float *A, float *B, float *C, int numARows,int numAColumns, int numBRows, int numBColumns){
    cudaError_t cudaStatus;
    //@@ Initialize the grid and block dimensions here
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((float)numAColumns) / blockDim.x),ceil(((float)numBRows) / blockDim.y));
    
    /*
    dim3 blockDim(numARows, numBColumns);
    dim3 gridDim(1, 1);
    //<<<blocksPerGrid,threadsPerBlock>>>
    if (numARows * numBColumns > 512){
        blockDim.x = 512;
        blockDim.y = 512;
        gridDim.x = ceil(double(numBColumns)/double(blockDim.x));
        gridDim.y = ceil(double(numARows)/double(blockDim.y));
    }
    */ 
    
    float *d_A, *d_B, *d_C;

    cudaStatus = cudaMalloc((void **) &d_A, sizeof(float)*numARows*numAColumns);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void **) &d_B, sizeof(float)*numBRows*numBColumns);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void **) &d_C, sizeof(float)*numARows * numBColumns);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    
    //copio i vettori
    cudaMemcpy(d_A,A,sizeof(float)*numARows*numAColumns,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,sizeof(float)*numBRows*numBColumns,cudaMemcpyHostToDevice);
    
   

    cudaMemset(d_C, 0, numARows * numBColumns * sizeof(float));

    generic_mat_mul<<<gridDim, blockDim>>>(d_A, d_B, d_C, numARows, numAColumns, numBRows, numBColumns);
    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(C, d_C,numARows * numBColumns * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    //@@ Free the GPU memory here
Error:
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return cudaStatus;
    
}