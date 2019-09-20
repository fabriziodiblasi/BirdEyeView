#include "../lib/utilities.h"
using namespace std;
using namespace cv;


__global__ void pixelRemappingCudaKernel(cv::cuda::PtrStepSz<uchar3> src,
                                            cv::cuda::PtrStepSz<uchar3> out,
                                            size_t step,
                                            int numChannel,
                                            float *H, 
                                            int *transfArray, 
                                            int numRows, 
                                            int numCols){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int MaxX,MaxY = -1000;
    int MinX,MinY =  1000;
    uchar3 pxval;
    int homeX, homeY;
    int newhomeX, newhomeY;
    
    //if (row < numARows && col < numAColumns) {
    if (idx < numRows * numCols) {
        homeX=idx % numCols;
        homeY=idx / numCols;

        float x  = (H[0] * (homeX)) +( H[1] * (homeY)) +  H[2] ;
        float y  = (H[3] * (homeX)) +( H[4] * (homeY)) +  H[5] ;
        float s  = (H[6] * (homeX)) +( H[7] * (homeY)) +  H[8] ;

        // cout << " x = " << x << " y= " << y << " s= " << s;
        x = floor(x/s);

        y = floor(y/s);

        // for the first col in TransMatrix
        if (homeX == 0){
            if (x > MaxX) MaxX = x;
            if (x < MinX) MinX = x; 
        }

        //for thee first row in TransMatrix
        if (homeY == 0){
            if (y > MaxY) MaxY = y;
            if (y < MinY) MinY = y;
        }
        if( y >= numRows || y<0 || x >= numCols ||  x < 0){
            transfArray[idx]  = -1;
            // cout << "x= " << x << "y= "<< y << endl;
        }else{
            transfArray[idx] = (y * numCols + x); 
            
            //------pezzo aggiunto 
            
            homeX=idx % numCols;
            homeY=idx / numCols;
            newhomeX = transfArray[idx] % numCols; // Col ID
            newhomeY = transfArray[idx] / numCols;  // Row ID
            //srcval = src(homeY, homeX*numChannel);

            pxval = src(homeY, homeX );
            out(newhomeY, newhomeX ) = pxval;
        }

   

    }

}






cv::cuda::GpuMat input, output;
cudaError_t warpPerspectiveRemappingCUDA(Mat inputFrame, Mat &outputFrame, Mat H){
    cudaError_t cudaStatus;
    int size = inputFrame.rows * inputFrame.cols;
    int channels   = input.channels();
    int *TransArry = (int *)malloc(sizeof(int)*size);
    float *vecH = (float *)malloc(sizeof(float) * H.rows * H.cols);
    float *d_H;
    int *d_T;
    
    
    matToArray(vecH, H, H.rows, H.cols);

    // ALLOCO LA MEMORIA sulla GPU
 

    cudaStatus = cudaMalloc((void **) &d_H, sizeof(float)*H.rows * H.cols);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto ErrorWarp;
    }
    
    cudaStatus = cudaMalloc((void **) &d_T, sizeof(int) * size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto ErrorWarp;
    }

    // COPIO I DATI SULLA GPU

    //copio sul device la matrice H
    cudaStatus = cudaMemcpy(d_H,vecH,sizeof(float)*H.rows * H.cols,cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaMemCpy failed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorWarp;
    }

    //copio sul device lo spazio per il vettor di trasposizione
    cudaStatus = cudaMemset(d_T, 0, sizeof(int) * size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaMemSetfailed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorWarp;
    }

    //carico il frame in input sulla GPU

    input.upload(inputFrame);
    //output.create(cv::Size(image.rows, image.cols), CV_8UC3);
    output = input.clone();
    output.setTo(Scalar::all(0));


    // cout <<" \n RICHIAMO IL KERNEL \n";


    pixelRemappingCudaKernel<<<ceil(size/1024.0),1024>>>(input, output, inputFrame.step, inputFrame.channels(), d_H, d_T, inputFrame.rows, inputFrame.cols);

    
    
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorWarp;
    }

    output.download(outputFrame);

    cudaFree(d_H);
    cudaFree(d_T);
    return cudaStatus;

ErrorWarp:
    cudaFree(d_H);
    cudaFree(d_T);
    return cudaStatus;
}

