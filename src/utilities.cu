#include "../lib/utilities.h"

using namespace std;
using namespace cv;

void stampaMatrice(float *matrice, int rig, int col){
    //stampa a matrice
    for(int i = 0; i < rig; i++){
        for(int j = 0; j < col; j++){
            
            cout << matrice[i * col + j] << "\t";
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



void arrayToMat(cv::Mat &mat, float *array, int numElem){
    memcpy(mat.ptr(),array,numElem * sizeof(float));
}


void matToArray(float *array, cv::Mat &mat, int numElem){
    memcpy(array,mat.ptr(),numElem * sizeof(float));
}


// implementazione manuale di

//---------------------------- cv::warpPerspective() ---------------------------------------

//  warpPerspective(InputArray src, OutputArray dst, InputArray M, Size dsize, int flags=INTER_LINEAR, int borderMode=BORDER_CONSTANT, const Scalar& borderValue=Scalar())

// M è sempre 3 x 3 
// dsize è la dimensione dell'immagine sorgente
// cudaError_t warpPerspectiveCPU(float *src, float *dst, float *m, int numSrcRows,int numSrcColumns){
    
//     for(int i = 0; i < rig; i++){
//         for(int j = 0; j < col; j++){
            
//             //cout << matrice[i * col + j] << "\t";
//             dst[i * col *j] = src[i * col *j]
//         }
//         cout<<"\n";
//     }
// }


// Mat A = immagine da traslare
// Mat H = matrice di transformazione (3 X 3)

Mat warpPerspectiveCPU(Mat A, Mat H){
    // allocate array of all locations
    int Numrows = A.rows;
    int Numcols = A.cols;
    int channels   = A.channels();
    // cout << "rows " << Numrows << "col " << Numcols << "channels " << channels <<endl;
    int size = Numrows*Numcols;
    int MaxX,MaxY = -1000;
    int MinX,MinY =  1000;
    int *TransArry = (int *)malloc(sizeof(int)*size);
    int Idx;

    int homeX=Idx % Numcols;
    int homeY=Idx / Numcols;
    // cout << H << endl;

    //waitKey();         
    for (Idx=0; Idx < size; ++Idx ){

        homeX=Idx % Numcols;
        homeY=Idx / Numcols;

        float x  = (H.at<float>(0,0) * (homeX)) +( H.at<float>(0,1) * (homeY)) + ( H.at<float>(0,2) * 1) ;
        float y  = (H.at<float>(1,0) * (homeX)) +( H.at<float>(1,1) * (homeY)) + ( H.at<float>(1,2) * 1) ;
        float s  = (H.at<float>(2,0) * (homeX)) +( H.at<float>(2,1) * (homeY)) + ( H.at<float>(2,2) * 1) ;

        // cout << " x = " << x << " y= " << y << " s= " << s;
        x = floor(x/s);

        y = floor(y/s);

        // for the first col in TransMatrix
        if (homeX ==0){
            if (x > MaxX) MaxX = x;
            if (x < MinX) MinX = x; 
        }

        //for thee first row in TransMatrix
        if (homeY ==0){
            if (y > MaxY) MaxY = y;
            if (y < MinY) MinY = y;
        }
        if((y)>=A.rows || (y)<0 || (x)>=A.cols || (x)<0){
            TransArry[Idx]  = -1;
            // cout << "x= " << x << "y= "<< y << endl;
        }else{
            TransArry[Idx] = (y * Numcols + x); 
        }           

        //cout << Numcols << endl;
        // cout <<     "New index of " << Idx << "is " << TransArry[Idx] << endl;
    }

    Mat   tranImg ;

    A.copyTo(tranImg);
    tranImg = tranImg - tranImg;
    // cout <<     "Rows" << tranImg.rows << "cols" << tranImg.cols << "cha" <<  A.channels() << endl;


    //waitKey();
    // Remap Image
    for (Idx=0; Idx < size; Idx ++ ){

        homeX=Idx % Numcols;
        homeY=Idx / Numcols;                
        //tranImg.at<uchar>(homeY, homeX) =0;
        if(TransArry[Idx] != -1){   
            //cout << "Index " << Idx << "Passed " << endl;
            int newhomeX=TransArry[Idx] % Numcols; // Col ID
            int newhomeY=TransArry[Idx] / Numcols;  // Row ID


            // cout << "Index is " << Idx << endl;
            // cout << "HomeX is " << homeX << " and HomeY is " << homeY << endl;
            // cout << "New Index is " << TransArry[Idx] << endl;
            // cout << "New HomeX is " << newhomeX << " and New HomeY is " << newhomeY << endl;   
            // cout << "*****************************************"<< endl; 
            // if (!(Idx%100)) sleep(20);  

            tranImg.at<uchar>(newhomeY, (newhomeX*channels)) = A.at<uchar>(homeY, homeX*channels);
            if(channels>1)
                tranImg.at<uchar>(newhomeY, newhomeX*channels+1) = A.at<uchar>(homeY, homeX*channels+1);
            if(channels>2)
                tranImg.at<uchar>(newhomeY, newhomeX*channels+2) = A.at<uchar>(homeY, homeX*channels+2);
            // if (!(Idx%100)){
                // imshow("inside", tranImg);
                // waitKey(1);
                // }
            }
    }
    //cout << tranImg << endl;  
    
    return tranImg;

}

