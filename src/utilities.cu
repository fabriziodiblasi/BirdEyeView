#include "../lib/utilities.h"

using namespace std;
using namespace cv;

typedef unsigned char uchar;

cudaError_t cudaStatus;


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
    cudaThreadSynchronize();
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

/**
    * converte un vettore in un oggetto Mat
    * src : array
    * dst : Mat
*/

void arrayToMat(cv::Mat &mat, const float *array, int numElem){
    memcpy(mat.ptr(),array,numElem * sizeof(float));
}

/**
    * converte un oggetto Mat in un array
    * src : Mat
    * dst : array
*/
void matToArray(float *array, const cv::Mat &mat, int rig, int col){
    memcpy(array, mat.ptr(), rig * col * sizeof(float));
    /*for(int i = 0; i < rig; i++){
        for(int j = 0; j < col; j++){
            array[i * col + j] = mat.at<float>(i,j);
            //cout << matrice[i * col + j] << "\t";
        }
//        cout<<"\n";
    }
//    cout<<"\n\n";
    */
}


// Mat A = immagine da traslare
// Mat H = matrice di transformazione (3 X 3)
// --------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------- warping in CPU con OpenCV -------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------

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

    int homeX, homeY;

    // int homeX=Idx % Numcols;
    // int homeY=Idx / Numcols;
    // cout << H << endl;

    for (Idx=0; Idx < size; ++Idx ){

        homeX=Idx % Numcols;
        homeY=Idx / Numcols;

        // float x  = (H.at<float>(0,0) * (homeX)) +( H.at<float>(0,1) * (homeY)) + ( H.at<float>(0,2) * 1) ;
        // float y  = (H.at<float>(1,0) * (homeX)) +( H.at<float>(1,1) * (homeY)) + ( H.at<float>(1,2) * 1) ;
        // float s  = (H.at<float>(2,0) * (homeX)) +( H.at<float>(2,1) * (homeY)) + ( H.at<float>(2,2) * 1) ;

        float x  = (H.at<float>(0,0) * (homeX)) +( H.at<float>(0,1) * (homeY)) +  H.at<float>(0,2) ;
        float y  = (H.at<float>(1,0) * (homeX)) +( H.at<float>(1,1) * (homeY)) +  H.at<float>(1,2) ;
        float s  = (H.at<float>(2,0) * (homeX)) +( H.at<float>(2,1) * (homeY)) +  H.at<float>(2,2) ;


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
        if( y >= A.rows || y < 0 || x >= A.cols || x < 0){
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


// --------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------- warping in cuda -------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------------



__global__ void calc_tranf_array(float *H, int *transfArray, int numARows, int numAColumns) {
    //int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int MaxX,MaxY = -1000;
    int MinX,MinY =  1000;
    
    int homeX, homeY;
    
    //if (row < numARows && col < numAColumns) {
    if (idx < numARows * numAColumns) {
        homeX=idx % numAColumns;
        homeY=idx / numAColumns;

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
        if( y >= numARows || y<0 || x >= numAColumns ||  x < 0){
            transfArray[idx]  = -1;
            // cout << "x= " << x << "y= "<< y << endl;
        }else{
            transfArray[idx] = (y * numAColumns + x); 
        }           
    }
}



cudaError_t calculateTransferArray(Mat H, int *TransArry, int rows, int cols){
    int size = rows * cols;
    cudaError_t cudaStatus;
    float *d_H;
    float *vecH = (float *)malloc(sizeof(float) * H.rows * H.cols);
    int *d_T;
   
    // cout <<" \n prima della copia della matrice H \n";
    
    // cout << "tipo matrice H :" << "CV_" + type2str(H.type()) <<endl;

    matToArray(vecH, H, H.rows, H.cols);

    // cout <<" \n DOPO della copia della matrice H \n";
 

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
    
    // cout <<" \n copio i vettori \n";

    //copio i vettori
    cudaStatus = cudaMemcpy(d_H,vecH,sizeof(float)*H.rows * H.cols,cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaMemCpy failed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorWarp;
    }
    cudaStatus = cudaMemset(d_T, 0, sizeof(int) * size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaMemSetfailed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorWarp;
    }
    // 
    // dim3 blockDim(16, 16);
    // dim3 gridDim(ceil(((float)numAColumns) / blockDim.x),ceil(((float)numBRows) / blockDim.y));
    // 
    //ceil(n/256.0),256
    //dim3 DimGrid(ceil(size/256.0),1,1);
    //dim3 DimBlock(256,1,1);

    // cout <<" \n richiamo il kernell \n";
    calc_tranf_array<<<ceil(size/256.0),256>>>(d_H, d_T, rows, cols);
    cudaThreadSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorWarp;
    }

    // cout <<" \n copio il risultato del kernel \n";
    cudaStatus = cudaMemcpy(TransArry,d_T,sizeof(int) * size,cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaMemCpy failed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorWarp;
    }

ErrorWarp:
    //cout<< "****** ERRORE CUDA ****** : " << cudaStatus << endl;
    cudaFree(d_H);
    cudaFree(d_T);
    
    return cudaStatus;
    

}





// lavora correttamente, manca la parralellizzazione dell'ultimo for

cudaError_t warpPerspectiveCUDA(Mat input, Mat &output, const Mat H){
    // allocate array of all locations
    int Numrows = input.rows;
    int Numcols = input.cols;
    int channels   = input.channels();
    // cout << "rows " << Numrows << "col " << Numcols << "channels " << channels <<endl;
    int size = Numrows*Numcols;
    int MaxX,MaxY = -1000;
    int MinX,MinY =  1000;
    int *TransArry = (int *)malloc(sizeof(int)*size);
    int Idx;
    int homeX, homeY;
    Mat tranImg;
    
        
    calculateTransferArray(H,TransArry,Numrows, Numcols);


    //input.copyTo(tranImg);
    tranImg = input.clone();
    tranImg = tranImg - tranImg;
    
    // Remap Image
    for (Idx=0; Idx < size; Idx ++ ){

        homeX=Idx % Numcols;
        homeY=Idx / Numcols;                
        //tranImg.at<uchar>(homeY, homeX) =0;
        if(TransArry[Idx] != -1){   
            //cout << "Index " << Idx << "Passed " << endl;
            int newhomeX=TransArry[Idx] % Numcols; // Col ID
            int newhomeY=TransArry[Idx] / Numcols;  // Row ID

            tranImg.at<uchar>(newhomeY, (newhomeX*channels)) = input.at<uchar>(homeY, homeX*channels);
            if(channels>1)
                tranImg.at<uchar>(newhomeY, newhomeX*channels+1) = input.at<uchar>(homeY, homeX*channels+1);
            if(channels>2)
                tranImg.at<uchar>(newhomeY, newhomeX*channels+2) = input.at<uchar>(homeY, homeX*channels+2);
            // if (!(Idx%100)){
                // imshow("inside", tranImg);
                // waitKey(1);
                // }
            }
    }
    //cout << tranImg << endl;  
    
    output = tranImg.clone();
    //return tranImg;




}

