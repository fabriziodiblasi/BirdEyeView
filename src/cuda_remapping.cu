#include "../lib/utilities.h"
using namespace std;
using namespace cv;

//typedef unsigned char uchar;


vector<Mat> imageSplitting(Mat image){
     
    Mat Bands[3],merged;
    split(image, Bands);
    vector<Mat> channels = {Bands[0],Bands[1],Bands[2]};
    // merge(channels,merged);
    // imshow("red", Bands[2]);
    // imshow("blue", Bands[0]);
    // imshow("green",Bands[1]);
    // imshow("merged",merged);

    return channels;
}


void cvMatToVector(Mat mat, uchar *v){
    cout << "cvMatToVector image :\n \t \ttipo matrice :" << "CV_" + type2str(mat.type()) <<endl;

    std::vector<uchar> array;
    cout << "mat size : " << mat.rows * mat.cols <<"\n";
    
    if (mat.isContinuous()) {
        array.assign(mat.data, mat.data + mat.total());
    }else{
        for (int i = 0; i < mat.rows; ++i) {
            array.insert(array.end(), mat.ptr<uchar>(i), mat.ptr<uchar>(i)+mat.cols);
        }
    }
    /*
    for(int r = 0; r < mat.rows; r++){
        for(int c = 0; c < mat.rows; c++){
            array.push_back(mat.at<cv::Vec3b>(r,c)[0]);
            array.push_back(mat.at<cv::Vec3b>(r,c)[1]);
            array.push_back(mat.at<cv::Vec3b>(r,c)[2]); 
        }
    }
    */

    v = (uchar *)malloc(sizeof(uchar)*array.size());
    cout << "array size : " << array.size() <<endl;
    //getchar();
    for (int i = 0; i <array.size(); i++){
        v[i] = array[i];
    }

}

std::vector<uchar> multiChannelMatToVector(Mat mat){
    std::vector<uchar> array;
    cout << "size : "<< mat.size() << " num canali : "<< mat.channels() << endl;
    /*
    for(int rows = 0; rows < mat.rows; rows++){
        for(int cols = 0; cols < mat.cols * mat.channels(); cols++){
            array.push_back(mat.at<uchar>(rows, cols));

        }
    }
    */
    if (mat.isContinuous()) {
        array.assign(mat.data, mat.data + mat.total());
    }else{
        for (int i = 0; i < mat.rows; ++i) {
            array.insert(array.end(), mat.ptr<uchar>(i), mat.ptr<uchar>(i)+mat.cols);
        }
    }
    
    /*
    cout << "\nmat.at<uchar>(0, 0) = " << (int) mat.at<uchar>(0, 0) 
         << " mat.at<uchar>(0, 1) = " << (int) mat.at<uchar>(0, 1) 
         << " mat.at<uchar>(0, 2) = " << (int) mat.at<uchar>(0, 2) << endl;
    cout << "\n v[0] = " << (int) array[0] 
         << " v[1] = " << (int) array[1]
         << " v[2] = " << (int) array[2] << endl;

    cout << "\nmat.at<uchar>(0, 3) = " << (int) mat.at<uchar>(0, 0) 
         << " mat.at<uchar>(0, 4) = " << (int) mat.at<uchar>(0, 1) 
         << " mat.at<uchar>(0, 5) = " << (int) mat.at<uchar>(0, 2) << endl;
    cout << "\n v[3] = " << (int) array[0] 
         << " v[4] = " << (int) array[1]
         << " v[5] = " << (int) array[2] << endl;
    */
    return array;
}

void stdVectorToArray(std::vector<uchar> &input, uchar *out){
    //out = (uchar *)malloc(sizeof(uchar)*input.size());
    cout << " elementi del vettore : "<< input.size() << endl;
    for(int i = 0; i < input.size(); i++){
        out[i] = input[i];
    }
}

__global__ void new_remapping_kernel(cv::cuda::PtrStepSz<uchar3> src, int numRows, int numCols, size_t step, int numChannel, int *tranfArray, cv::cuda::PtrStepSz<uchar3> out){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int size = numRows * numCols;
    int idx;
    int homeX, homeY;
    int newhomeX, newhomeY;
    //int iStep = step / sizeof(uchar3);
    //int oStep = step / sizeof(uchar3);
    uchar3 pxval;
    //uchar3 srcval, outval;
    //vector<float> result(3, 0);
    idx = row * numCols + col;
    //if ((row < numRows) && (col < numCols))
    if (idx < numRows *numCols)
    {
         
        
        homeX=idx % numCols;
        homeY=idx / numCols; 
        if(tranfArray[idx] != -1 ){ 
            newhomeX = tranfArray[idx] % numCols; // Col ID
            newhomeY = tranfArray[idx] / numCols;  // Row ID
            //srcval = src(homeY, homeX*numChannel);

            pxval = src(homeY, homeX );
            out(newhomeY, newhomeX ) = pxval;
                        
        }
        

    }
}
std::ofstream os_remapping;
//std::chrono::steady_clock::time_point begin,end;
cv::Mat remappingMultiChannelImage(Mat image, int *tranfArray){
    cudaError_t cudaStatus;
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil((float)image.cols / blockDim.x), ceil((float)image.cols / blockDim.y), 1);
    //dim3 gridDim(ceil(((float)image.cols) / blockDim.x),ceil(((float)image.rows) / blockDim.y));
    int num_RGBelem,size = image.rows * image.cols;
    // Mat img = image.clone();
    Mat null_mat = Mat::zeros(cv::Size(image.rows, image.cols), CV_8UC3);
    // img = image.clone();
    cout << "Remapping image :\n \t \ttipo matrice :" << "CV_" + type2str(image.type()) <<endl;
    cv::Mat img;
    uchar *d_image, *d_output;
    int *d_tranfArray;
    // vector<uchar> image_array;
    std::chrono::steady_clock::time_point begin, end;
    
    //definisco l'immagine 
    cv::cuda::GpuMat input, output;
    input.upload(image);
    //output.create(cv::Size(image.rows, image.cols), CV_8UC3);
    output = input.clone();
    output.setTo(Scalar::all(0));
    //output.upload(null_mat);
    //cout << "Remapping image :\n \t \ttipo matrice :" << "CV_" + type2str(input.type()) <<endl;

    //cout <<" \n alloco il vettore di transposizione \n";
    cudaStatus = cudaMalloc((void **) &d_tranfArray, sizeof(int) * size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed RemappingMultiChannelImage!");
        goto ErrorNewMultiRemapping;
    }

    /*
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);
    cudaEventRecord (start, 0);
    */
    // os_remapping.open("tempi_cudamemcpy_remapping2.txt", std::ofstream::out | std::ofstream::app);
    // begin = std::chrono::steady_clock::now();

    //cout <<" \n copio il vettore di transposizione \n";
    cudaStatus = cudaMemcpy(d_tranfArray,tranfArray,sizeof(int) * size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaMemSetfailed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorNewMultiRemapping;
    }
    // end = std::chrono::steady_clock::now();
    // os_remapping << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<"\n";
    // os_remapping.close();
    /*
    cudaEventRecord (stop, 0);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&time, start, stop);
    // return value is expressed in milliseconds (with resolution of 0.5 us)
    os_remapping << time << "\n";
    os_remapping.close();
    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    */
    //cout << "\n step / sizeof(uchar3) = " << ( int )input.step / sizeof(uchar3) << endl;

    new_remapping_kernel<<<gridDim,blockDim>>> (input, input.rows, input.cols, input.step, image.channels(), d_tranfArray, output);
    cudaDeviceSynchronize();
    
    output.download(img);

    cudaFree(d_tranfArray);

    return img;

ErrorNewMultiRemapping:
    
    cudaFree(d_tranfArray);
    //return Mat::zeros(cv::Size(image.rows, image.cols), CV_8UC1);
    exit(0);

}




cudaError_t warpPerspectiveRemappingCUDA(Mat input, Mat &output, const Mat H){
    cudaError_t cudaStatus;

    // allocate array of all locations
    int Numrows = input.rows;
    int Numcols = input.cols;
    int channels   = input.channels();
    // cout << "rows " << Numrows << "col " << Numcols << "channels " << channels <<endl;
    int size = Numrows*Numcols;
    // int MaxX,MaxY = -1000;
    // int MinX,MinY =  1000;
    
    // int Idx;
    // int homeX, homeY;
    int *TransArry = (int *)malloc(sizeof(int)*size);
    // float *d_H;
    // float *vecH = (float *)malloc(sizeof(float) * H.rows * H.cols);
    // int *d_T;

    Mat tranImg;
    
    cudaStatus = calculateTransferArray(H,TransArry,Numrows, Numcols);


    //input.copyTo(tranImg);
    input.copyTo(tranImg);
    tranImg = tranImg - tranImg;
    
    //cv::Mat remappingImage(Mat &image, int *tranfArray)
    
    cout <<" \n richiamo la funzione per il remapping \n";
    cout <<" \n NUMERO DI CANALI : " << input.channels() << "\n";
    

    
    output = remappingMultiChannelImage(input, TransArry);

    return cudaStatus;
}





