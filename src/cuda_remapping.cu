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



__global__ void remapping_single_ch_image_cuda_kernel(uchar *image, int numRows, int numCols, int *tranfArray, int numChannel, uchar *output){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * numCols + col;
    int homeX, homeY;
    int newhomeX, newhomeY;
    if (idx < numCols * numRows){
    //if (col < numCols && row < numRows){
        homeX=idx % numCols;
        homeY=idx / numCols; 
        if(tranfArray[idx] != -1 ){  
        //if(tranfArray[idx] != -1 && (homeY * numCols + homeX) < (numCols * numRows)){   
            //cout << "Index " << Idx << "Passed " << endl;
            newhomeX = tranfArray[idx] % numCols; // Col ID
            newhomeY = tranfArray[idx] / numCols;  // Row ID

            //i * col + j
            output[newhomeY * numCols + newhomeX] = image [homeY * numCols + homeX];
            
            
        }
    }
    // codice da parallelizzare :

    // Remap Image
    // for (Idx=0; Idx < size; Idx ++ ){

    //     homeX=Idx % Numcols;
    //     homeY=Idx / Numcols;                
    //     //tranImg.at<uchar>(homeY, homeX) =0;
    //     if(TransArry[Idx] != -1){   
    //         //cout << "Index " << Idx << "Passed " << endl;
    //         int newhomeX=TransArry[Idx] % Numcols; // Col ID
    //         int newhomeY=TransArry[Idx] / Numcols;  // Row ID
    //         tranImg.at<uchar>(newhomeY, (newhomeX*channels)) = input.at<uchar>(homeY, homeX*channels);
    //         if(channels>1)
    //             tranImg.at<uchar>(newhomeY, newhomeX*channels+1) = input.at<uchar>(homeY, homeX*channels+1);
    //         if(channels>2)
    //             tranImg.at<uchar>(newhomeY, newhomeX*channels+2) = input.at<uchar>(homeY, homeX*channels+2);
            
    //         }
    // }

}


__global__ void remapping_multi_ch_image_cuda_kernel(uchar *image, int numRows, int numCols, int *tranfArray, int numChannel, uchar *output){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int idx = row * numCols + col;
    int homeX, homeY;
    int newhomeX, newhomeY;
    //if (idx < numCols * numRows){
    if (idx < numCols * numRows){
    //if (col < numCols*numChannel && row < numRows){
        homeX=idx % numCols;
        homeY=idx / numCols; 
        if(tranfArray[idx] != -1 ){   
            //cout << "Index " << Idx << "Passed " << endl;
            newhomeX = tranfArray[idx] % numCols; // Col ID
            newhomeY = tranfArray[idx] / numCols;  // Row ID

            //i * col + j
            output[newhomeY * numCols + (newhomeX * numChannel)] = image [homeY * numCols + (homeX* numChannel)]; // B
            
            if(numChannel > 1)
                output[(newhomeY * numCols) + (newhomeX * numChannel + 1)] = image [(homeY * numCols) + (homeX * numChannel + 1)]; // G
            if(numChannel > 2)
                output[(newhomeY * numCols) + (newhomeX * numChannel + 2)] = image [(homeY * numCols) + (homeX * numChannel + 2)]; // R
            
        }
    }
    

}



/**
    restituisce in output l'immagine rimappata
*/
cv::Mat remappingSingleChannelImage(Mat image, int *tranfArray){
    cudaError_t cudaStatus;
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((float)image.cols) / blockDim.x),ceil(((float)image.rows) / blockDim.y));

    int size = image.rows * image.cols;
    Mat img = image.clone();
    // Mat img = Mat::zeros(cv::Size(image.rows, image.cols), CV_32FC3);
    // img = image.clone();

    
	cout << "Remapping image :\n \t \ttipo matrice :" << "CV_" + type2str(image.type()) <<endl;

    //cout <<"\n (float *)malloc(sizeof(float)*size) ";
    uchar *h_image = (uchar *)malloc(sizeof(uchar)*size);
    
    uchar *d_image, *d_output;

    int *d_tranfArray;
    cout <<" \n alloco il vettore sul device per l'immagine \n";
    cudaStatus = cudaMalloc((void **) &d_image, sizeof(uchar) * size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto ErrorRemapping;
    }
    cout <<" \n alloco il vettore immagine per l'output\n";
    cudaStatus = cudaMalloc((void **) &d_output, sizeof(uchar) * size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto ErrorRemapping;
    }
    cout <<" \n alloco il vettore di transposizione \n";
    cudaStatus = cudaMalloc((void **) &d_tranfArray, sizeof(int) * size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto ErrorRemapping;
    }

    //matToArray(h_image, image, image.rows, image.cols);
    //std::memcpy( h_image,img.data, size*sizeof(uchar));
    cvMatToVector(img, h_image);


    //copio i vettori
    cudaStatus = cudaMemcpy(d_image,h_image,sizeof(uchar) * size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaMemCpy failed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorRemapping;
    }

    cudaStatus = cudaMemcpy(d_tranfArray,tranfArray,sizeof(int) * size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaMemSetfailed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorRemapping;
    }



    //__global__ void remapping_image_cuda_kernel(float *image, int numRows, int numCols, int *tranfArray, int numChannel, float *output){
    cout<<"\n RICHIAMO IL KERNELL PER IL REMAPPING DELL'IMMAGINE \n";
    //   <<<gridDim, blockDim>>>
    //remapping_single_ch_image_cuda_kernel<<<ceil(size/256.0),256>>>(d_image, image.rows, image.cols, d_tranfArray, image.channels(),d_output);
    remapping_single_ch_image_cuda_kernel<<<gridDim, blockDim>>>(d_image, image.rows, image.cols, d_tranfArray, image.channels(),d_output);
    cudaThreadSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorRemapping;
    }

    cout <<" \n copio il risultato del kernel \n";
    cudaStatus = cudaMemcpy(h_image,d_output,sizeof(uchar) * size,cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaMemCpy failed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorRemapping;
    }
    /**
    * converte un vettore in un oggetto Mat
    * src : array
    * dst : Mat
    */
    cout <<" \n copio il risultato del kernel nell'oggetto mat\n";
    //arrayToMat(img, h_image, size);
    //memcpy(img.data())
    std::memcpy(img.data, h_image, size*sizeof(uchar));
    cout <<" \n finita la copia \n";

    return img;



ErrorRemapping:
    //cout<< "****** ERRORE CUDA ****** : " << cudaStatus << endl;
    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_tranfArray);
    return Mat::zeros(cv::Size(image.rows, image.cols), CV_8UC1);
    

}


cv::Mat OLDremappingMultiChannelImage(Mat image, int *tranfArray){
    cudaError_t cudaStatus;
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((float)image.cols) / blockDim.x),ceil(((float)image.rows) / blockDim.y));
    int num_RGBelem,size = image.rows * image.cols;
    Mat img = image.clone();
    // Mat img = Mat::zeros(cv::Size(image.rows, image.cols), CV_32FC3);
    // img = image.clone();
	cout << "Remapping image :\n \t \ttipo matrice :" << "CV_" + type2str(image.type()) <<endl;
     
    uchar *d_image, *d_output;
    int *d_tranfArray;
    vector<uchar> image_array;

    image_array = multiChannelMatToVector(img);
    //conto il numero di elementi totali
    num_RGBelem = image_array.size();

    uchar *h_image = (uchar *)malloc(sizeof(uchar)*num_RGBelem);
    
    cout << "num_RGBelem : "<< num_RGBelem << endl;

    

    // alloco la memoria sulla GPU
    //cout <<" \n alloco il vettore sul device per l'immagine \n";
    cudaStatus = cudaMalloc((void **) &d_image, sizeof(uchar) * num_RGBelem);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto ErrorMultiRemapping;
    }
    //cout <<" \n alloco il vettore immagine per l'output\n";
    cudaStatus = cudaMalloc((void **) &d_output, sizeof(uchar) * num_RGBelem);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto ErrorMultiRemapping;
    }
    //cout <<" \n alloco il vettore di transposizione \n";
    cudaStatus = cudaMalloc((void **) &d_tranfArray, sizeof(int) * size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto ErrorMultiRemapping;
    }

    //traduco lo std::vector in un array normale
    stdVectorToArray(image_array, h_image);

    //copio i dati sulla GPU

    //copio i vettori
    cudaStatus = cudaMemcpy(d_image,h_image,sizeof(uchar) * num_RGBelem, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaMemCpy failed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorMultiRemapping;
    }

    cudaStatus = cudaMemcpy(d_tranfArray,tranfArray,sizeof(int) * size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaMemSetfailed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorMultiRemapping;
    }

    // richiamo il cuda kernel
    remapping_multi_ch_image_cuda_kernel<<<gridDim,blockDim>>>(d_image, image.rows, image.cols , d_tranfArray, image.channels(),d_output);
    //remapping_multi_ch_image_cuda_kernel<<<ceil((float)(num_RGBelem/256.0)),256>>>(d_image, image.rows, image.cols , d_tranfArray, image.channels(),d_output);
    //remapping_multi_ch_image_cuda_kernel<<<ceil(num_RGBelem/256.0),256>>>(d_image, image.rows, image.cols * image.channels(), d_tranfArray, image.channels(),d_output);
    cudaThreadSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorMultiRemapping;
    }

    cout <<" \n copio il risultato del kernel \n";
    cudaStatus = cudaMemcpy(h_image,d_output,sizeof(uchar) * num_RGBelem,cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaMemCpy failed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorMultiRemapping;
    }
    /**
    * converte un vettore in un oggetto Mat
    * src : array
    * dst : Mat
    */
    cout <<" \n copio il risultato del kernel nell'oggetto mat\n";
    //arrayToMat(img, h_image, size);
    //memcpy(img.data())
    memcpy(img.data, h_image, num_RGBelem*sizeof(uchar));
    cout <<" \n finita la copia \n";

    return img;



ErrorMultiRemapping:
    //cout<< "****** ERRORE CUDA ****** : " << cudaStatus << endl;
    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_tranfArray);
    //return Mat::zeros(cv::Size(image.rows, image.cols), CV_8UC1);
    exit(0);

}
// __global__ void new_remapping_kernel(uchar* src, int numRows, int numCols, size_t step, int numChannel, int *tranfArray, uchar* out){

__global__ void new_remapping_kernel(cv::cuda::PtrStepSz<uchar3> src, int numRows, int numCols, size_t step, int numChannel, int *tranfArray, cv::cuda::PtrStepSz<uchar3> out){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int idx = row * numCols + col;
    int homeX, homeY;
    int newhomeX, newhomeY;
    int iStep = step / sizeof(uchar3);
    int oStep = step / sizeof(uchar3);
    uchar3 pxval;


    if ((row < numRows) && (col < numCols))
    {
        // azzero il pixell attuale
        // uchar * px = out + (row * step);
        // px[col] = 0; px[col+1] = 0; px[col+1] = 0;

        homeX=idx % numCols;
        homeY=idx / numCols; 
        if(tranfArray[idx] != -1 ){ 
            newhomeX = tranfArray[idx] % numCols; // Col ID
            newhomeY = tranfArray[idx] / numCols;  // Row ID
            //uchar *outrowptr = out + newhomeY * step;
            //uchar *srcrowptr = src + homeY * step;
            
            // outrowptr[newhomeX] = srcrowptr[homeX];
            // out(newhomeY, newhomeX*numChannel) = src(homeY, homeX*numChannel);
            // out(newhomeY, newhomeX) = src(homeY, homeX);
            pxval = src(homeY, homeX );
            out(newhomeY, newhomeX) = pxval;
            
            if (numChannel > 1){
                pxval = src(homeY, homeX  + 1);
                out(newhomeY, newhomeX  + 1) = pxval;
            }
            
            if (numChannel > 2){
                // outrowptr[newhomeX * numChannel + 2] = srcrowptr[homeX * numChannel + 2];
                // outrowptr[newhomeX + 2] = srcrowptr[homeX + 2];
                // out(newhomeY, newhomeX*numChannel + 2) = src(homeY, homeX*numChannel+2);
                // out(newhomeY, newhomeX + 2) = src(homeY, homeX + 2);
                pxval = src(homeY, homeX  +2);
                out(newhomeY, newhomeX  + 2) = pxval;
            }
        }
    

    }
}

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

    //definisco l'immagine 
    cv::cuda::GpuMat input, output;
    input.upload(image);
    //output.create(cv::Size(image.rows, image.cols), CV_8UC3);
    //output = input.clone();
    output.upload(null_mat);
    //cout << "Remapping image :\n \t \ttipo matrice :" << "CV_" + type2str(input.type()) <<endl;

    //cout <<" \n alloco il vettore di transposizione \n";
    cudaStatus = cudaMalloc((void **) &d_tranfArray, sizeof(int) * size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto ErrorNewMultiRemapping;
    }

    //cout <<" \n copio il vettore di transposizione \n";
    cudaStatus = cudaMemcpy(d_tranfArray,tranfArray,sizeof(int) * size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaMemSetfailed: %s\n", cudaGetErrorString(cudaStatus));
        goto ErrorNewMultiRemapping;
    }
    
    cout << "\n step / sizeof(uchar3) = " << ( int )input.step / sizeof(uchar3) << endl;

    new_remapping_kernel<<<gridDim,blockDim>>> (input, input.rows, input.cols, input.step, image.channels(), d_tranfArray, output);
    cudaThreadSynchronize();
    
    output.download(img);

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
    

    //single channel img
    /*
    vector<Mat> splitImg = imageSplitting(input);
    // getchar();
    // Mat merged;
    // imshow("red", splitImg[2]);
    // imshow("blue", splitImg[0]);
    // imshow("green",splitImg[1]);
    // merge(splitImg,merged);
    // imshow("green",merged);
    vector<Mat> result;
    
    // Mat output_blue = remappingImage(splitImg[0], TransArry);
    // Mat output_green = remappingImage(splitImg[1], TransArry);
    // Mat output_red = remappingImage(splitImg[2], TransArry);
    result.push_back(remappingSingleChannelImage(splitImg[0], TransArry));
    result.push_back(remappingSingleChannelImage(splitImg[1], TransArry));
    result.push_back(remappingSingleChannelImage(splitImg[2], TransArry));

    merge(result,output);
    */
    
    // output = OLDremappingMultiChannelImage(input, TransArry);
    output = remappingMultiChannelImage(input, TransArry);

    return cudaStatus;
}





