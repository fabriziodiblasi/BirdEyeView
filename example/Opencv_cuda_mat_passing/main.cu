#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cuda_runtime.h>

using std::cout;
using std::endl;

using namespace cv;
using namespace std;

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if(err!=cudaSuccess)
	{
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)


__global__ void bgr_to_gray_kernel( unsigned char* input, 
									unsigned char* output, 
									int width,
									int height,
									int inputWidthStep,
									int outputWidthStep)
{
	//2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if((xIndex<width) && (yIndex<height))
	{
		//Location of colored pixel in input
		const int color_tid = yIndex * inputWidthStep + (3 * xIndex);
		
		
		const unsigned char blue	= input[color_tid];
		const unsigned char green	= input[color_tid + 1];
		const unsigned char red		= input[color_tid + 2];

		
		output[color_tid] = static_cast<unsigned char>(blue);
		output[color_tid + 1] = static_cast<unsigned char>(green);
		output[color_tid + 2] = static_cast<unsigned char>(red);
	}
}

void convert_to_gray(const cv::Mat& input, cv::Mat& output)
{
	/*
	L'immagine di output è così definita :
		cv::Mat output(input.rows,input.cols,CV_8UC1);
	CV_8UC1 è una matrice di unsigned char ad un solo canale

	quello che devo scoprire è come poter riottenere l'immagine originale

	ho ridefinito l'immagine come CV_8UC3
	*/

	//Calculate total number of bytes of input and output image
	const int inputBytes = input.step * input.rows;
	const int outputBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	//Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input,inputBytes),"CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output,outputBytes),"CUDA Malloc Failed");

	//Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input,input.ptr(),inputBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

	//Specify a reasonable block size
	const dim3 block(16,16);

	//Calculate grid size to cover the whole image
	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);

	//Launch the color conversion kernel
	bgr_to_gray_kernel<<<grid,block>>>(d_input,d_output,input.cols,input.rows,input.step,output.step);

	//Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

	//Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(),d_output,outputBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

	//Free the device memory
	SAFE_CALL(cudaFree(d_input),"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output),"CUDA Free Failed");
}


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


int main()
{
	// CV_8UC3

	std::string imagePath = "image.jpg";

	//Read input image from the disk
	cv::Mat input = cv::imread(imagePath,IMREAD_COLOR);

	if(input.empty())
	{
		std::cout<<"Image Not Found!"<<std::endl;
		std::cin.get();
		return -1;
	}

	string ty = "CV_" + type2str( input.type() );
	cout << "tipo matrice :" << ty.c_str() <<endl;

	//Create output image
	cv::Mat output(input.rows,input.cols,CV_8UC3);

	//Call the wrapper function
	/*
	passo per riferimento i due oggetti Mat al wrapper del kernel cuda
	*/
	convert_to_gray(input,output);

	//Show the input and output
	cv::imshow("Input",input);
	cv::imshow("Output",output);
	
	//Wait for key press
	cv::waitKey();

	return 0;
}