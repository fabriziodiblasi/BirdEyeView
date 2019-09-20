#include "../lib/utilities.h"

// namespaces
using namespace std;
using namespace cv;
#define PI 3.1415926


// int frameWidth = 640;
// int frameHeight = 480;
#define FRAMEWIDTH  640
#define FRAMEHEIGHT 480

bool CUDA = false;


// ---- GLOBAL VAR ----
int alpha_ = 90, beta_ = 90, gamma_ = 90;
int f_ = 500, dist_ = 500;



void birdsEyeView(const Mat &input, Mat &output){
	double focalLength, dist, alpha, beta, gamma; 

	alpha =((double)alpha_ -90) * PI/180;
	beta =((double)beta_ -90) * PI/180;
	gamma =((double)gamma_ -90) * PI/180;
	focalLength = (double)f_;
	dist = (double)dist_;

	Size input_size = input.size();
	double w = (double)input_size.width, h = (double)input_size.height;


	// Projecion matrix 2D -> 3D
	
	Mat A1 = (Mat_<float>(4, 3)<< 
		1, 0, -w/2,
		0, 1, -h/2,
		0, 0, 0,
		0, 0, 1 );
	
	
	// Rotation matrices Rx, Ry, Rz

	Mat RX = (Mat_<float>(4, 4) << 
		1, 0, 0, 0,
		0, cos(alpha), -sin(alpha), 0,
		0, sin(alpha), cos(alpha), 0,
		0, 0, 0, 1 );

	Mat RY = (Mat_<float>(4, 4) << 
		cos(beta), 0, -sin(beta), 0,
		0, 1, 0, 0,
		sin(beta), 0, cos(beta), 0,
		0, 0, 0, 1	);

	Mat RZ = (Mat_<float>(4, 4) << 
		cos(gamma), -sin(gamma), 0, 0,
		sin(gamma), cos(gamma), 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1	);

	// R - rotation matrix
	Mat R = RX * RY * RZ;
	// cout<< " R : \n "<< R << endl;



	// T - translation matrix
	Mat T = (Mat_<float>(4, 4) << 
		1, 0, 0, 0,  
		0, 1, 0, 0,  
		0, 0, 1, dist,  
		0, 0, 0, 1); 
	
	// K - intrinsic matrix 
	Mat K = (Mat_<float>(3, 4) << 
		focalLength, 0, w/2, 0,
		0, focalLength, h/2, 0,
		0, 0, 1, 0
		); 

	// cout << "\n R * A1 :\n"<< R * A1 <<endl;
	
	// cout << "\n T * (R * A1) : \n"<< T * (R * A1) <<endl;	

	// cout << "\n K * (T * (R * A1)) : \n"<< K * (T * (R * A1)) <<endl;	

	Mat transformationMat = K * (T * (R * A1));

	//cout<< " transformationMat : \n "<< transformationMat << endl;

	//cout<< "transformationMat.rows : " << transformationMat.rows << "\ttransformationMat.cols : " << transformationMat.cols << endl;
	//cout << "tipo matrice di transformazione : "<< "CV_" + type2str( transformationMat.type()) << endl;

	warpPerspective(input, output, transformationMat, input_size, INTER_CUBIC | WARP_INVERSE_MAP);

	return;



}





void CUDA_birdsEyeView(const Mat &input, Mat &output){

	cudaError_t error;

	double focalLength, dist, alpha, beta, gamma; 

	alpha =((double)alpha_ -90) * PI/180;
	beta =((double)beta_ -90) * PI/180;
	gamma =((double)gamma_ -90) * PI/180;
	focalLength = (double)f_;
	dist = (double)dist_;

	Size input_size = input.size();
	double w = (double)input_size.width, h = (double)input_size.height;
	

	float A1[12] = {
		1, 0, -w/2,
		0, 1, -h/2,
		0, 0, 0,
		0, 0, 1 
	};



	float RX[16] = {
		1, 0, 0, 0,
		0, cos(alpha), -sin(alpha), 0,
		0, sin(alpha), cos(alpha), 0,
		0, 0, 0, 1 
	};

	float RY[16] ={
		cos(beta), 0, -sin(beta), 0,
		0, 1, 0, 0,
		sin(beta), 0, cos(beta), 0,
		0, 0, 0, 1
	};

	float RZ[16] = {
		cos(gamma), -sin(gamma), 0, 0,
		sin(gamma), cos(gamma), 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	};

	// cout << "stampo RX \n";
	// stampaMatrice(RX , 4, 4);
	// R - rotation matrix
	// Mat R = RX * RY * RZ;

	float R[16], XY[16];
	error = matrixMultiplication(RX, RY, XY, 4, 4, 4, 4);
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(0);
	}
	error = matrixMultiplication(XY, RZ, R, 4, 4, 4, 4);
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(0);
	}
	
	// cout << "stampo R \n";
	// stampaMatrice(R, 4, 4);
	
	// T - translation matrix
	float T[16] = { 
		1, 0, 0, 0,  
		0, 1, 0, 0,  
		0, 0, 1, dist,  
		0, 0, 0, 1
	}; 
	// K - intrinsic matrix 
	float K[12] = { 
		focalLength, 0, w/2, 0,
		0, focalLength, h/2, 0,
		0, 0, 1, 0
	};

	//Mat transformationMat = K * (T * (R * A1));
	float R_A1[12], T_RA1[12], transformationvector[9];

	error = matrixMultiplication(R, A1, R_A1, 4, 4, 4, 3);
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(0);
	}
	// cout << "R * A1 \n";
	// stampaMatrice(R_A1, 4, 3);

	error = matrixMultiplication(T, R_A1, T_RA1, 4, 4, 4, 3);
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(0);
	}
	// cout << "T* (R * A1) \n";
	// stampaMatrice(T_RA1, 4, 3);

	error = matrixMultiplication(K, T_RA1, transformationvector, 4, 4, 4, 3);
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(0);
	}

	cv::Mat tranf_mat(3,3,CV_32FC1);

	// cout << "stampo T \n";
	// stampaMatrice(transformationvector, 3, 3);


	arrayToMat(tranf_mat,transformationvector,9);
	
	// cout << "richiamo la funzione warpPerspectiveCUDA \n";
	
	
	//warpPerspectiveCUDA(input, output, tranf_mat);
	warpPerspectiveRemappingCUDA(input, output, tranf_mat);
	
	return;

}




int main(int argc, char const *argv[]) {

	
	if(argc > 3 || argc == 1) {
	  cerr << "Usage: " << argv[0] << " < CUDA : y / n > <' /path/to/video/ ' | nothing > " << endl;
	  cout << "Exiting...." << endl;
	  return -1;
	}
	int flag=0;
	Mat image,output;
	
	std::ofstream os_cuda;
	// os_cuda.open("misurazioniCUDA.txt", std::ofstream::out | std::ofstream::app);
	std::ofstream os_opencv;
	// os_opencv.open("misurazioniOPENCV.txt", std::ofstream::out | std::ofstream::app);

	VideoCapture capture;
	string cudaflag = argv[1];
	if (cudaflag == "y"){
		CUDA = true;
		cout<<"** CUDA ON ** \n";
	}else{
		CUDA = false;
		cout<<"** CUDA OFF ** \n";
		
	}

	if (argc == 2){
		capture.open(0);
	}
	if (argc == 3){
		string filename = argv[2];
		capture.open(filename);
	}

	if(!capture.isOpened()) throw "Error reading video";

	

	/*
		definisco i parametri e le trackbar
	*/

	namedWindow("Result", 1);

	createTrackbar("Alpha", "Result", &alpha_, 180);
	createTrackbar("Beta", "Result", &beta_, 180);
	createTrackbar("Gamma", "Result", &gamma_, 180);
	createTrackbar("f", "Result", &f_, 2000);
	createTrackbar("Distance", "Result", &dist_, 2000);




	cout << "Capture is opened" << endl;
	for(;;)
	{
		capture >> image;
		//stampo il tipo di immagine
		if(flag == 0){
			string ty = "CV_" + type2str( image.type() );
			cout << "tipo matrice :" << ty.c_str() <<endl;
			flag = 1;
		}
		resize(image, image,Size(FRAMEWIDTH, FRAMEHEIGHT));

		if (CUDA){
			// os_cuda.open("misurazioniCUDA.txt", std::ofstream::out | std::ofstream::app);
			// std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
			CUDA_birdsEyeView(image, output);
			// std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			// std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " [µs]" << std::endl;
			// os_cuda << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<"\n";
			// os_cuda.close();

		}else{
			// os_opencv.open("misurazioniOPENCV.txt", std::ofstream::out | std::ofstream::app);
			// std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
			birdsEyeView(image, output);
			// std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			// std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " [µs]" << std::endl;
			// os_opencv << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<"\n";
			// os_opencv.close();

		}
		//per la visualizzazione 
		if(output.empty())
			break;
		//drawText(image);
		imshow("Result", output);
		if(waitKey(10) >= 0)
			break;
	}
	
	
	return 0;
}