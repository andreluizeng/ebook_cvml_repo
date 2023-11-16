#include <iostream>
#include <string>
#include <fstream>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

// ANN class for plates and ocr
cv::dnn::Net net_plates;
cv::dnn::Net net_ocr;

// variables for plates and ocr classification
float conf_plates;
float conf_ocr;
bool swap_rb; // swap rb channels as the input from opencv is bgr
float scale;
cv::Scalar mean;

// Model sizes for plates and ocr (same used in plates.cfg and ocr.cfg)
int w_plates = 768;
int h_plates = 448;
int w_ocr = 512;
int h_ocr = 256;

// plate and ocr class information
std::vector <std::string> classes_plates;
std::vector <cv::Mat> outs_plates;
std::vector <int> classIds_plates;

std::vector <std::string> classes_ocr;
std::vector <cv::Mat> outs_ocr;
std::vector <int> classIds_ocr;

// data from model, names and config files
std::string plates_model;
std::string plates_cfg;
std::string plates_names;

std::string ocr_model;
std::string ocr_cfg;
std::string ocr_names;

// scope of our external functions

// plates detector functions
void initPlatesDetector(void);
std::vector<cv::Rect> postProcessingPlates(cv::Mat& frame_in, const std::vector<cv::Mat>& outs, std::vector<int>& class_ids);
void drawPredPlates(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame_in);

// ocr functions
void initOCR(void);
std::vector<cv::Rect> postProcessingOCR(cv::Mat& frame_in, const std::vector<cv::Mat>& outs, std::vector<int>& class_ids);
void drawPredOCR(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame_in);
std::vector <int> sortClassesOCR(std::vector <cv::Rect> classes, std::vector <int> classes_id);
std::string convertClassesId(std::vector <int> classes_id);
std::string fixCharsNumbersOCR(std::string ocr_result, int class_id);


// general functions
std::vector <std::string> getResultsNN(const cv::dnn::Net& net);
void readLicensePlate(cv::Mat frame_in);

// main function
int main(int argc, char** argv)
{
	// input image
	cv::Mat frame;
	
	std::string msg;

	// make sure the vectores are empty
	classes_plates.clear();
	classIds_plates.clear();
	outs_plates.clear();

	classes_ocr.clear();
	classIds_ocr.clear();
	outs_ocr.clear();

		// checks if the image file name is in the command line
	if (argc < 2)
	{
		std::cout << "\nUsage: <application> <image_file>\n";
		return 0;
	}

	// opens the image
	frame = cv::imread(argv[1]);

	// check if image is ok
	if (frame.empty())
	{
		std::cout << "\nCan't open input image !\n";
		return 0;
	}

	// initialize the plates and ocr (YOLO/Darknet setup)
	initPlatesDetector();
	initOCR();

	// calls our read plate function
	readLicensePlate(frame);

	// wait a key to be pressed
	cv::waitKey(0);
	
	// free all allocated memory
	frame.release();

	// exit application
	return 0;
}

//******************************************************
// initPlatesDetector function
// initialize the darknet for our custom plates model
//******************************************************
void initPlatesDetector(void)
{
	// yolo/darknet files 
	plates_model = "darknet/model/plates.weights";
	plates_cfg = "darknet/cfg/plates.cfg";
	plates_names = "darknet/names/plates.names";

	// confiability 1 = MAX (100%)
	conf_plates = (float) 0.7;

	// normalization 
	scale = (float)(1.0 / 255.0);
	mean = 0;
	
	// swap r b channels
	swap_rb = true;

	// read classes names from plates.names
	std::ifstream ifs_plates(plates_names.c_str());
	std::string lines_plates;

	while (std::getline(ifs_plates, lines_plates))
		classes_plates.push_back(lines_plates);

	// Yolo/Darknet information (weights and configuration)
	net_plates = cv::dnn::readNet(plates_model, plates_cfg, "Darknet");
	
	// neural network
	//net_plates = new cv::dnn::Net(tmp_net);
	net_plates.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
	
	// DNN_TARGET_OPENCV if OpenCL is supported 
	// DNN_TARGET_CUDA if CUDA is supported 
	net_plates.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);  
	
	return;
}

//******************************************************
// drawPredPlates function
// draw the bounding boxes from detected plates
//******************************************************
void drawPredPlates(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame_in)
{
	cv::Scalar color;

	// assign a label and a color to every class 
	if (!classes_plates.empty())
	{
		CV_Assert(classId < (int)classes_plates.size());

		// mercosul
		if (classId == 0)
			color = cv::Scalar(255, 0, 0);

		// old
		if (classId == 1)
			color = cv::Scalar(255, 255, 255);

		// special
		if (classId == 2)
			color = cv::Scalar(0, 0, 255);

		// fake
		if (classId == 3)
			color = cv::Scalar(255, 255, 255);

		//Draw a rectangle displaying the bounding box
		cv::rectangle(frame_in, cv::Point(left, top), cv::Point(right, bottom), color, 3);
	}
	return;
}

//******************************************************
// postProcessingPlates function
// post processing operations: apply non-max supression 
// for overlaped bouding boxes
//******************************************************
std::vector<cv::Rect> postProcessingPlates(cv::Mat& frame_in, const std::vector<cv::Mat>& outs, std::vector<int>& class_ids)
{
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<cv::Rect> objects_boxes;

	// Non-maximum suppression threshold, lets keep low to detect 
	// almost all squared plates like pattern
	float nmsThreshold = (float)0.4;  

	if (outs.empty())
	{
		objects_boxes.clear();
		return objects_boxes;
	}

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. 
		// Assign the box's class label as the class with the highest score for the box.
		float* data = (float*)outs[i].data;

		for (int j = 0; j < outs[i].rows; j++, data += outs[i].cols)
		{
			cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			cv::Point classIdPoint;

			double confidence;

			// Get the value and location of the maximum score
			cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > conf_plates)
			{
				int centerX = (int)(data[0] * frame_in.cols);
				int centerY = (int)(data[1] * frame_in.rows);
				int width = (int)(data[2] * frame_in.cols);
				int height = (int)(data[3] * frame_in.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant 
	// overlapping boxes with lower confidences
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, conf_plates, nmsThreshold, indices);

	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];

		// increase the bounding box size to make sure it doesn't crop any letter/number in it
		box.width += 5;
		box.height += 5;

		if (box.width + box.x > frame_in.cols)
			box.width -= 5;

		if (box.height + box.y > frame_in.rows)
			box.height -= 5;

		// draw the predicted bounding boxes
		drawPredPlates(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame_in);

		// if a fake plate, ignore and remove from the list
		if (classIds[idx] != 3)
		{
			objects_boxes.push_back(box);
			class_ids.push_back(classIds[idx]);
		}
	}

	return objects_boxes;
}


//******************************************************
// initOCR function
// initialize the darknet for our custom plates model
//******************************************************
void initOCR(void)
{
	// yolo/darknet files 
	ocr_model = "darknet/model/ocr.weights";
	ocr_cfg = "darknet/cfg/ocr.cfg";
	ocr_names = "darknet/names/ocr.names";

	// normalization 
	scale = (float)(1.0 / 255.0);
	mean = 0;
	
	// swap r b channels
	swap_rb = true;

	// read classes names from plates.nmaes
	std::ifstream ifs_ocr(ocr_names.c_str());
	std::string lines_ocr;

	while (std::getline(ifs_ocr, lines_ocr))
		classes_ocr.push_back(lines_ocr);

	// confiability 1 = MAX (100%)
	conf_ocr = (float) 0.4;

	// Yolo/Darknet information (weights and configuration)
	net_ocr = cv::dnn::readNet(ocr_model, ocr_cfg, "Darknet");

	// neural network
	net_ocr.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);

	// DNN_TARGET_OPENCV if OpenCL is supported 
	// DNN_TARGET_CUDA if CUDA is supported 
	net_ocr.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	
	return;
}

//**************************************************************
// drawPredOCR function
// draw the bounding boxes from detected letters and numbers
//**************************************************************
void drawPredOCR(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame_in)
{
	//Get the label for the class name and its confidence
	std::string label;

	cv::Scalar color(255, 255, 255);
	cv::Scalar textcolor(0, 0, 0);

	// assign a label and a color to every class (letters = red, numbers = blue)
	if (!classes_ocr.empty())
	{
		CV_Assert(classId < (int)classes_ocr.size());

		if (classId == 0)
		{
			label = "A";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 1)
		{
			label = "B";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 2)
		{
			label = "C";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 3)
		{
			label = "D";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 4)
		{
			label = "E";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 5)
		{
			label = "F";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 6)
		{
			label = "G";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 7)
		{
			label = "H";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 8)
		{
			label = "I";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 9)
		{
			label = "J";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 10)
		{
			label = "K";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 11)
		{
			label = "L";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 12)
		{
			label = "M";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 13)
		{
			label = "N";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 14)
		{
			label = "O";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 15)
		{
			label = "P";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 16)
		{
			label = "Q";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 17)
		{
			label = "R";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 18)
		{
			label = "S";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 19)
		{
			label = "T";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 20)
		{
			label = "U";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 21)
		{
			label = "V";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 22)
		{
			label = "X";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 23)
		{
			label = "W";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 24)
		{
			label = "Y";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 25)
		{
			label = "Z";
			color = cv::Scalar(0, 0, 255);
		}

		if (classId == 26)
		{
			label = "0";
			color = cv::Scalar(255, 0, 0);
		}

		if (classId == 27)
		{
			label = "1";
			color = cv::Scalar(255, 0, 0);
		}

		if (classId == 28)
		{
			label = "2";
			color = cv::Scalar(255, 0, 0);
		}

		if (classId == 29)
		{
			label = "3";
			color = cv::Scalar(255, 0, 0);
		}

		if (classId == 30)
		{
			label = "4";
			color = cv::Scalar(255, 0, 0);
		}

		if (classId == 31)
		{
			label = "5";
			color = cv::Scalar(255, 0, 0);
		}

		if (classId == 32)
		{
			label = "6";
			color = cv::Scalar(255, 0, 0);
		}

		if (classId == 33)
		{
			label = "7";
			color = cv::Scalar(255, 0, 0);
		}

		if (classId == 34)
		{
			label = "8";
			color = cv::Scalar(255, 0, 0);
		}

		if (classId == 35)
		{
			label = "9";
			color = cv::Scalar(255, 0, 0);
		}

		// perfumary for display the in-picture text
		int baseLine;
		cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1, 0.7, &baseLine);
		top = cv::max(top, labelSize.height);

		cv::rectangle(frame_in, cv::Point(left, top-30), cv::Point(left + 25, top - 10 + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
		cv::putText(frame_in, label, cv::Point(left+5, top-10), cv::FONT_HERSHEY_SIMPLEX, 0.7, textcolor, 2);

		cv::rectangle(frame_in, cv::Point(left, top), cv::Point(right, bottom), color, 1);
	}

	return;
}

//******************************************************
// postProcessingOCR function
// post processing operations: apply non-max supression 
// for overlaped bouding boxes
//******************************************************
std::vector<cv::Rect> postProcessingOCR(cv::Mat& frame_in, const std::vector<cv::Mat>& outs, std::vector<int>& class_ids)
{
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<cv::Rect> objects_boxes;

	// Non-maximum suppression threshold
	float nmsThreshold = (float)0.4;  

	if (outs.empty())
	{
		objects_boxes.clear();
		return objects_boxes;
	}
	for (size_t i = 0; i < outs.size(); i++)
	{
		// Scan through all the bounding boxes output from the network_letter and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; j++, data += outs[i].cols)
		{
			cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			cv::Point classIdPoint;
			double confidence;

			// Get the value and location of the maximum score
			cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > conf_ocr)
			{
				int centerX = (int)(data[0] * frame_in.cols);
				int centerY = (int)(data[1] * frame_in.rows);
				int width = (int)(data[2] * frame_in.cols);
				int height = (int)(data[3] * frame_in.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
	}
	
	// Perform non maximum suppression to eliminate redundant 
	// overlapping boxes with lower confidences
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, conf_ocr, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); i++)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];

		// only add the list if it is big enough to be a char or number
		// it can be a noise caused by the flag drawing in the mercosul plate
		// size is based in our minimum detection height and width (25x65 pixels)
		if ((box.width > 20) && (box.height > 60))
		{
			objects_boxes.push_back(box);
			drawPredOCR(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame_in);
			class_ids.push_back(classIds[idx]);

		}
	}

	return objects_boxes;
}

//******************************************************
// ConvertClassesId function
// convert the class id to your real value
//******************************************************
std::string convertClassesId(std::vector <int> classes_id)
{
	std::string aux;
	char a;

	for (int i = 0; i < classes_id.size(); i++)
	{
		if (classes_id[i] == 0)
			a = 'A';

		if (classes_id[i] == 1)
			a = 'B';

		if (classes_id[i] == 2)
			a = 'C';

		if (classes_id[i] == 3)
			a = 'D';

		if (classes_id[i] == 4)
			a = 'E';

		if (classes_id[i] == 5)
			a = 'F';

		if (classes_id[i] == 6)
			a = 'G';

		if (classes_id[i] == 7)
			a = 'H';

		if (classes_id[i] == 8)
			a = 'I';

		if (classes_id[i] == 9)
			a = 'J';

		if (classes_id[i] == 10)
			a = 'K';

		if (classes_id[i] == 11)
			a = 'L';

		if (classes_id[i] == 12)
			a = 'M';

		if (classes_id[i] == 13)
			a = 'N';

		if (classes_id[i] == 14)
			a = 'O';

		if (classes_id[i] == 15)
			a = 'P';

		if (classes_id[i] == 16)
			a = 'Q';

		if (classes_id[i] == 17)
			a = 'R';

		if (classes_id[i] == 18)
			a = 'S';

		if (classes_id[i] == 19)
			a = 'T';

		if (classes_id[i] == 20)
			a = 'U';

		if (classes_id[i] == 21)
			a = 'V';

		if (classes_id[i] == 22)
			a = 'X';

		if (classes_id[i] == 23)
			a = 'W';

		if (classes_id[i] == 24)
			a = 'Y';

		if (classes_id[i] == 25)
			a = 'Z';

		if (classes_id[i] == 26)
			a = '0';

		if (classes_id[i] == 27)
			a = '1';

		if (classes_id[i] == 28)
			a = '2';

		if (classes_id[i] == 29)
			a = '3';

		if (classes_id[i] == 30)
			a = '4';

		if (classes_id[i] == 31)
			a = '5';

		if (classes_id[i] == 32)
			a = '6';

		if (classes_id[i] == 33)
			a = '7';

		if (classes_id[i] == 34)
			a = '8';

		if (classes_id[i] == 35)
			a = '9';

		aux.push_back(a);
	}

	return aux;
}

//******************************************************
// fixCharsNumbersOCR function
// fix wrong detections when numbers should be letters
// and vice-versa
//******************************************************
std::string fixCharsNumbersOCR(std::string ocr_result, int class_id)
{
	char text[100];

	if (ocr_result.length() == 7)
	{
		sprintf(text, "%s", ocr_result.c_str());

		// fix for mercosul model
		// pattern:  CCCNCNN   (C = char, N = number)
		if (class_id == 0)
		{
			// the cases below is based on how much the number
			// "looks like" a letters
			// first 3 positions are letters
			for (int i = 0; i < 3; i++)
			{
				switch (text[i])
				{
				case '0': text[i] = 'O';
					break;

				case '1': text[i] = 'I';
					break;

				case '2': text[i] = 'Z';
					break;

				case '3': text[i] = 'E';
					break;

				case '4': text[i] = 'A';
					break;

				case '5': text[i] = 'S';
					break;

				case '6': text[i] = 'G';
					break;

				case '7': text[i] = 'Z';
					break;

				case '8': text[i] = 'B';
					break;

				case '9': text[i] = 'S';
					break;

				default: break;

				}
			}
			
			// 4th position should be a number, if a letter is detected
			// we need to convert it back to a number
			switch (text[3])
			{
			case 'A': text[3] = '4';
				break;

			case 'B': text[3] = '8';
				break;

			case 'D': text[3] = '0';
				break;

			case 'E': text[3] = '3';
				break;

			case 'F': text[3] = '7';
				break;

			case 'G': text[3] = '6';
				break;

			case 'I': text[3] = '1';
				break;

			case 'J': text[3] = '1';
				break;

			case 'L': text[3] = '1';
				break;

			case 'O': text[3] = '0';
				break;

			case 'Q': text[3] = '0';
				break;

			case 'S': text[3] = '5';
				break;

			case 'T': text[3] = '1';
				break;

			case 'Z': text[3] = '2';
				break;

			default: break;
			}

			// 5th position has to be a letter again
			switch (text[4])
			{
			case '0': text[4] = 'O';
				break;

			case '1': text[4] = 'I';
				break;

			case '2': text[4] = 'Z';
				break;

			case '3': text[4] = 'E';
				break;

			case '4': text[4] = 'A';
				break;

			case '5': text[4] = 'S';
				break;

			case '6': text[4] = 'G';
				break;

			case '7': text[4] = 'Z';
				break;

			case '8': text[4] = 'B';
				break;

			case '9': text[4] = 'S';
				break;

			default: break;
			}

			// and the last 2 positions should be a number
			for (int i = 5; i < 7; i++)
			{
				switch (text[i])
				{
				case 'A': text[i] = '4';
					break;

				case 'B': text[i] = '8';
					break;

				case 'D': text[i] = '0';
					break;

				case 'E': text[i] = '3';
					break;

				case 'F': text[i] = '7';
					break;

				case 'G': text[i] = '6';
					break;

				case 'I': text[i] = '1';
					break;

				case 'J': text[i] = '1';
					break;

				case 'L': text[i] = '1';
					break;

				case 'O': text[i] = '0';
					break;

				case 'Q': text[i] = '0';
					break;

				case 'S': text[i] = '5';
					break;

				case 'T': text[i] = '1';
					break;

				case 'Z': text[i] = '2';
					break;

				default: break;
				}
			}

			ocr_result = text;
		}

		// old or special model
		// pattern:  CCCNNNN
		else
		{
			// first 3 position are numbers
			for (int i = 0; i < 3; i++)
			{
				switch (text[i])
				{
				case '0': text[i] = 'O';
					break;

				case '1': text[i] = 'I';
					break;

				case '2': text[i] = 'Z';
					break;

				case '3': text[i] = 'A';
					break;

				case '4': text[i] = 'A';
					break;

				case '5': text[i] = 'S';
					break;

				case '6': text[i] = 'G';
					break;

				case '7': text[i] = 'Z';
					break;

				case '8': text[i] = 'B';
					break;

				case '9': text[i] = 'S';
					break;

				default: break;

				}
			}

			// last four positions are numbers
			for (int i = 3; i < 7; i++)
			{
				switch (text[i])
				{

				case 'A': text[i] = '4';
					break;

				case 'B': text[i] = '8';
					break;

				case 'D': text[i] = '0';
					break;

				case 'E': text[i] = '3';
					break;

				case 'F': text[i] = '7';
					break;

				case 'G': text[i] = '6';
					break;

				case 'I': text[i] = '1';
					break;

				case 'J': text[i] = '1';
					break;

				case 'L': text[i] = '1';
					break;

				case 'O': text[i] = '0';
					break;

				case 'Q': text[i] = '0';
					break;

				case 'S': text[i] = '5';
					break;

				case 'T': text[i] = '1';
					break;

				case 'Z': text[i] = '2';
					break;

				default: break;
				}
			}

			ocr_result = text;
		}
	}

	return ocr_result;
}

//******************************************************
// sortClassesOCR function
// sort detected classes based on their position
// from left to right
//******************************************************
std::vector <int> sortClassesOCR(std::vector <cv::Rect> classes, std::vector <int> classes_id)
{
	int sorted = 0;

	// return if no class is empty
	if (classes_id.size() < 1) return classes_id;

	cv::Rect aux = classes[0];
	int aux_id = classes_id[0];

	while (!sorted)
	{
		sorted = 1;

		// sort the classes by their horizontal position
		for (int i = 0; i < classes.size() - 1; i++)
		{
			if (classes[i].x > classes[i + 1].x)
			{
				aux = classes[i + 1];
				aux_id = classes_id[i + 1];

				classes[i + 1] = classes[i];
				classes_id[i + 1] = classes_id[i];

				classes[i] = aux;
				classes_id[i] = aux_id;

				sorted = 0;
				i++;
			}
		}
	}

	return classes_id;
}

//******************************************************
// getResultsNN function
// return all detected classes from the Neural Network
//******************************************************
std::vector <std::string> getResultsNN(const cv::dnn::Net& net)
{
	//static std::vector<cv::String> names; // remover
	std::vector<cv::String> names;

	//Get the indices of the output layers, i.e. the layers with unconnected outputs
	std::vector <int> outLayers = net.getUnconnectedOutLayers();

	//get the names of all the layers in the network
	std::vector<cv::String> layersNames = net.getLayerNames();

	// Get the names of the output layers in names
	names.resize(outLayers.size());

	for (size_t i = 0; i < outLayers.size(); i++)
		names[i] = layersNames[outLayers[i] - 1];

	return names;
}

//******************************************************
// readLicensePlate function
// performs the entire processing
//******************************************************
void readLicensePlate(cv::Mat frame_in)
{
	// images for NN
	cv::Mat out, blob_from_image;

	// plate image
	cv::Mat frame_plate (h_plates, w_plates, CV_8UC3);

	cv::Mat frame_copy;

	bool complete_flag = false; // 7 classes detected in from a plate
	std::string plate_lic_string; // model name
	std::string ocr_string; // license number
	
	int plate_model = -1; 

	// output bounding boxes from NN
	std::vector <cv::Rect> object_rois_plates;
	std::vector <cv::Rect> object_rois_ocr;

	object_rois_plates.clear();
	object_rois_ocr.clear();

	// creates a blob from the input image
	cv::dnn::blobFromImage(frame_in, blob_from_image, 1, cv::Size(w_plates, h_plates), cv::Scalar(), swap_rb, false);
	
	// set the input for plates NN
	net_plates.setInput(blob_from_image, "", scale, mean);
	
	// runs forward pass to comput the output of DNN layers
	net_plates.forward(outs_plates, getResultsNN(net_plates));

	frame_in.copyTo(frame_copy);

	// post processing draws the bounding boxes
	// we use the copied image for it
	object_rois_plates = postProcessingPlates(frame_copy, outs_plates, classIds_plates);

	// if any class was detected
	if (object_rois_plates.size())
	{
		std::cout << "\nNumber of plates detected: " << object_rois_plates.size() << "\n";
		// scan all detected plates in the image
		for (int i = 0; i < object_rois_plates.size(); i++) 
		{
			// crop the detected plate from input image and resize it to 635x300 
			cv::resize(frame_in(object_rois_plates[i]), frame_plate, frame_plate.size(), cv::INTER_LINEAR);
			
			// gets the detected plate model 
			plate_model = classIds_plates[i];

			// set the cropped plate frame to OCR NN and compute the output
			cv::dnn::blobFromImage(frame_plate, blob_from_image, 1, cv::Size(w_ocr, h_ocr), cv::Scalar(), swap_rb, false);
			net_ocr.setInput(blob_from_image, "", scale, mean);
			net_ocr.forward(outs_ocr, getResultsNN(net_ocr));

			// draw the bounding boxes of any detected letter and number
			object_rois_ocr = postProcessingOCR(frame_plate, outs_ocr, classIds_ocr);

			// if any letter/number was detected
			if (object_rois_ocr.size())
			{
				// fix the classes order
				classIds_ocr = sortClassesOCR(object_rois_ocr, classIds_ocr);
				
				// convert class id to the respective value
				ocr_string = convertClassesId(classIds_ocr);

				// fix any classification mistake
				ocr_string = fixCharsNumbersOCR(ocr_string, plate_model);

				// mercosul
				std::string out_string;
				if (plate_model == 0)
				{
					plate_lic_string = "mercosul";
					out_string = "mercosul plate model: " + ocr_string;
				}

				// mercosul
				if (plate_model == 1)
				{
					plate_lic_string = "old";
					out_string = "old plate model: " + ocr_string;
				}

				// mercosul
				if (plate_model == 2)
				{
					plate_lic_string = "special";
					out_string = "special plate model: " + ocr_string;
				}

				cv::imshow(out_string.c_str(), frame_plate);

				// saves a copy of the images
				out_string = ocr_string + ".jpg";
				cv::imwrite(out_string.c_str(), frame_plate);

				// terminal output information
				std::cout << "\n" << i << ")" << " Model: " << plate_lic_string << "\n   License Number: " << ocr_string << "\n";

				object_rois_ocr.clear();
				classIds_ocr.clear();
				ocr_string = "";
			}
		}

		cv::imshow("Image", frame_copy);
		std::cout << "\n";
	}

	out.release();
	blob_from_image.release();
	frame_plate.release();
	frame_copy.release();

	return;
}
