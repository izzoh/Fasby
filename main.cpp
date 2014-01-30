const char *facerecAlgorithm = "FaceRecognizer.Fisherfaces";
//const char *facerecAlgorithm = "FaceRecognizer.Eigenfaces";
const float UNKNOWN_PERSON_THRESHOLD = 0.7f;
const char *faceCascadeFilename = "lbpcascade_frontalface.xml";
//const char *faceCascadeFilename = "haarcascade_frontalface_alt_tree.xml";  // Haar face detector.
//const char *eyeCascadeFilename1 = "haarcascade_lefteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename2 = "haarcascade_righteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename1 = "haarcascade_mcs_lefteye.xml";       // Good eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename2 = "haarcascade_mcs_righteye.xml";       // Good eye detector for open-or-closed eyes.
const char *eyeCascadeFilename1 = "haarcascade_eye.xml";               // Basic eye detector for open eyes only.
const char *eyeCascadeFilename2 = "haarcascade_eye_tree_eyeglasses.xml"; // Basic eye detector for open eyes if they might wear glasses.
//setting face dimensions. Must be square
const int faceWidth = 70;
const int faceHeight = faceWidth;
//setting camera resolution. Only works for some cameras.
const int DESIRED_CAMERA_WIDTH = 640;
const int DESIRED_CAMERA_HEIGHT = 480;
//setting parameters before taking another snapshot
const double CHANGE_IN_IMAGE_FOR_COLLECTION = 0.3; //change in face expression
const double CHANGE_IN_SECONDS_FOR_COLLECTION = 1.0; //change in time

const char *WindowName = "Rasby Face Recognition";//name of GUI window
const int BORDER = 8;//padding
const bool preprocessLeftAndRightSeparately = true; 
bool m_debug = false;
//include statements
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <opencv2\opencv.hpp>
//my classes includes
#include "detectObject.h"
#include "preprocessFace.h"
#include "recognition.h"
#include "ImageUtils.h"
using namespace cv;
using namespace std;

#if !defined VK_ESCAPE
	#define VK_ESCAPE 0x1B //setting the escape character(27)
#endif

enum MODES {MODE_STARTUP=0, MODE_DETECTION,MODE_COLLECT_FACES, MODE_TRAINING, MODE_RECOGNITION,MODE_DELETE_ALL, MODE_END};
const char* MODE_NAMES[] = {"Startup","Detection","Collect Faces","Training","Recognition","Delete All","ERROR"};
MODES m_mode = MODE_STARTUP;

int m_selectedPerson = -1;
int m_numPersons = 0;
vector<int> m_latestFaces;

//Position of Gui buttons:
Rect m_rcBtnAdd;
Rect m_rcBtnDel;
Rect m_rcBtnDebug;
int m_gui_faces_left = -1;
int m_gui_faces_top = -1;
//convert c++ integer to string
template <typename T> string toString(T t)
{
	ostringstream out;
	out << t;
	return out.str();
}
template <template T> T fromString(string t)
{
	T out;
	istringstream in(t);
	in >>out;
	return out;
}
//load the cascade classifiers for the face and two eyes
void initDetectors(CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1,CascadeClassifier &eyeCascade2)
{
	//load face detection cascade classifier xml file
	try {
		faceCascade.load(faceCascadeFilename);
	}catch (cv::Exception &e){}
	if (faceCascade.empty()){
		cerr << "ERROR: Could not load face Detection Cascade classifier ["<<faceCascadeFilename<<"]"<<endl;
		exit(1);
	}
	cout << "Loaded the Face Detection Cascade classifier ["<<faceCascadeFilename <<"]"<<endl;

	try {
		eyeCascade1.load(eyeCascadeFilename1);
	}catch (cv::Exception &e){}
	if (eyeCascade1.empty()){
		cerr << "ERROR: Could not load eye Detection Cascade classifier ["<<eyeCascadeFilename1<<"]"<<endl;
		exit(1);
	}
	cout << "Loaded the eye Detection Cascade classifier ["<<eyeCascadeFilename1 <<"]"<<endl;

	try {
		eyeCascade2.load(eyeCascadeFilename2);
	}catch (cv::Exception &e){}
	if (eyeCascade2.empty()){
		cerr << "ERROR: Could not load eye Detection Cascade classifier ["<<eyeCascadeFilename2<<"]"<<endl;
		exit(1);
	}
	cout << "Loaded the eye Detection Cascade classifier ["<<eyeCascadeFilename2 <<"]"<<endl;

}
//Get access to the webcam.
void initWebcam(VideoCapture &videoCapture, int cameraNumber)
{
	//Get access to the default camera
	try {
		videoCapture.open(cameraNumber);
	}catch (cv::Exception &e) {}
	if (!videoCapture.isOpened() ){
		cerr <<"ERROR: Could not access the camera."<<endl;
		exit(1);
	}
	cout<<"Loaded Camera" <<cameraNumber <<"."<<endl;
}
//drawing text into the image
Rect drawString(Mat img, string text, Point coord, Scalar color,float fontScale=0.6f,int thickness = 1, int fontFace = FONT_HERSHEY_COMPLEX)
{
	//get the text size & baseline
	int baseline = 0;
	Size textSize = getTextSize(text,fontFace, fontScale, thickness, &baseline);
	baseline += thickness;
	if (coord.y >= 0) {
		coord.y += textSize.height;
	}else {
		coord.y += img.rows - baseline + 1;
	}
	//become right justified if desired
	if (coord.x < 0) {
		coord.x += img.cols - textSize.width + 1;
	}
	//get the bounding box around the text
	Rect boundingRect = Rect(coord.x,coord.y-textSize.height,textSize.width, baseline + textSize.height);
	putText(img, text, coord, fontFace, fontScale, color, thickness, CV_AA);
	return boundingRect;
}
//use drawString to draw GUI Buttons
Rect drawButton(Mat img, string text, Point coord, int minWidth = 0)
{
	int B = BORDER;
	Point textCoord = Point(coord.x + B, coord.y + B);
	Rect rcText = drawString(img, text, textCoord, CV_RGB(0,0,0));
	Rect rcButton = Rect(rcText.x-B, rcText.y -B, rcText.width + 2*B, rcText.height + 2*B);
	//set a minimum button width
	if (rcButton.width < minWidth)
		rcButton.width = minWidth;
	// Make a semi-transparent white rectangle
	Mat matButton = img(rcButton);
	matButton += CV_RGB(90,90,90);
	//Draw a non transparent white border
	rectangle(img, rcButton, CV_RGB(200,200,200), 1, CV_AA);
	// Draw the actual text that will be displayed using anti-aliasing
	drawString(img, text,textCoord, CV_RGB(10,55,20));
	return rcButton;
}

bool isPointInRect(const Point pt, const Rect rc)
{
	if (pt.x >= rc.x && pt.x <= (rc.x + rc.width - 1))
		if (pt.y >= rc.y && pt.y <= (rc.y + rc.height - 1))
			return true;
	return false;
}
//Mouse event handler
void onMouse (int event, int x, int y, int, void*)
{
	//we only care about left clicks not right clicks
	if (event != CV_EVENT_LBUTTONDOWN)
		return;

	//check if user clicked on one of our GUI buttons.
	Point pt = Point(x,y);
	if (isPointInRect(pt, m_rcBtnAdd)){
		cout<<"User clicked [add person] button when numPerson was "<<m_numPersons <<endl;
		if ((m_numPersons == 0) || (m_latestFaces[m_numPersons-1] >= 0)) {
			m_numPersons++;
			m_latestFaces.push_back(-1);
			cout<<"Num Persons: "<<m_numPersons<<endl;
		}
		m_selectedPerson = m_numPersons - 1;
		m_mode = MODE_COLLECT_FACES;
	}
	else if (isPointInRect(pt, m_rcBtnDel)){
		cout<<"User clicked [Delete All] button."<<endl;
		m_mode = MODE_DELETE_ALL;
	}
	else if (isPointInRect(pt, m_rcBtnDebug)){
		cout<<"User clicked [Debug] button. "<<endl;
		m_debug = !m_debug;
		cout<<"Debug mode: "<<m_debug<<endl;
	}
	else {
		cout << " user clicked on the image" <<endl;
		//check if user clicked on one of the faces in the list
		int clickedPerson = -1;
		for (int i=0; i<m_numPersons; i++){
			if (m_gui_faces_top >= 0) {
				Rect rcFace = Rect(m_gui_faces_left, m_gui_faces_top + i * faceHeight,faceWidth, faceHeight);
				if (isPointInRect(pt, rcFace)){
					clickedPerson = i;
					break;
				}
			}
		}
		if (clickedPerson >= 0) {
			m_selectedPerson = clickedPerson;
			m_mode = MODE_COLLECT_FACES;
		}
		//they clicked in the center
		else {
			//change to training mode if it was collecting faces.
			if (m_mode == MODE_COLLECT_FACES) {
				cout<<"User wants to begin training."<<endl;
				m_mode = MODE_TRAINING;
			}
	}

}
}
//the loops at program startup
void recognizeAndTrainUsingWebcam(VideoCapture &videoCapture, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2){
	Ptr<FaceRecognizer> model;
	vector<Mat> preprocessedFaces;
	vector<int> faceLabels;
	Mat old_preprocessedFace;
	double old_time = 0;

	//start in detection Mode
	m_mode = MODE_DETECTION;
	//run forever loop until esc break
	while (true){
		//grab the next camera frame
		Mat cameraFrame;
		do {
			videoCapture >> cameraFrame;
		}while (cameraFrame.empty());
		if (cameraFrame.empty()){
			cerr<<"ERROR: Couldn't grab the next camera frame."<<endl;
			exit(1);
		}
		//Get a copy of the camera frame that we can draw onto.
		Mat displayedFrame;
		cameraFrame.copyTo(displayedFrame);
		//Run the face recognition system on the camera image
		int identity = -1;
		//Find a face and preprocess it to have a standard size and contrast
		Rect faceRect;
		Rect searchedLeftEye, searchedRightEye;
		Point leftEye, rightEye;
		Mat preprocessedFace = getPreprocessedFace(displayedFrame,faceWidth, faceCascade, eyeCascade1,eyeCascade2,preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);
		bool gotFaceAndEyes = false;
		if (preprocessedFace.data)
			gotFaceAndEyes = true;

		//Draw an anti-aliased rectangle around the detected face
		if (faceRect.width > 0) {
			rectangle(displayedFrame, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);
			//Draw light-blue anti-aliased circles for the 2 eyes
			Scalar eyeColor = CV_RGB(0,255,255);
			if (leftEye.x >= 0) {
				circle(displayedFrame, Point(faceRect.x+leftEye.x,faceRect.y + leftEye.y), 6, eyeColor, 1, CV_AA);
			}
			if (rightEye.x >= 0 ){
				circle(displayedFrame, Point(faceRect.x+rightEye.x,faceRect.y + rightEye.y), 6, eyeColor, 1, CV_AA);
			}
		}
		if (m_mode == MODE_DETECTION) {
			//do nothing
		}
		else if (m_mode == MODE_COLLECT_FACES) {
			//check if we have a detected face
			if (gotFaceAndEyes) {
				//check if it is different from previous face
				double imageDiff = 10000000000.0;
				if (old_preprocessedFace.data) {
					imageDiff = getSimilarity(preprocessedFace, old_preprocessedFace);
				}
				//Also record when it happened
				double current_time = (double)getTickCount();
				double timeDiff_seconds = (current_time - old_time) / getTickFrequency();
				//only process the face if it is noticeably diff fro previous
				if ((imageDiff > CHANGE_IN_IMAGE_FOR_COLLECTION) && (timeDiff_seconds > CHANGE_IN_SECONDS_FOR_COLLECTION)) {
					//also add the mirror image to the training set
					Mat mirroredFace;
					flip(preprocessedFace, mirroredFace, 1);
					//Add the face images to the list of detected faces.
					preprocessedFaces.push_back(preprocessedFace);
					preprocessedFaces.push_back(mirroredFace);
					faceLabels.push_back(m_selectedPerson);
					faceLabels.push_back(m_selectedPerson);
					//keeping a reference to the latest of each person
					m_latestFaces[m_selectedPerson] = preprocessedFaces.size() - 2;//point to non-mirrored face
					//show number of collected faces. divide by 2 to remove mirrored images
					cout<<"Saved face " <<(preprocessedFaces.size()/2) <<"for person "<< m_selectedPerson <<endl;
					//Make a white flash on the face, so the user knows a photo has been taken
					Mat displayedFaceRegion = displayedFrame(faceRect);
					displayedFaceRegion += CV_RGB(90,90,90);
					//Keep a copy of the preprocessed face, to compare on next iteration
					old_preprocessedFace = preprocessedFace;
					old_time = current_time;

				}
			}
		}
		else if (m_mode == MODE_TRAINING){

		}
	}
}