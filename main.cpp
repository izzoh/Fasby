//const char *facerecAlgorithm = "FaceRecognizer.Fisherfaces";
const char *facerecAlgorithm = "FaceRecognizer.Eigenfaces";
const float UNKNOWN_PERSON_THRESHOLD = 0.5f;
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

const char *windowName = "Rasby Face Recognition";//name of GUI window
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
template <typename T> T fromString(string t)
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
			//check if there is enoughData to train from
			bool haveEnoughData = true;
			if (strcmp(facerecAlgorithm, "FaceRecognizer.Fisherfaces")==0) {
				if ((m_numPersons < 2) || (m_numPersons == 2 && m_latestFaces[1] < 0)){
					cout <<"Warning: More data needed to train the system. Fisherface needs at least two people to train with."<<endl;
					haveEnoughData = false;
				}
			}
			if (m_numPersons < 1 || preprocessedFaces.size() <= 0 || preprocessedFaces.size() != faceLabels.size()) {
				cout <<"Warning: Need some training data before it can be learnt! collect more faces."<<endl;
				haveEnoughData = false;
			}

			if (haveEnoughData) {
				model = learnCollectedFaces(preprocessedFaces, faceLabels, facerecAlgorithm);
				if (m_debug)
					showTrainingDebugData(model, faceWidth, faceHeight);
				//start recognizing
				m_mode = MODE_RECOGNITION;


			}else {
				//go back to collection mode
				m_mode = MODE_COLLECT_FACES;
			}
		}
		else if (m_mode == MODE_RECOGNITION) {
			if (gotFaceAndEyes && (preprocessedFaces.size() > 0) && (preprocessedFaces.size() == faceLabels.size())){
				//generate a face approximation by back-projecting eignevectors and eigenvalues
				Mat reconstructedFace;
				reconstructedFace = reconstructFace(model, preprocessedFace);
				if (m_debug)
					if (reconstructedFace.data)
						imshow("reconstructedFace",reconstructedFace);
				//verify if the reconstructed face looks like that on the screen
				double similarity = getSimilarity(preprocessedFace, reconstructedFace);
				string outputStr;
				if (similarity < UNKNOWN_PERSON_THRESHOLD) {
					//identify who is in the preprocessed face image
					identity = model->predict(preprocessedFace);
					outputStr = toString(identity);
				}
				else {
					//since the confidence is low, assume it is an unknown person
					outputStr = "unknown";
				}
				cout <<"Identity: "<<outputStr<<". similarity: "<< ((1-similarity)*100) <<"%"<<endl;
				//Show the confidence rating on rating
				int cx = (displayedFrame.cols - faceWidth) / 2;
				Point ptBottomRight = Point(cx - 5, BORDER + faceHeight);
				Point ptTopLeft = Point(cx - 15, BORDER);
				//Draw a grey line showing threshold for an unknown person
				Point ptThreshold = Point(ptTopLeft.x, ptBottomRight.y - (1.0 - UNKNOWN_PERSON_THRESHOLD) * faceHeight);
				rectangle(displayedFrame, ptThreshold, Point(ptBottomRight.x, ptThreshold.y), CV_RGB(200,200,200), 1, CV_AA);
				//crop the confidence rating between 0.0 to 1.0
				double confidenceRatio = 1.0 - min(max(similarity, 0.0), 1.0);
				Point ptConfidence = Point(ptTopLeft.x, ptBottomRight.y - confidenceRatio * faceHeight);
				//show light blue confidence bar
				rectangle(displayedFrame, ptConfidence, ptBottomRight, CV_RGB(0,255,255), CV_FILLED, CV_AA);
				//show gray border
				rectangle(displayedFrame, ptTopLeft, ptBottomRight, CV_RGB(200,200,200),1,CV_AA);

			}
		}
		else if (m_mode == MODE_DELETE_ALL) {
			//Restart everything
			m_selectedPerson = -1;
			m_numPersons = 0;
			m_latestFaces.clear();
			preprocessedFaces.clear();
			faceLabels.clear();
			old_preprocessedFace = Mat();

			//Restart in Detection mode.
			m_mode = MODE_DETECTION;
		}
		else {
			cerr << "ERROR: Invalid run mode " << m_mode <<endl;
			exit(1);
		}
		//help info
		string help;
        Rect rcHelp;
        if (m_mode == MODE_DETECTION)
            help = "Click [Add Person] when ready to collect faces.";
        else if (m_mode == MODE_COLLECT_FACES)
            help = "Click anywhere to train from your " + toString(preprocessedFaces.size()/2) + " faces of " + toString(m_numPersons) + " people.";
        else if (m_mode == MODE_TRAINING)
            help = "Please wait while your " + toString(preprocessedFaces.size()/2) + " faces of " + toString(m_numPersons) + " people builds.";
        else if (m_mode == MODE_RECOGNITION)
            help = "Click people on the right to add more faces to them, or [Add Person] for someone new.";
        if (help.length() > 0) {
            // Draw it with a black background and then again with a white foreground.
            // Since BORDER may be 0 and we need a negative position, subtract 2 from the border so it is always negative.
            float txtSize = 0.4;
            drawString(displayedFrame, help, Point(BORDER, -BORDER-2), CV_RGB(0,0,0), txtSize);  // Black shadow.
            rcHelp = drawString(displayedFrame, help, Point(BORDER+1, -BORDER-1), CV_RGB(255,255,255), txtSize);  // White text.
		}
		if (m_mode >= 0 && m_mode < MODE_END) {
			string modeStr = "MODE: "+string(MODE_NAMES[m_mode]);
			drawString(displayedFrame, modeStr, Point(BORDER, -BORDER-2 - rcHelp.height), CV_RGB(0,0,0));       // Black shadow
            drawString(displayedFrame, modeStr, Point(BORDER+1, -BORDER-1 - rcHelp.height), CV_RGB(0,255,0)); // Green text 
		}
		// Show the current preprocessed face in the top-center of the display.
        int cx = (displayedFrame.cols - faceWidth) / 2;
        if (preprocessedFace.data) {
            // Get a BGR version of the face, since the output is BGR color.
            Mat srcBGR = Mat(preprocessedFace.size(), CV_8UC3);
            cvtColor(preprocessedFace, srcBGR, CV_GRAY2BGR);
            // Get the destination ROI (and make sure it is within the image!).
            //min(m_gui_faces_top + i * faceHeight, displayedFrame.rows - faceHeight);
            Rect dstRC = Rect(cx, BORDER, faceWidth, faceHeight);
            Mat dstROI = displayedFrame(dstRC);
            // Copy the pixels from src to dst.
            srcBGR.copyTo(dstROI);
        }
        // Draw an anti-aliased border around the face, even if it is not shown.
        rectangle(displayedFrame, Rect(cx-1, BORDER-1, faceWidth+2, faceHeight+2), CV_RGB(200,200,200), 1, CV_AA);

        // Draw the GUI buttons into the main image.
        m_rcBtnAdd = drawButton(displayedFrame, "Add Person", Point(BORDER, BORDER));
        m_rcBtnDel = drawButton(displayedFrame, "Delete All", Point(m_rcBtnAdd.x, m_rcBtnAdd.y + m_rcBtnAdd.height), m_rcBtnAdd.width);
        m_rcBtnDebug = drawButton(displayedFrame, "Debug", Point(m_rcBtnDel.x, m_rcBtnDel.y + m_rcBtnDel.height), m_rcBtnAdd.width);

        // Show the most recent face for each of the collected people, on the right side of the display.
        m_gui_faces_left = displayedFrame.cols - BORDER - faceWidth;
        m_gui_faces_top = BORDER;
        for (int i=0; i<m_numPersons; i++) {
            int index = m_latestFaces[i];
            if (index >= 0 && index < (int)preprocessedFaces.size()) {
                Mat srcGray = preprocessedFaces[index];
                if (srcGray.data) {
                    // Get a BGR version of the face, since the output is BGR color.
                    Mat srcBGR = Mat(srcGray.size(), CV_8UC3);
                    cvtColor(srcGray, srcBGR, CV_GRAY2BGR);
                    // Get the destination ROI (and make sure it is within the image!).
                    int y = min(m_gui_faces_top + i * faceHeight, displayedFrame.rows - faceHeight);
                    Rect dstRC = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
                    Mat dstROI = displayedFrame(dstRC);
                    // Copy the pixels from src to dst.
                    srcBGR.copyTo(dstROI);
                }
            }
        }

        // Highlight the person being collected, using a red rectangle around their face.
        if (m_mode == MODE_COLLECT_FACES) {
            if (m_selectedPerson >= 0 && m_selectedPerson < m_numPersons) {
                int y = min(m_gui_faces_top + m_selectedPerson * faceHeight, displayedFrame.rows - faceHeight);
                Rect rc = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
                rectangle(displayedFrame, rc, CV_RGB(255,0,0), 3, CV_AA);
            }
        }

        // Highlight the person that has been recognized, using a green rectangle around their face.
        if (identity >= 0 && identity < 1000) {
            int y = min(m_gui_faces_top + identity * faceHeight, displayedFrame.rows - faceHeight);
            Rect rc = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
            rectangle(displayedFrame, rc, CV_RGB(0,255,0), 3, CV_AA);
        }

        // Show the camera frame on the screen.
        imshow(windowName, displayedFrame);

        // If the user wants all the debug data, show it to them!
        if (m_debug) {
            Mat face;
            if (faceRect.width > 0) {
                face = cameraFrame(faceRect);
                if (searchedLeftEye.width > 0 && searchedRightEye.width > 0) {
                    Mat topLeftOfFace = face(searchedLeftEye);
                    Mat topRightOfFace = face(searchedRightEye);
                    imshow("topLeftOfFace", topLeftOfFace);
                    imshow("topRightOfFace", topRightOfFace);
                }
            }

            if (!model.empty())
                showTrainingDebugData(model, faceWidth, faceHeight);
        }

        
        // IMPORTANT: Wait for atleast 20 milliseconds, so that the image can be displayed on the screen!
        // Also checks if a key was pressed in the GUI window. Note that it should be a "char" to support Linux.
        char keypress = waitKey(20);  // This is needed if you want to see anything!

        if (keypress == VK_ESCAPE) {   // Escape Key
            // Quit the program!
            break;
        }
	}//end while loop

}


int main(int argc, char *argv[]) {
	CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade1;
    CascadeClassifier eyeCascade2;
    VideoCapture videoCapture;

	initDetectors(faceCascade, eyeCascade1, eyeCascade2);
	cout<<endl;
	cout<<"Press Esc key to quit the program"<<endl;
	//specify a camera
	int cameraNumber = 0;
	if (argc > 1) {
		cameraNumber = atoi(argv[1]);
	}
	//access webcam
	initWebcam(videoCapture,cameraNumber);
	//set camera resolutions
	videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
	videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);
	//create gui
	namedWindow(windowName);
	//mouse
	setMouseCallback(windowName, onMouse, 0);
	recognizeAndTrainUsingWebcam(videoCapture, faceCascade, eyeCascade1, eyeCascade2);
	return 0;
}