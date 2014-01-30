#pragma once

#include <stdio.h>
#include <iostream>
#include <vector>

// Include OpenCV's C++ Interface
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;


/*detects the largest object in the image. The object can be a face, mouth, eyes or a car depending on the cascade
 xml chosen. It is much faster than detectManyObjects */
void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth = 320);

// detect many objects in a face including faces . Its slower than detectLargeObject()
void detectManyObjects(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth = 320);
