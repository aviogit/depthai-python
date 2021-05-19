#!/usr/bin/env python
import cv2

in_fn  = '../video.h265'
out_fn = 'video.mp4'

in_cap  = cv2.VideoCapture(in_fn)
out_cap = cv2.VideoWriter(out_fn, cv2.VideoWriter.fourcc('A','V','C','1'), 30, (3840, 2160))
#out_cap = cv2.VideoWriter(out_fn, 0x21, 30, (3840, 2160))

while True:
	inret, inframe = in_cap.read()
	print(inret)
	insize = (inframe.shape[1], inframe.shape[0])
	print(inret, insize)
	out_cap.write(inframe)



'''
#include <iostream> // for standard I/O
#include <string>   // for strings

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write

using namespace std;
using namespace cv;

int main()
{
    VideoWriter outputVideo; // For writing the video

    int width = ...; // Declare width here
    int height = ...; // Declare height here
    Size S = Size(width, height); // Declare Size structure

    // Open up the video for writing
    const string filename = ...; // Declare name of file here

    // Declare FourCC code - OpenCV 2.x
    // int fourcc = CV_FOURCC('H','2','6','4');
    // Declare FourCC code - OpenCV 3.x and beyond
    int fourcc = VideoWriter::fourcc('H','2','6','4');

    // Declare FPS here
    double fps = ...;
    outputVideo.open(filename, fourcc, fps, S);

    // Put your processing code here
    // ...

    // Logic to write frames here... see below for more details
    // ...

    return 0;
}
'''
