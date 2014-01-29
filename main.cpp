#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src_gray; Mat src_gray2, src_gray3;
int thresh = 255;
int max_thresh = 255;
RNG rng(12345);

Mat dst, detected_edges, drawing;

int edgeThresh = 1;
int lowThreshold = 100;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Output";

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(int, void*)
{
  /// Reduce noise with a kernel 3x3
  blur( src_gray2, detected_edges, Size(3,3) );

  /// Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);

  src.copyTo( dst, detected_edges);
  cvtColor(dst, src_gray3, CV_BGR2GRAY );
  unsigned char *input = (unsigned char*)(src_gray3.data);
    for(int i = 0;i < src_gray3.rows;i++) {
      for(int j = 0;j < src_gray3.cols;j++) {
            if (input[src_gray3.step * j + i ] != 0) {
                input[src_gray3.step * j + i ] = 255;
            }
        }
    }
  imshow(window_name, src_gray3);
}

/**
 * @function main
 */
int main( int argc, char** argv )
{
  /// Load source image and convert it to gray
  if (argc!=2) {
    cout<<"Invalid number of arguments - Exiting now...\n";
    exit(0);
  }
  src = imread( argv[1], 1 );

  /// Convert image to gray and blur it
  cvtColor( src, src_gray, COLOR_BGR2GRAY );
  blur( src_gray, src_gray, Size(3,3) );

  /// Create Window
  const char* source_window = "Source";
  namedWindow( source_window, WINDOW_AUTOSIZE );
  imshow( source_window, src );

  /// Create a matrix of the same type and size as src (for dst)
  dst.create( src.size(), src.type() );

  /// Convert the image to grayscale
  drawing = src;
  cvtColor(drawing, src_gray2, CV_BGR2GRAY );

  /// Create a window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  Mat canny_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  /// Detect edges using canny
  Canny( src_gray, canny_output, thresh, thresh*2, 3 );
  /// Find contours
  findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

  /// Draw contours
  drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
  for( size_t i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( drawing, contours, (int)i, color, 2, 8, hierarchy, 0, Point() );
     }

  /// Show the image
  CannyThreshold(0, 0);

  waitKey(0);
  return(0);
}