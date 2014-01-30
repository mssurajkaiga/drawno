#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/// Global variables

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map - 60";
char* window_name2 = "Edge Map - 80";
char* window_name3 = "Edge Map - 100";

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(int lowThreshold, char* window_name)
{
  /// Reduce noise with a kernel 3x3
  GaussianBlur( src_gray, detected_edges, Size(1,1), 1.5, 1.5 );

  /// Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);

  src.copyTo( dst, detected_edges);
  bitwise_not(dst, detected_edges);
  imshow( window_name, detected_edges);
 }


/** @function main */
int main( int argc, char** argv )
{
  /// Load an image
  src = imread( argv[1] );

  if( !src.data )
  { return -1; }

  /// Create a matrix of the same type and size as src (for dst)
  dst.create( src.size(), src.type() );

  /// Convert the image to grayscale
  cvtColor( src, src_gray, CV_BGR2GRAY );

  /// Create a window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Create a Trackbar for user to enter threshold
  /// Show the image
  CannyThreshold(60, window_name);
  CannyThreshold(80, window_name2);
  CannyThreshold(100, window_name3);

  /// Wait until user exit program by pressing a key
  waitKey(0);

  return 0;
  }