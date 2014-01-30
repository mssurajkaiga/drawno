#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <iostream>

using namespace cv;
using namespace std;

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

Mat range_filter(Mat input, Scalar bgr_min, Scalar bgr_max)
{
    Mat output;
    output.create(input.size(), input.type());
    cout<<"bgr_min = "<<bgr_min;
    cout<<"\nbgr_max = "<<bgr_max;
    cout<<"\n input step = "<<input.step<<"\n";
    cout<<"\n output step = "<<output.step<<"\n";
    cout<<"\n input rows = "<<input.rows<<"\n";
    cout<<"\n input cols = "<<input.cols<<"\n";
    
    for (int row = 0; row < input.rows; row++) {
      for (int col = 0; col < input.cols; col++) {
       
       int b = input.data[input.step * row + col];
       int g = input.data[input.step * row + col + 1];
       int r = input.data[input.step * row + col + 2];
       
       if(b>=bgr_min[0] && b<=bgr_max[0] && g>=bgr_min[1] && g<=bgr_max[1] && r>=bgr_min[2] && r<=bgr_max[2]) {
          output.data[output.step * row + col] = b;
          output.data[output.step * row + col + 1] = g;
          output.data[output.step * row + col + 2] = r;
        }

       else {
          output.data[output.step * row + col] = 255;
          output.data[output.step * row + col + 1] = 255;
          output.data[output.step * row + col + 2] = 255;
        }

      }
    }
    return output;
}

void CannyThreshold(int lowThreshold, char* window_name)
{
  /// Reduce noise with a kernel 3x3
  GaussianBlur( src_gray, detected_edges, Size(1,1), 1.5, 1.5 );

  /// Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);
  Mat output;
  src.copyTo( dst, detected_edges);
  bitwise_not(dst, detected_edges);
  
  Scalar bgr_min = Scalar(0,0,0);
  Scalar bgr_max = Scalar(255,255,255);
  output = range_filter(src, bgr_min, bgr_max);

  imshow( window_name, output);
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
  //CannyThreshold(80, window_name2);
  //CannyThreshold(100, window_name3);

  /// Wait until user exit program by pressing a key
  waitKey(0);

  return 0;
  }