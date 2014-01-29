#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src_gray; Mat src_gray2, src_gray3, output;
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

int morph_elem = 0;
int morph_size = 0;
int const max_operator = 4;
int const max_elem = 2;
int const max_kernel_size = 21;

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
            if (input[src_gray3.step * j + i ] == 0) {
                input[src_gray3.step * j + i ] = 255;
            }
            else input[src_gray3.step * j + i ] = 0;
        }
    }
  imshow(window_name, src_gray3);
}

void morph(Mat &src, Mat &dst, int morph_operator)
{

  int operation = morph_operator + 2;

  Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

  morphologyEx(src, dst, operation, element);
}

Mat remove_hand(Mat* img_hand)
{
  Mat output;
  output.create(img_hand->size(), img_hand->type());
  Mat hsv_img;
  hsv_img.create(img_hand->size(), img_hand->type());

  Scalar hsv_min = Scalar(0, 30, 80, 0);
  Scalar hsv_max = Scalar(50, 220, 255, 0);

  cvtColor(*img_hand, hsv_img, CV_BGR2HSV);
  inRange(hsv_img, hsv_min, hsv_max, output);
  return output;
}

Mat contour_bound(Mat input)
{
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  findContours(input, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );
  vector<vector<Point> > contours_poly( contours.size() );
  vector<Rect> boundRect( contours.size() );
  vector<Point2f>center( contours.size() );
  vector<float>radius( contours.size() );

  for( int i = 0; i < contours.size(); i++ )
     { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
       boundRect[i] = boundingRect( Mat(contours_poly[i]) );
       minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
     }


  /// Draw polygonal contour + bonding rects + circles
  Mat drawing = Mat::zeros(output.size(), CV_8UC3 );
  for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
       rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
     }

  return drawing;
}

Mat fill_white(Mat& input)
{
   int col, row, z;
   uchar b, g, r;
   for( y = 0; row < input->height; y++ ) {
     for ( col = 0; col < img->width; col++ ) {
       b = input->imageData[input->widthStep * row + col * 3];
     }
   }
}


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

  Mat canny_output;

  /// Detect edges using canny
  Canny(src_gray, canny_output, thresh, thresh*2, 3);
  namedWindow(window_name, CV_WINDOW_AUTOSIZE);
  imshow(window_name, canny_output);

  output = remove_hand(&src);
  namedWindow("Hand removed", CV_WINDOW_AUTOSIZE);
  imshow("Hand removed", output);

  //threshold(output ,canny_output, 0, 255, 1);
  imshow(window_name, output);

//  drawing = contour_bound(output);
//  imshow(window_name, drawing);
  /*
  /// Show the image
  CannyThreshold(0, 0);
  */
  waitKey(0);
  return(0);
}