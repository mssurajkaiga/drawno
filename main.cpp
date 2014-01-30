#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src, src1, src2; Mat src_gray; Mat src_gray2, src_gray3, input, output, final;
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

double alpha = 0.5;

char* window_name2 = "Edge Map - 80";
char* window_name3 = "Edge Map - 100";

float progress = 0;

Mat range_filter(Mat input, Scalar bgr_min, Scalar bgr_max)
{
    uchar b, g, r;
    int row, col;
    Mat output;
    output.create(input.size(), input.type());
    
    for (row = 0; row < input.rows; row++) {
      for (col = 0; col < input.cols; col++) {
       
        b = input.data[input.step * row + col*3];
        g = input.data[input.step * row + col*3 + 1];
        r = input.data[input.step * row + col*3 + 2];
       
       if(b>=bgr_min[0] && b<=bgr_max[0] && g>=bgr_min[1] && g<=bgr_max[1] && r>=bgr_min[2] && r<=bgr_max[2]) {
          output.data[output.step * row + col*3] = b;
          output.data[output.step * row + col*3 + 1] = g;
          output.data[output.step * row + col*3 + 2] = r;
        }

       else {
          output.data[output.step * row + col*3] = 255;
          output.data[output.step * row + col*3 + 1] = 255;
          output.data[output.step * row + col*3 + 2] = 255;
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
  
  Scalar bgr_min = Scalar(100, 100, 100);
  Scalar bgr_max = Scalar(255, 255, 255);
  output = range_filter(src, bgr_min, bgr_max);

  addWeighted(output, 0.5, detected_edges, 0.5, 0.0, dst);
  imshow( window_name, dst);
}

void morph(Mat &src, Mat &dst, int morph_operator)
{

  int operation = morph_operator + 2;

  Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

  morphologyEx(src, dst, operation, element);
}

/* Marks out region of hand */
Mat mark_hand(Mat* img_hand)
{
  Mat output;
  output.create(img_hand->size(), img_hand->type());
  Mat hsv_img;
  hsv_img.create(img_hand->size(), img_hand->type());

  Scalar hsv_min = Scalar(0, 30, 0, 0);
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

Mat fill_white(Mat &input)
{
  int col, row, min, max;
  uchar i;
  Mat output, temp;
  output.create(input.size(), input.type());
  temp = input.clone();


  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  findContours(temp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

  //medianBlur(temp, input, 1);

  // Finds the contour with the largest area

  int area = 0;
  int idx;
  for(int i=0; i<contours.size();i++) {
    if(area < contours[i].size()) {
      area = contours[i].size();
      idx = i;
    }
  }

  for (row = 0; row < input.rows; row++ ) {
   min = input.cols;
   max = 0;
   for (col = 0; col < input.cols; col++ ) {
    int o = pointPolygonTest(contours[idx], Point2f(col, input.rows - row), false);
    if(o==-1) {
      continue;
    }
     i = input.data[input.step * row + col];
     if (i==255) {
      if (col < min) min = col;
      if (col > max) max = col;
     }
   }

   for (col = 0; col < output.cols; col++ ) {
    if(col>=min && col<=max){
      output.data[output.step * row + col] = 255;
    }
    else {
      output.data[output.step * row + col] = 0;
    }
   }
  }
   return output;
}

Mat combine(Mat &input1, Mat &hand1, Mat &input2, Mat &hand2)
{
  int col, row, min, max;
  uchar i1, i2;
  Mat output;
  output = input1.clone();

  for (row = 0; row < hand1.rows; row++ ) {
    for (col = 0; col < hand1.cols; col++ ) {
     i1 = hand1.data[hand1.step * row + col];
     i2 = hand2.data[hand2.step * row + col];
     if(i1==255 && i2==0) {
        output.data[output.step * row + col] = input2.data[input2.step * row + col];
        output.data[output.step * row + col + 1] = input2.data[input2.step * row + col + 1];
        output.data[output.step * row + col + 2] = input2.data[input2.step * row + col + 2];
     }

     else if(i1==255 && i2==255) {
        output.data[output.step * row + col] = 255;
        output.data[output.step * row + col + 1] = 255;
        output.data[output.step * row + col + 2] = 255;
     }

    }
  }

  return output;
}

void run(Mat src, bool is_cam)
{
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

    output = mark_hand(&src);
    namedWindow("Hand marked", CV_WINDOW_AUTOSIZE);
    imshow("Hand marked", output);

    //threshold(output ,canny_output, 0, 255, 1);
    canny_output = fill_white(output);
    imshow(window_name, canny_output);

    drawing = contour_bound(output);
    namedWindow("Contour bound", CV_WINDOW_AUTOSIZE);
    imshow("Contour bound", drawing);

    cvtColor(canny_output, output, COLOR_GRAY2BGR);
    addWeighted(output, alpha, src, 1.0 - alpha, 0.0, final);
    namedWindow("final window", CV_WINDOW_AUTOSIZE);
    imshow("final window", final);

    if (!is_cam)
      waitKey(0);
    else return;
}

Mat generate_image(Mat src)
{
  if (progress==0) {
    return input;
  }
}

int main( int argc, char** argv )
{
  /// Load source image and convert it to gray
  /*
  if (argc==2) {
  
    src = imread(argv[1], 1);
    run(src, false);
  }
  
  else if(argc==3) {
    cout<<"here";
    src1 = imread(argv[1], 1);
    src2 = imread(argv[2], 1);

    Mat canny_output, output, hand1, hand2;
    canny_output = mark_hand(&src1);
    hand1 = fill_white(canny_output);

    canny_output = mark_hand(&src2);
    hand2 = fill_white(canny_output);

    namedWindow("hand1", CV_WINDOW_AUTOSIZE);
    imshow("hand1", hand1);
    namedWindow("hand2", CV_WINDOW_AUTOSIZE);
    imshow("hand2", hand2);

    output = combine(src1, hand1, src2, hand2);
    namedWindow("Combined image", CV_WINDOW_AUTOSIZE);
    imshow("Combined image", output);\
    waitKey(0);
  }
  */
  input = imread(argv[1], 1);

  //else {
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    namedWindow("Display", CV_WINDOW_AUTOSIZE);
    Mat edges;
    Mat frame, display_image;
    for(;;)
    {
        cap >> frame; // get a new frame from camera
        src = frame.clone();
        display_image = generate_image(src);
        imshow("Display", display_image);
        waitKey(10);
    }
  //}

  return 0;
}