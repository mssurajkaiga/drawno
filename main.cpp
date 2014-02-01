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
Mat frame, display_image, display_image_copy;

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

int progress = 0; /* tracks progress of user's drawing - 0, 1, 2, 3 */

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

Mat CannyThreshold(int lowThreshold, Mat input)
{
  /// Reduce noise with a kernel 3x3
  GaussianBlur( src_gray, detected_edges, Size(1,1), 1.5, 1.5 );

  /// Canny detector
  Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);
  Mat output;
  input.copyTo( dst, detected_edges);
  bitwise_not(dst, detected_edges);
  
  cvtColor(detected_edges, dst, CV_BGR2GRAY);
  cvtColor(dst, detected_edges, CV_GRAY2BGR);
  
  int erosion_type, erosion_elem = 0, erosion_size = 2;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
  /// Apply the erosion operation
  erode( detected_edges, output, element );

  return output;

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
/*
void calculate_progress(Mat frame)
{
    int i1,i2, t = 50;
    float wc = 0.01, mean;
    Mat display_image2, filtered, frame_g, filtered_g;
    display_image2 = display_image.clone();
    Scalar bgr_min = Scalar(20,20,0);
    Scalar bgr_max = Scalar(100,100,255);
    filtered = range_filter(frame, bgr_min, bgr_max);
    resize(display_image2, frame, filtered.size(), 0, 0, INTER_LINEAR);
    
    imwrite("images/saved/compared01.jpg", frame);
    imwrite("images/saved/compared1.jpg", filtered);

    cvtColor(filtered, filtered_g, CV_BGR2GRAY);
    //medianBlur(display_image2, filtered);
    //cvtColor(display_image2, filtered, CV_GRAY2BGR);
    cvtColor(frame, frame_g, CV_GRAY2BGR);
    imshow("before", frame);
    imshow("after", filtered);
    waitKey(100);

    for (row = 0; row < frame_g.rows; row++) {
      for (col = 0; col < frame_g.cols; col++) {
        i1 = frame_g.data[frame_g.step * row + col];
        for (col2 = col - frame_g.cols * wc; col2 < col + frame_g.cols*wc; col2++){
          i2 = frame_g.data[frame_g.step * row + col];
          mean = mean + (i1-i2)*(i1-i2);
        }

      }
    }

}
*/

Mat generate_image(Mat src)
{
  /*calculate_progress()*/
    Scalar bgr_min, bgr_max;
    Mat src_test1 = display_image.clone();

    Mat hsv_base;
    Mat hsv_test1;

    cvtColor(src, hsv_base, CV_BGR2HSV);
    cvtColor(src_test1, hsv_test1, CV_BGR2HSV);

    int h_bins = 50; int s_bins = 32;
    int histSize[] = { h_bins, s_bins };
    float h_ranges[] = { 0, 256 };
    float s_ranges[] = { 0, 180 };
    const float* ranges[] = { h_ranges, s_ranges };
    int channels[] = { 0, 1 };
    MatND hist_base;
    MatND hist_test1;

    calcHist(&hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false);
    normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());
    calcHist(&hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false);
    normalize(hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat());
    double base_test1 = compareHist(hist_base, hist_test1, 0);

  switch(progress) {
    case 0:
      dst.create(input.size(), input.type());
      cvtColor(input, src_gray, CV_BGR2GRAY);
      return CannyThreshold(100, input);
      break;

    case 1:
      dst.create(input.size(), input.type());
      cvtColor(input, src_gray, CV_BGR2GRAY);
      return CannyThreshold(80, input);
      break;

    case 2:
      bgr_min = Scalar(200, 200, 200);
      bgr_max = Scalar(255, 255, 255);
      output = range_filter(input, bgr_min, bgr_max);

      dst.create(input.size(), input.type());
      cvtColor(input, src_gray, CV_BGR2GRAY);
      detected_edges = CannyThreshold(80, input);
      
      addWeighted(output, 0.2, detected_edges, 0.8, 0.0, dst);
      return dst;
      break;

    case 3:
      bgr_min = Scalar(150, 150, 150);
      bgr_max = Scalar(255, 255, 255);
      output = range_filter(input, bgr_min, bgr_max);

      dst.create(input.size(), input.type());
      cvtColor(input, src_gray, CV_BGR2GRAY);
      detected_edges = CannyThreshold(80, input);

      addWeighted(output, 0.5, detected_edges, 0.5, 0.5, dst);
      return dst;
      break;

    case 4:
      bgr_min = Scalar(100, 100, 100);
      bgr_max = Scalar(255, 255, 255);
      output = range_filter(input, bgr_min, bgr_max);

      dst.create(input.size(), input.type());
      cvtColor(input, src_gray, CV_BGR2GRAY);
      detected_edges = CannyThreshold(80, input);

      addWeighted(output, 0.5, detected_edges, 0.5, 0.0, dst);
      return dst;
      break;

    case 5:
      bgr_min = Scalar(50, 50, 50);
      bgr_max = Scalar(255, 255, 255);
      output = range_filter(input, bgr_min, bgr_max);
      
      dst.create(input.size(), input.type());
      cvtColor(input, src_gray, CV_BGR2GRAY);
      detected_edges = CannyThreshold(80, input);

      addWeighted(output, 0.5, detected_edges, 0.5, 0.0, dst);
      return dst;
      break;

    case 6:
      bgr_min = Scalar(0, 0, 0);
      bgr_max = Scalar(255, 255, 255);
      output = range_filter(input, bgr_min, bgr_max);
      
      dst.create(input.size(), input.type());
      cvtColor(input, src_gray, CV_BGR2GRAY);
      detected_edges = CannyThreshold(80, input);

      addWeighted(output, 0.5, detected_edges, 0.5, 0.0, dst);
      return dst;
      break;

    default:
      return src;
  }

  return src;
}

int main( int argc, char** argv )
{
  char filename[26] = "images/saved/image000.jpg";
  int c;
  
  namedWindow("before", CV_WINDOW_AUTOSIZE);
  namedWindow("after", CV_WINDOW_AUTOSIZE);

  input = imread(argv[1], 1);
  if(argc==3) {
    progress = (int)argv[2][0] - 48;
  }

  //else {
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    namedWindow("Display", CV_WINDOW_NORMAL);
    setWindowProperty("Display", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    Mat edges, white;
    display_image = Mat::zeros(input.size(), CV_8UC3);
    white = imread("images/white.jpg");
    cap>>frame; // get a new frame from camera
    src = frame.clone();
    for(;;) {
        display_image_copy = generate_image(src);
        display_image = display_image_copy.clone();
        imshow("Display", display_image);
        c = waitKey(5000);
        if(c==27) {
            break;
        }
        else if(c==32){
          progress++;
        }

        imshow("Display", white);
        waitKey(200);
        frame = imread("images/saved/image002.jpg");
        waitKey(300);
        /*
        filename[20]++;
        if(filename[20]>'9') {
            filename[19]++;
            filename[20]='0';
            if(filename[19]>'9') {
              filename[18]++;
              filename[19]='0';
            }
        }
        cout<<filename<<"\n";
        imwrite(filename, frame);
        */
        //calculate_progress(frame);
      }

  return 0;
}