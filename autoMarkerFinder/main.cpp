/************************************************************************************\
  This is improved variant of chessboard corner detection algorithm that
  uses a graph of connected quads. It is based on the code contributed
  by Vladimir Vezhnevets and Philip Gruebele.
  Here is the copyright notice from the original Vladimir's code:
  ===============================================================

  The algorithms developed and implemented by Vezhnevets Vldimir
  aka Dead Moroz (vvp@graphics.cs.msu.ru)
  See http://graphics.cs.msu.su/en/research/calibration/opencv.html
  for detailed information.

  Reliability additions and modifications made by Philip Gruebele.
  <a href="mailto:pgruebele@cox.net">pgruebele@cox.net</a>

  His code was adapted for use with low resolution and omnidirectional cameras
  by Martin Rufli during his Master Thesis under supervision of Davide Scaramuzza, at the ETH Zurich. Further enhancements include:
  - Increased chance of correct corner matching.
  - Corner matching over all dilation runs.

  If you use this code, please cite the following articles:

  1. Scaramuzza, D., Martinelli, A. and Siegwart, R. (2006), A Toolbox for Easily Calibrating Omnidirectional Cameras, Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems  (IROS 2006), Beijing, China, October 2006.
  2. Scaramuzza, D., Martinelli, A. and Siegwart, R., (2006). "A Flexible Technique for Accurate Omnidirectional Camera Calibration and Structure from Motion", Proceedings of IEEE International Conference of Vision Systems  (ICVS'06), New York, January 5-7, 2006.
  3. Rufli, M., Scaramuzza, D., and Siegwart, R. (2008), Automatic Detection of Checkerboards on Blurred and Distorted Images, Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2008), Nice, France, September 2008.

  \************************************************************************************/


// Includes
#include <cstdlib>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/videoio/videoio_c.h>

#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <fstream>
using namespace std;
using std::ifstream;

#ifdef __cplusplus
extern "C" {
#endif
#include "cvcalibinit3.h"
#ifdef __cplusplus
}
#endif
//===========================================================================
// MAIN LOOP 
//===========================================================================
int main( int argc, char** argv )
{
  std::cout << "Starting" << std::endl;

  char cwd[50];
  getcwd(cwd, sizeof(cwd));
  std::cout << "CWD: " << cwd << std::endl;

  // Initializations
  cv::Size board_size = {7, 6};
  const char* input_filename = "pictures.txt";
  FILE* f = nullptr;
  char imagename[1024];
  std::vector<cv::Point2f> image_points_buf; // Matches cvcalibinit3.h
  int found = -2;
  int min_number_of_corners = 42;
  bool DEBUG = true;

  // Create error message file
  std::ofstream error("outputImages/error.txt");

  // Read the "argv" function input arguments
  for(int i = 1; i < argc; i++ )
  {
    const char* s = argv[i];
    if( strcmp( s, "-w" ) == 0 )
    {
      if( sscanf( argv[++i], "%u", &board_size.width ) != 1 || board_size.width <= 0 )
      {
        error << "Invalid board width" << endl;
        error.close();
        return -1;
      }
    }
    else if( strcmp( s, "-h" ) == 0 )
    {
      if( sscanf( argv[++i], "%u", &board_size.height ) != 1 || board_size.height <= 0 )
      {
        error << "Invalid board height" << endl;
        error.close();
        return -1;
      }
    }
    else if( strcmp( s, "-m" ) == 0 )
    {
      if( sscanf( argv[++i], "%u", &min_number_of_corners ) != 1 )
      {
        error << "Invalid minimal number of corners" << endl;
        error.close();
        return -1;
      }
    }
    else if( strcmp( s, "-q" ) == 0 )
    {
      DEBUG = false;
    }
    else if( s[0] != '-' )
      input_filename = s;
    else
    {
      error << "Unknown option" << endl;
      error.close();
      return -1;
    }
  }

  std::cout << "filename: " << input_filename << std::endl;


  if( input_filename )
  {
    f = fopen( input_filename, "rt" );
    if( !f ) {
      error << "The input file could not be opened" << endl;
      return fprintf( stderr, "The input file could not be opened\n" ), -1;
    }
  }

  while (f && fgets(imagename, sizeof(imagename) - 2, f)) 
  {
    int l = (int)strlen(imagename);
    if (l > 0 && imagename[l - 1] == '\n')
      imagename[--l] = '\0';

    if (l > 0 && imagename[0] != '#') // Skip comments
    {
      std::cout << "Loading image: " << imagename << std::endl;
      cv::Mat view = cv::imread(imagename, cv::IMREAD_COLOR);

      if (view.empty()) {
        std::cerr << "Could not load image: " << imagename << std::endl;
        continue;
      }

      // The detector specifically requires a grayscale image
      cv::Mat view_gray;
      cv::cvtColor(view, view_gray, cv::COLOR_BGR2GRAY);

      // Call your improved detector
      int count = 0;
      found = cvFindUVMarkers(view_gray, board_size,
          image_points_buf, &count, 
          min_number_of_corners, DEBUG);

      // Original behavior: if ESC is pressed, stop processing
      int key = cv::waitKey(10);
      if (key == 27) break; 
    }
  }

  // Cleanup
  if (f) fclose(f);
  error.close();

  return found;
}
