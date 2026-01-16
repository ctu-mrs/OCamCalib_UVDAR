#ifndef CVCALIBINIT3_H
#define CVCALIBINIT3_H

#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
#endif

int cvFindUVMarkers( cv::InputArray arr, cv::Size pattern_size,
                             std::vector<cv::Point2f> out_corners, int* out_corner_count,
                             int min_number_of_corners, bool set_debug );

#ifdef __cplusplus
}
#endif

#endif
