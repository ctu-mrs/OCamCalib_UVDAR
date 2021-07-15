#!/bin/bash

num=$( cat current_cam.txt )
cp ./calib_results.txt $(rospack find mrs_uav_general)/config/uvdar_calibrations/camera_calibrations/calib_results_bf_uv_${num}.txt
