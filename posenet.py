#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse

from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="densenet121-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = poseNet(args.network, sys.argv, args.threshold)

# create video sources & outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  

    
    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=args.overlay)


    # print the pose results
    print("detected {:d} objects in image".format(len(poses)))

    for pose in poses:
         # find the keypoint index from the list of detected keypoints
        # you can find these keypoint names in the model's JSON file, 
        # or with net.GetKeypointName() / net.GetNumKeypoints()
        left_wrist_idx = pose.FindKeypoint('left_wrist')
        right_wrist_idx = pose.FindKeypoint('right_wrist')
        left_shoulder_idx = pose.FindKeypoint('left_shoulder')
        right_shoulder_idx = pose.FindKeypoint('right_shoulder')
        left_elbow_idx = pose.FindKeypoint('left_elbow')
        right_elbow_idx = pose.FindKeypoint('right_elbow')
        left_hip_idx = pose.FindKeypoint('left_hip')
        right_hip_idx = pose.FindKeypoint('right_hip')
        left_knee_idx = pose.FindKeypoint('left_knee')
        right_knee_idx = pose.FindKeypoint('right_knee')
        left_ankle_idx = pose.FindKeypoint('left_ankle')
        right_ankle_idx = pose.FindKeypoint('right_ankle')
        

        # if the keypoint index is < 0, it means it wasn't found in the image
        #if left_wrist_idx < 0 or left_shoulder_idx < 0:
        #    continue
        if left_wrist_idx < 0 and left_elbow_idx < 0 and left_shoulder_idx < 0 and left_hip_idx < 0 and left_knee_idx < 0 and left_ankle_idx < 0:
            print("Full body is not in frame.")
            continue
        elif right_wrist_idx < 0 and right_elbow_idx < 0 and right_shoulder_idx < 0 and right_hip_idx < 0 and right_knee_idx < 0 and right_ankle_idx < 0:
            print("Full body is not in frame.")
            continue
        else:
            print("Full body is in frame.")
        
        left_wrist = pose.Keypoints[left_wrist_idx]
        left_shoulder = pose.Keypoints[left_shoulder_idx]

        point_x = left_shoulder.x - left_wrist.x
        point_y = left_shoulder.y - left_wrist.y


    

        
        print(f"person {pose.ID} is pointing towards ({point_x}, {point_y})")
        
        
        print(pose)
        print(pose.Keypoints)
        print('Links', pose.Links)

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break

