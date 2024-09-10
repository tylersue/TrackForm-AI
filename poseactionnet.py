
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


    from jetson_inference import actionNet
    from jetson_utils import videoSource, videoOutput, cudaFont, Log

    # parse the command line
    parser = argparse.ArgumentParser(description="Classify the action/activity of an image sequence.", 
                                    formatter_class=argparse.RawTextHelpFormatter, 
                                    epilog=actionNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

    parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
    parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
    parser.add_argument("--network", type=str, default="resnet-34", help="pre-trained model to load (see below for options)")

    try:
        args = parser.parse_known_args()[0]
    except:
        print("")
        parser.print_help()
        sys.exit(0)


    # load the recognition network
    net = actionNet(args.network, sys.argv)

    # create video sources & outputs
    input = videoSource(args.input, argv=sys.argv)
    output = videoOutput(args.output, argv=sys.argv)
    font = cudaFont()

    # process frames until EOS or the user exits
    while True:
        # capture the next image
        img = input.Capture()

        if img is None: # timeout
            continue  

        # classify the action sequence
        class_id, confidence = net.Classify(img)
        class_desc = net.GetClassDesc(class_id)
    
        print(f"actionnet:  {confidence * 100:2.5f}% class #{class_id} ({class_desc})")
    
        # overlay the result on the image	
        font.OverlayText(img, img.width, img.height, "{:25.2f}% {:s}".format(confidence * 100, class_desc), 400, 5, font.White, font.Gray40)

        # render the image
        output.Render(img)

        # update the title bar
        output.SetStatus("actionNet {:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

        # print out performance info
        net.PrintProfilerTimes()

        # exit on input/output EOS
        if not input.IsStreaming() or not output.IsStreaming():
            break