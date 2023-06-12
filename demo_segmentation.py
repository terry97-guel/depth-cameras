# %%
import numpy as np
import rospy
import rosnode
import rostopic
import ros_numpy

from pathlib import Path

from sensor_msgs.msg import Image
from matplotlib import pyplot as plt
from cv_bridge import CvBridge

# %%
# Load SAM model
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

from segment_anything import sam_model_registry, SamPredictor
model_type = "vit_l"
checkpoint_path = Path("ckpt")
device = "cuda"

ckpt_dict = {'vit_l': 'sam_vit_l_0b3195.pth','vit_b':'sam_vit_b_01ec64.pth', 'vit_h':'sam_vit_h_4b8939.pth'}
sam = sam_model_registry[model_type](checkpoint=checkpoint_path/ckpt_dict[model_type])
sam.to(device=device)
predictor = SamPredictor(sam)


# %%
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Align Module
align = rs.align(rs.stream.color)

# Create Filters
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

colorizer = rs.colorizer()


# Start streaming
pipeline.start(config)

profile = pipeline.get_active_profile()

depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()


try:
    while True:
        # Wait for a coherent pair of frames: color and depth
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()        
        depth_frame = aligned_frames.get_depth_frame()

        depth_frame = depth_to_disparity.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = disparity_to_depth.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)        
        
        # print(np.asanyarray(depth_frame.get_data()).shape)

        
        colorized_depth_frame = colorizer.colorize(depth_frame)
        
        # Display the color and depth images
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()) * 0.001
        colorized_depth_image = np.asanyarray(colorized_depth_frame.get_data())
        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Image', colorized_depth_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()

# Close all OpenCV windows
cv2.destroyAllWindows()


# %%
image = color_image.copy()
predictor.set_image(image)


# %%
# importing the module
import cv2

input_point_ls = []
input_label_ls = []
img = image.copy()

# function to display the coordinates of
# of the points clicked on the image 
def click_event(event, x, y, flags, params):
    # Reset
    if event == cv2.EVENT_MBUTTONDOWN:
        global img
        global input_point_ls
        global input_label_ls
        
        img = image.copy()
        input_point_ls = []
        input_label_ls = []
        
        print(input_point_ls)
        print("Reset")
        cv2.imshow('image', img)
        
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        input_point_ls.append([x,y])
        input_label_ls.append(1)
        print(input_point_ls)
        # displaying the coordinates
        # on the image window
        cv2.drawMarker(img, (x, y), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
        cv2.imshow('image', img)

  
    # checking for right mouse clicks     
    if event == cv2.EVENT_RBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        input_point_ls.append([x,y])
        input_label_ls.append(0)
        # displaying the coordinates
        # on the image window
        cv2.drawMarker(img, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
        cv2.imshow('image', img)


# displaying the image
cv2.imshow('image', img)

# setting mouse handler for the image
# and calling the click_event() function
cv2.setMouseCallback('image', click_event)

while True:
    # wait for a key to be pressed to exit
    res = cv2.waitKey(0)
    if res == ord('q'):
        cv2.destroyAllWindows()
        break

# %%
input_point = np.stack(input_point_ls)
input_label = np.stack(input_label_ls)

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
masks.shape  # (number_of_masks) x H x W

# %%
def click_event(event, x, y, flags, params):
    global clicked_coord
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        # displaying the coordinates
        # on the image window
        canvas_ = canvas.copy()
        cv2.drawMarker(canvas_, (x, y), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
        cv2.imshow('canvas', canvas_)        
        clicked_coord = [x,y]

h,w = img.shape[:2]
canvas_ls = [cv2.resize(img, (w//2, h//2))]

img = image.copy()
for i, (mask, score) in enumerate(zip(masks, scores)):
    mask_image = np.zeros_like(img)
    mask_image[mask] = [255, 144, 30]  # Set color of Mask
    opacity = 0.4 # Set the opacity of the mask
    overlay = cv2.addWeighted(img, 1-opacity, mask_image, opacity, 0)
    
    canvas_ls.append(cv2.resize(overlay, (w//2, h//2)))

canvas = cv2.vconcat([
    cv2.hconcat([canvas_ls[0], canvas_ls[1]]), 
    cv2.hconcat([canvas_ls[2], canvas_ls[3]])
    ])

cv2.imshow('canvas', canvas)
cv2.setMouseCallback('canvas', click_event)
while True:
    # wait for a key to be pressed to exit
    res = cv2.waitKey(0)
    if res == ord('q'):
        cv2.destroyAllWindows()
        break

x, y = clicked_coord
if x<w//2 and y<h//2:
    idx = 0
elif x>=w//2 and y<=h//2:
    idx = 1
elif x<w//2 and y>h//2:
    idx = 2
elif x>=w//2 and y>=h//2:
    idx = 3
else: raise ValueError

img = image.copy()
mask_image = np.zeros_like(img)
target_mask = masks[idx-1]
mask_image[target_mask] = [255, 144, 30]  # Set color of Mask
opacity = 0.4 # Set the opacity of the mask
overlay = cv2.addWeighted(img, 1-opacity, mask_image, opacity, 0)

cv2.imshow('overlay', overlay)
while True:
    # wait for a key to be pressed to exit
    res = cv2.waitKey(0)
    if res == ord('q'):
        cv2.destroyAllWindows()
        break

# %%
def depth2pcl(depth_img,fx,fy,cx,cy, mask=None):
    """
        Scaled depth image to pointcloud
    """
    
    height = depth_img.shape[0]
    width = depth_img.shape[1]
    indices = np.indices((height, width),dtype=np.float32).transpose(1,2,0)
    z_e = depth_img
    
    if mask is not None:
        indices = indices[mask]
        z_e = z_e[mask]
        
    x_e = (indices[..., 1] - cx) * z_e / fx
    y_e = (indices[..., 0] - cy) * z_e / fy
    
    # Order of y_ e is reversed !
    pcl = np.stack([z_e, -x_e, -y_e], axis=-1)
    return pcl # [H x W x 3]

pcl = depth2pcl(depth_image,
                fx=depth_intrinsics.fx,
                fy=depth_intrinsics.fy,
                cx=depth_intrinsics.ppx,
                cy=depth_intrinsics.ppy,
                mask=None)
pcl.shape

pcl[mask].shape
# %%
depth_img = depth_image
fx=depth_intrinsics.fx
fy=depth_intrinsics.fy
cx=depth_intrinsics.ppx
cy=depth_intrinsics.ppy


height = depth_img.shape[0]
width = depth_img.shape[1]
indices = np.indices((height, width),dtype=np.float32).transpose(1,2,0)[mask]

z_e = depth_img[mask]
x_e = (indices[..., 1] - cx) * z_e / fx
y_e = (indices[..., 0] - cy) * z_e / fy

# Order of y_ e is reversed !
pcl = np.stack([z_e, -x_e, -y_e], axis=-1) # [H x W x 3] 
pcl

# %%

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import sys

from model.mujoco_parser import MuJoCoParserClass
from model.util import sample_xyzs,rpy2r,r2quat
np.set_printoptions(precision=2,suppress=True,linewidth=100)

plt.rc('xtick',labelsize=6); plt.rc('ytick',labelsize=6)
print ("MuJoCo version:[%s]"%(mujoco.__version__))



# %%
from sensor_msgs.msg import JointState
import actionlib
from control_msgs.msg import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import copy


class UR(object):
    def __init__(self,):
        self.JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.client      = None
        
   
    def move_arm_speed(self, traj:JointTrajectory, speed_limit):
        joint_pos1 = rospy.wait_for_message("joint_states", JointState).position
        rospy.sleep(0.1)
        joint_pos2 = rospy.wait_for_message("joint_states", JointState).position

        joint_pos1, joint_pos2 = np.array(joint_pos1), np.array(joint_pos2)

        diff = np.linalg.norm(joint_pos2 - joint_pos1)

        
        if diff > 5e-3:            
            raise Exception(f"Loose Connection, diff:{diff}")
        assert np.linalg.norm(joint_pos2) < 20

        try: 
            g = FollowJointTrajectoryGoal()
            g.trajectory = copy.deepcopy(traj)
            g.trajectory.joint_names = self.JOINT_NAMES
            joint_states = rospy.wait_for_message("joint_states", JointState)
            joints_pos   = joint_states.position

            if np.linalg.norm(joints_pos) > 10:
                print("Loose Connection")
                return None

            init_point = JointTrajectoryPoint()
            init_point.positions = np.array(joints_pos)
            init_point.time_from_start = rospy.Duration.from_sec(0.0)
            init_point.velocities = [0 for _ in range(6)]
            g.trajectory.points.insert(0, copy.deepcopy(init_point))

            q_list = []
            time_list = []
            for point in g.trajectory.points:
                q_list.append(point.positions)
                time_list.append(point.time_from_start.to_sec())

            print(q_list)

            for i in range(len(q_list)-1):
                q_before = np.array(q_list[i])
                q_after  = np.array(q_list[i+1])
                print(q_after, q_before)
                time_before = time_list[i]
                time_after = time_list[i+1]

                diff_q = q_after - q_before
                diff_time = time_after - time_before
                print(f"diff_q:{diff_q}")
                print(f"diff_time:{diff_time}")
                speed = np.linalg.norm(diff_q)/diff_time
                print(speed)
                if speed >= speed_limit:
                    raise Exception(f"Speed is too fast: {speed} < {speed_limit}")
                    

            self.client.send_goal(g)
        except KeyboardInterrupt:
            self.client.cancel_goal()
            raise
        except:
            raise  

    def execute_arm_speed(self, traj, speed_limit):
        try:
            self.client = actionlib.SimpleActionClient('follow_joint_trajectory', FollowJointTrajectoryAction)
            print("Waiting for server...")
            self.client.wait_for_server()
            print("Connected to server")
            """ Initialize """
            self.move_arm_speed(traj, speed_limit)
            print("Finish plan")

        except KeyboardInterrupt:
            rospy.signal_shutdown("KeyboardInterrupt")
            raise      

    def execute_arm_time(self, joints, movetime):
        try:
            self.client = actionlib.SimpleActionClient('follow_joint_trajectory', FollowJointTrajectoryAction)
            print("Waiting for server...")
            self.client.wait_for_server()
            print("Connected to server")
            """ Initialize """
            self.move_arm_time(joints, movetime)
            print("Finish plan")

        except KeyboardInterrupt:
            rospy.signal_shutdown("KeyboardInterrupt")
            raise      
# %%

