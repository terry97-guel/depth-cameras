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

rospy.init_node("segmetation_demo")

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
# Assert that rs2 node is activated!
assert '/camera/realsense2_camera' in rosnode.get_node_names()

topic_tuples = rostopic.get_topic_list()
# (pub, sub)
# pub/sub: (topic, msg, [nodes])

topic_info = list(filter(lambda iterator: iterator[0]=='/camera/color/image_raw', topic_tuples[0]))[0]

topic_info[0], topic_info[1], topic_info[2]
# %%
from cv_bridge import CvBridge

# Wait for the image message
image_msg = rospy.wait_for_message(topic_info[0], Image, 3)

# Convert the ROS Image message to a OpenCV image
bridge = CvBridge()
image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")

# Plot the image
plt.imshow(image)
plt.axis("off")
plt.show()


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

# %%
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
mask_image[masks[idx-1]] = [255, 144, 30]  # Set color of Mask
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
# Assert that rs2 node is activated!
assert '/camera/realsense2_camera' in rosnode.get_node_names()

topic_tuples = rostopic.get_topic_list()
# (pub, sub)
# pub/sub: (topic, msg, [nodes])

topic_info = list(filter(lambda iterator: iterator[0]=='/camera/depth/color/points', topic_tuples[0]))[0]

topic_info[0], topic_info[1], topic_info[2]

# %%
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2

# Wait for the image message
pcl_msg = rospy.wait_for_message(topic_info[0], PointCloud2, 3)
point_cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(pcl_msg)

# Access the data in the NumPy array
# (example: get the x, y, and z coordinates of the first point)
x = point_cloud_array['x']
y = point_cloud_array['y']
z = point_cloud_array['z']

# %%
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Enable temporal filtering
temporal_filter = rs.temporal_filter()

# Create alignment object
align = rs.align(rs.stream.color)


# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: color and depth
        frames = pipeline.wait_for_frames()
        
        aligned_frames = align.process(frames)
        
        # Apply temporal filtering to the depth frame
        filtered_frames = temporal_filter.process(frames)
        color_frame = filtered_frames.get_color_frame()
        depth_frame = filtered_frames.get_depth_frame()

        # Convert color frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Scale depth values for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Display the color and depth images
        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Image', depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()

# Close all OpenCV windows
cv2.destroyAllWindows()




# %%
# Compare the depth images before and after applying temporal filtering

import pyrealsense2 as rs
import numpy as np
import cv2

# Create a pipeline
pipeline = rs.pipeline()

# Configure the pipeline
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

# Create a temporal filter
temporal_filter = rs.temporal_filter()



filter_magnitude = rs.option.filter_magnitude
filter_smooth_alpha = rs.option.filter_smooth_alpha
filter_smooth_delta = rs.option.filter_smooth_delta

temporal_smooth_alpha=1
temporal_smooth_delta=1


temporal_filter.set_option(filter_smooth_alpha, temporal_smooth_alpha)
temporal_filter.set_option(filter_smooth_delta, temporal_smooth_delta)
    
try:
    while True:
        # Wait for the next frame
        frames = pipeline.wait_for_frames()
        
        # Apply temporal filtering to the depth frame
        filtered_frames = temporal_filter.process(frames).as_frameset()
        
        # Get the depth frame
        depth_frame = frames.get_depth_frame()
        filtered_depth_frame = filtered_frames.get_depth_frame()
        

        depth_image = np.asanyarray(depth_frame.get_data())
        filtered_depth_image = np.asanyarray(filtered_depth_frame.get_data())

        # Scale depth values for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.20), cv2.COLORMAP_JET)
        filtered_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(filtered_depth_image, alpha=0.20), cv2.COLORMAP_JET)

        # Display the color and depth images
        cv2.imshow('Depth Image', depth_colormap)
        cv2.imshow('Filtered Depth Image', filtered_depth_colormap)
        
        cv2.imshow("Differnce", cv2.absdiff(depth_colormap, filtered_depth_colormap))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        

        # Continue with further processing using the filtered frame
        # ...
except Exception as e:
    print(e)
finally:
    # Stop the pipeline
    pipeline.stop()

    
# %%

import pyrealsense2 as rs

# Create a pipeline
pipeline = rs.pipeline()

# Configure the pipeline to stream depth frames
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

import time
time.sleep(1)

for _ in range(10):
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()

    # Get the depth scale for converting depth values
    depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

    # Get the depth intrinsics
    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    # Deproject pixel (100, 100) to point
    x = 100
    y = 100
    depth = depth_frame.get_distance(x, y)
    point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)

    # Print the 3D coordinates
    print("3D Coordinates (x, y, z):", point)

    # Process the point cloud data here...

# Stop the pipeline
pipeline.stop()

