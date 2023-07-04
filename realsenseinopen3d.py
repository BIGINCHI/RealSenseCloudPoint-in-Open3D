import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import time
import csv

align = rs.align(rs.stream.color)
config = rs.config()

colorizer = rs.colorizer()
config.enable_device_from_file("you//bag//file")

pipeline = rs.pipeline()
profile = pipeline.start(config)
intr = profile.get_stream(
    rs.stream.color).as_video_stream_profile().get_intrinsics()
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
extrinsic = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]


def convert_rs_frames_to_pointcloud(rs_frames):
    aligned_frames = align.process(rs_frames)
    rs_depth_frame = aligned_frames.get_depth_frame()
    np_depth = np.asanyarray(rs_depth_frame.get_data())
    o3d_depth = o3d.geometry.Image(np_depth)
    rs_color_frame = aligned_frames.get_color_frame()
    np_color = np.asanyarray(rs_color_frame.get_data())
    o3d_color = o3d.geometry.Image(np_color)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth, depth_scale=1500.0, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, pinhole_camera_intrinsic, extrinsic)

    return pcd

csv_file = "points.csv"

def main():
    bounding_boxes = []
    rs_frames = pipeline.wait_for_frames()
    bbs_updated_csv_at_time = 0
    pcd = convert_rs_frames_to_pointcloud(rs_frames)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Visualizer",
                      width=1280, height=720)
        
    vis.add_geometry(pcd)

    while True:
        rs_frames = pipeline.wait_for_frames()
        #
        depth_frame = rs_frames.get_depth_frame()
        color_frame = rs_frames.get_color_frame()
    
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        #
        pcd_new = convert_rs_frames_to_pointcloud(rs_frames)
        pcd.points = pcd_new.points
        pcd.colors = pcd_new.colors
        
        # vis.capture_screen_image("file.png", do_render=True)

        vis.update_geometry(pcd)
        if vis.poll_events():
            vis.update_renderer()
        else:
            break

    vis.destroy_window()
    pipeline.stop()


if __name__ == "__main__":
    main()
