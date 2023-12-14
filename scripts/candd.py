#!/usr/bin/env python3
import rospy
import pyrealsense2 as rs
import numpy as np
import cv2
import time
from std_msgs.msg import String

rospy.init_node('realsense_object_detection') # ノードの生成
pub = rospy.Publisher('chatter', String, queue_size=10)
rate = rospy.Rate(10)


# グローバル変数の初期化
prev_accumulated_mask_color = None
depth_frame = None
color_intrinsics = None
start_time = time.time()
object_counter = 0
object_positions = {}
r_x = 0


def get_point(event, x, y, flags, param):
    global depth_frame, color_intrinsics, object_point, r_x

    if event == cv2.EVENT_LBUTTONDOWN:
        if x >= 640:
            return

        depth = depth_frame.get_distance(x, y)
        point = np.array([x, y, depth])

        x = point[0]
        y = point[1]
        z = point[2]

        x, y, z = rs.rs2_deproject_pixel_to_point(color_intrinsics, [x, y], z)

        print("point:",[x,y,z])
        
        r_x = point[0]
        
        
     
def mark_color_object_edges(image, lower_rgb, upper_rgb, min_blob_area=500):
    global prev_accumulated_mask_color, start_time, object_counter, object_positions

    # RGBでの色の範囲を指定
    mask_color = cv2.inRange(image, lower_rgb, upper_rgb)

    # ぼかし処理を追加
    mask_color = cv2.GaussianBlur(mask_color, (5, 5), 0)
    # Cannyエッジ検出を使用してエッジを抽出
    edges = cv2.Canny(mask_color, 50, 150)

    # エッジ画像から輪郭を抽出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭に境界ボックスを描画
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_blob_area:
            x, y, w, h = cv2.boundingRect(contour)
            # マーキングを太くし、赤色で描画
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)

            # 物体の中心座標を取得
            center_x = x + w // 2
            center_y = y + h // 2

            # マーキングした正方形の中心点の座標(平面)からget_pointに値を代入
            get_point(cv2.EVENT_LBUTTONDOWN, center_x, center_y, None, None)

           

    # マスク画像を表示
    cv2.imshow("Mask Color", mask_color)

    # マーキングされた画像を表示
    cv2.imshow("Marked Image", image)

# RealSenseパイプラインの初期化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("ストリーミングを開始")
pipeline.start(config)

cv2.namedWindow('RealsenseImage', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('RealsenseImage', get_point)

# RGBでの色の範囲を指定（例: 緑色）
lower_rgb = np.array([0, 70, 0])
upper_rgb = np.array([75, 255, 100])

while cv2.waitKey(1) < 0:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # 色の範囲内のエッジをマーク
    mark_color_object_edges(color_image, lower_rgb, upper_rgb, min_blob_area=500)

    # 結果を表示
    cv2.imshow("RealsenseImage", color_image)
    
    xyz_str = String()
    xyz_str.data = str(r_x)
    pub.publish(xyz_str)

    rate.sleep()

# パイプラインを停止
pipeline.stop()
cv2.destroyAllWindows()
