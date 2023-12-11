#!/usr/bin/env python3
import rospy
import pyrealsense2 as rs
import numpy as np
import cv2
import time

rospy.init_node('realsense_object_detection') # ノードの生成


# グローバル変数の初期化
prev_accumulated_mask_color = None
depth_frame = None
color_intrinsics = None
start_time = time.time()
object_counter = 0
object_positions = {}



def get_point(event, x, y, flags, param):
    global depth_frame, color_intrinsics, object_point

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
        
        
        
     
def mark_color_object_edges(image, lower_rgb, upper_rgb, min_blob_area=500, max_distance_between_objects=30):
    global prev_accumulated_mask_color, start_time, object_counter, object_positions

    # RGBでの色の範囲を指定
    mask_color = cv2.inRange(image, lower_rgb, upper_rgb)

    # 累積マスクの初期化
    if prev_accumulated_mask_color is None:
        prev_accumulated_mask_color = np.zeros_like(mask_color)

    # 1秒間のデータを累積
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        accumulated_mask_color = cv2.bitwise_or(prev_accumulated_mask_color, mask_color)

        # 輪郭を検出
        edges = cv2.Canny(accumulated_mask_color, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 物体ごとに処理
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_blob_area:
                x, y, w, h = cv2.boundingRect(contour)

                # マーキングを太くし、赤色で描画
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)

                # 物体の中心座標を取得
                center_x = x + w // 2
                center_y = y + h // 2

                # 既存の物体と比較して近い座標の場合、同じ物体として扱う
                is_new_object = True
                for pos, obj_id in object_positions.items():
                    dist = np.sqrt((center_x - pos[0]) ** 2 + (center_y - pos[1]) ** 2)
                    if dist < max_distance_between_objects:
                        object_positions[(center_x, center_y)] = obj_id
                        is_new_object = False
                        break

                if is_new_object:
                    object_counter += 1
                    object_positions[(center_x, center_y)] = object_counter

                # 物体の番号を中心に表示
                cv2.putText(image, str(object_positions[(center_x, center_y)]),
                            (center_x - 10, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 1秒前のデータを保存
        prev_accumulated_mask_color = accumulated_mask_color
        start_time = time.time()

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

            # 物体の座標とIDを保存
            object_positions[(center_x, center_y)] = object_positions.get((center_x, center_y), object_counter)

            # 物体の番号を中心に表示
            cv2.putText(image, str(object_positions[(center_x, center_y)]),
                        (center_x - 10, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # マーキングした正方形の中心点の座標(平面)からget_pointに値を代入
            get_point(cv2.EVENT_LBUTTONDOWN, center_x, center_y, None, None)

    # 累積されたマスク画像を表示
    if prev_accumulated_mask_color is not None and prev_accumulated_mask_color.shape[0] > 0 and prev_accumulated_mask_color.shape[1] > 0:
        cv2.imshow("Accumulated Mask", prev_accumulated_mask_color)

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
    mark_color_object_edges(color_image, lower_rgb, upper_rgb, min_blob_area=500, max_distance_between_objects=30)

    # 結果を表示
    cv2.imshow("RealsenseImage", color_image)

# パイプラインを停止
pipeline.stop()
cv2.destroyAllWindows()
