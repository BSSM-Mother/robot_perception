#!/usr/bin/env python3
"""
간단한 카메라 시뮬레이터
테스트용 이미지를 /camera/image_raw 토픽으로 발행
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class CameraSimulator(Node):
    def __init__(self):
        super().__init__('camera_simulator')
        
        self.publisher = self.create_publisher(Image, 'camera/image_raw', 10)
        self.bridge = CvBridge()
        
        # 타이머 설정 (30Hz)
        self.timer = self.create_timer(0.033, self.timer_callback)
        
        self.frame_count = 0
        self.get_logger().info('Camera Simulator started - Publishing to /camera/image_raw')
    
    def timer_callback(self):
        """카메라 프레임 생성 및 발행"""
        # 640x480 이미지 생성
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 배경 (어두운 회색)
        image[:] = (50, 50, 50)
        
        # 텍스트 배경
        cv2.rectangle(image, (10, 10), (630, 50), (100, 100, 100), -1)
        cv2.putText(image, f'Frame: {self.frame_count}', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 움직이는 사람 시뮬레이션 (노란색 원)
        center_x = int(320 + 150 * np.sin(self.frame_count / 30.0))
        center_y = int(240 + 100 * np.cos(self.frame_count / 50.0))
        
        # 사람 원
        cv2.circle(image, (center_x, center_y), 40, (0, 255, 255), -1)
        
        # 십자 표시
        cv2.line(image, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)
        cv2.line(image, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2)
        
        # 화면 중앙 표시
        cv2.line(image, (319, 0), (319, 480), (255, 0, 0), 1)
        cv2.line(image, (0, 239), (640, 239), (255, 0, 0), 1)
        
        # 정보 텍스트
        cv2.putText(image, f'Person at ({center_x}, {center_y})', (20, 470),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # ROS 이미지 메시지로 변환
        msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        self.publisher.publish(msg)
        
        self.frame_count += 1

def main(args=None):
    rclpy.init(args=args)
    simulator = CameraSimulator()
    rclpy.spin(simulator)
    simulator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
