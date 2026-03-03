import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import sys

class PersonDetector(Node):
    def __init__(self):
        super().__init__('person_detector')

        # OpenCV HOG 사람 감지기 초기화
        try:
            self.get_logger().info('Initializing OpenCV HOG Person Detector...')
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.get_logger().info('✓ HOG detector initialized successfully')

            # 감지 파라미터
            self.detection_confidence = 0.5
            self.use_hog = True

        except Exception as e:
            self.get_logger().error(f'Failed to initialize HOG detector: {str(e)}')
            raise
        self.bridge = CvBridge()

        # 구독/발행 설정
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        self.person_pos_pub = self.create_publisher(
            Point, 'person_position', 10
        )

        self.detection_pub = self.create_publisher(
            Image, 'detection_image', 10
        )
        self.annotated_pub = self.create_publisher(
            Image, '/camera/annotated', 10
        )

        self.get_logger().info('Person Detector initialized')
        # 간단한 위치 평활화 변수
        self.smooth_x = 0.0
        self.smooth_y = 0.0
        self.alpha = 0.6  # 최근값 가중치

    def image_callback(self, msg):
        """카메라 이미지 콜백"""
        try:
            # ROS 메시지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = cv_image.shape[:2]

            # HOG 사람 감지 실행
            # Raspberry Pi 최적화: 이미지 크기 축소
            scale = 1.0
            if w > 640:
                scale = 640.0 / w
                small_image = cv2.resize(cv_image, None, fx=scale, fy=scale)
            else:
                small_image = cv_image

            # HOG 감지 실행
            boxes, weights = self.hog.detectMultiScale(
                small_image,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05
            )

            person_detected = False
            detection_count = 0

            # 감지된 사람들 처리
            if len(boxes) > 0:
                # 가장 큰 바운딩 박스 선택 (가장 가까운 사람)
                areas = [(x, y, w_box, h_box, w_box * h_box, weight)
                        for (x, y, w_box, h_box), weight in zip(boxes, weights)]
                areas.sort(key=lambda x: x[4], reverse=True)

                for x, y, w_box, h_box, area, weight in areas[:3]:  # 최대 3명까지
                    if weight < self.detection_confidence:
                        continue

                    detection_count += 1
                    person_detected = True

                    # 원래 크기로 변환
                    x1 = int(x / scale)
                    y1 = int(y / scale)
                    x2 = int((x + w_box) / scale)
                    y2 = int((y + h_box) / scale)

                    confidence = min(weight / 2.0, 1.0)  # 정규화

                    self.get_logger().info(
                        f"✅ PERSON DETECTED! Confidence: {confidence:.2f}")

                    # 중심점 계산
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # 바운딩 박스 크기 (거리 추정용)
                    box_width = x2 - x1
                    box_height = y2 - y1

                    # 정규화된 좌표로 변환 (-1 ~ 1)
                    norm_x = (center_x - w/2) / (w/2)
                    # 화면에서 차지하는 비율
                    screen_ratio = box_height / h
                    target_ratio = 0.50
                    norm_y = (screen_ratio - target_ratio) / target_ratio

                    # 위치 평활화 (EMA)
                    self.smooth_x = self.alpha * norm_x + (1 - self.alpha) * self.smooth_x
                    self.smooth_y = self.alpha * norm_y + (1 - self.alpha) * self.smooth_y

                    # 위치 발행 (평활화 사용)
                    pos_msg = Point()
                    pos_msg.x = self.smooth_x
                    pos_msg.y = self.smooth_y
                    pos_msg.z = confidence  # 신뢰도
                    self.person_pos_pub.publish(pos_msg)

                    # 시각화
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(cv_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                    cv2.putText(cv_image, f'PERSON {confidence:.2f}', (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    self.get_logger().debug(
                        f'Person detected: pos=({self.smooth_x:.2f}, {self.smooth_y:.2f}), conf={confidence:.2f}'
                    )

                    # 가장 큰 사람만 추적
                    break

            if not person_detected:
                self.get_logger().info(f'⚠️ No person detected in this frame (image size: {cv_image.shape[1]}x{cv_image.shape[0]})')
            else:
                self.get_logger().info(f'✓ Person detected: {detection_count} detection(s)')

            # 감지 결과 이미지 발행
            detection_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.detection_pub.publish(detection_msg)
            self.annotated_pub.publish(detection_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    try:
        detector = PersonDetector()
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in person_detector node: {e}', file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            detector.destroy_node()
        except:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()
