import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("YOLOv8이 설치되지 않았습니다. 설치하세요: pip install ultralytics opencv-python")

class PersonDetector(Node):
    def __init__(self):
        super().__init__('person_detector')
        
        # YOLO 모델 로드
        self.model = YOLO('yolov8n.pt')
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
            
            # YOLO 감지 실행 - person (사람) 감지
            results = self.model.predict(
                source=cv_image,
                classes=[0],  # 0: person (사람 인식 - 앞/뒤/옆 모두 가능)
                conf=0.3,  # 사람 감지를 위한 신뢰도 임계값
                verbose=False,
                show=False,
                save=False
            )
            
            person_detected = False
            detection_count = 0
            
            for r in results:
                # 모든 감지된 물체 확인
                self.get_logger().debug(f"Detections: {len(r.boxes)} boxes")
                
                for box in r.boxes:
                    # person class 확인 (COCO dataset에서 person은 0)
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if cls_id == 0:
                        detection_count += 1
                        person_detected = True
                        
                        self.get_logger().info(
                            f"✅ PERSON DETECTED! Confidence: {confidence:.2f}")
                        
                        # 바운딩 박스 좌표
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # 중심점 계산
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # 바운딩 박스 크기 (거리 추정용)
                        box_width = x2 - x1
                        box_height = y2 - y1
                        box_size = max(box_width, box_height)
                        
                        # 정규화된 좌표로 변환 (-1 ~ 1)
                        h, w = cv_image.shape[:2]
                        norm_x = (center_x - w/2) / (w/2)
                        # 새로운 방식: 화면에서 차지하는 비율
                        # 목표: 사람이 화면 높이의 약 50%를 차지할 때 이상적
                        screen_ratio = box_height / h
                        target_ratio = 0.50
                        # error_y: 양수 = 사람이 크다 (가까움) = 뒤로
                        #         음수 = 사람이 작다 (멀다) = 앞으로
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
                        cv2.putText(cv_image, f'Person {confidence:.2f}', (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        self.get_logger().debug(
                            f'Person detected: pos=({self.smooth_x:.2f}, {self.smooth_y:.2f}), conf={confidence:.2f}'
                        )
            
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
    detector = PersonDetector()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
