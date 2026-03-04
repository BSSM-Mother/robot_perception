import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import sys
import traceback
from ultralytics import YOLO

# COCO 클래스 인덱스
CLASS_PERSON = 0
CLASS_BALL = 32  # sports ball


class PersonDetector(Node):
    def __init__(self):
        super().__init__('person_detector')

        # 모델 경로 파라미터
        # PC/데스크탑:       'yolo11n.pt'
        # Raspberry Pi (권장): NCNN 포맷 경로 'yolo11n_ncnn_model'
        model_path = self.declare_parameter('model_path', 'yolo11n.pt').value
        conf_thresh = self.declare_parameter('conf_threshold', 0.4).value

        self.get_logger().info(f'Loading YOLO11 model: {model_path}')
        try:
            # NCNN 모델 폴더가 없으면 자동으로 export
            if not os.path.exists(model_path):
                self.get_logger().warn(
                    f'Model not found at "{model_path}". '
                    'Downloading yolo11n.pt and exporting to NCNN...'
                )
                base_pt = 'yolo11n.pt'
                tmp = YOLO(base_pt)          # .pt 자동 다운로드
                exported = tmp.export(format='ncnn')
                # export()가 반환하는 경로 or 기본 폴더명으로 재설정
                ncnn_dir = str(exported) if exported else 'yolo11n_ncnn_model'
                self.get_logger().info(f'✓ NCNN export complete: {ncnn_dir}')
                model_path = ncnn_dir

            self.model = YOLO(model_path)
            self.get_logger().info('✓ YOLO11 model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO11 model: {e}')
            raise

        self.conf_threshold = conf_thresh
        self.bridge = CvBridge()

        # EMA 평활화 가중치 (최근값 비율)
        self.alpha = 0.6
        self.smooth_x = 0.0
        self.smooth_y = 0.0
        self.smooth_ball_x = 0.0
        self.smooth_ball_y = 0.0

        # 구독/발행 설정
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.person_pos_pub = self.create_publisher(Point, 'person_position', 10)
        self.ball_pos_pub = self.create_publisher(Point, 'ball_position', 10)
        self.detection_pub = self.create_publisher(Image, 'detection_image', 10)
        self.annotated_pub = self.create_publisher(Image, '/camera/annotated', 10)

        self.get_logger().info('Person Detector (YOLO11) initialized')

    def image_callback(self, msg):
        """카메라 이미지 콜백"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = cv_image.shape[:2]

            # YOLO11 추론
            # imgsz=320: Raspberry Pi 성능 최적화 (속도 우선)
            # imgsz=640: 감지 정확도 우선 (데스크탑용)
            results = self.model(
                cv_image,
                imgsz=320,
                conf=self.conf_threshold,
                classes=[CLASS_PERSON, CLASS_BALL],
                verbose=False,
            )[0]

            person_detected = False
            ball_detected = False

            if results.boxes is not None:
                # 신뢰도 내림차순 정렬
                boxes = sorted(results.boxes, key=lambda b: float(b.conf[0]), reverse=True)

                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    if cls == CLASS_PERSON and not person_detected:
                        person_detected = True
                        box_h = y2 - y1

                        # 정규화 좌표 (-1 ~ 1)
                        norm_x = (cx - w / 2.0) / (w / 2.0)
                        screen_ratio = box_h / h
                        norm_y = (screen_ratio - 0.5) / 0.5

                        # EMA 평활화
                        self.smooth_x = self.alpha * norm_x + (1 - self.alpha) * self.smooth_x
                        self.smooth_y = self.alpha * norm_y + (1 - self.alpha) * self.smooth_y

                        pos_msg = Point()
                        pos_msg.x = self.smooth_x
                        pos_msg.y = self.smooth_y
                        pos_msg.z = conf
                        self.person_pos_pub.publish(pos_msg)

                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(cv_image, f'PERSON {conf:.2f}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        self.get_logger().info(
                            f'✅ PERSON pos=({self.smooth_x:.2f},{self.smooth_y:.2f}) conf={conf:.2f}')

                    elif cls == CLASS_BALL and not ball_detected:
                        ball_detected = True
                        r = min(x2 - x1, y2 - y1) / 2.0

                        norm_x_b = (cx - w / 2.0) / (w / 2.0)
                        screen_ratio_b = (2 * r) / h
                        norm_y_b = (screen_ratio_b - 0.05) / 0.05

                        # EMA 평활화
                        self.smooth_ball_x = self.alpha * norm_x_b + (1 - self.alpha) * self.smooth_ball_x
                        self.smooth_ball_y = self.alpha * norm_y_b + (1 - self.alpha) * self.smooth_ball_y

                        ball_msg = Point()
                        ball_msg.x = self.smooth_ball_x
                        ball_msg.y = self.smooth_ball_y
                        ball_msg.z = conf
                        self.ball_pos_pub.publish(ball_msg)

                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 80, 0), 2)
                        cv2.circle(cv_image, (int(cx), int(cy)), int(r), (255, 80, 0), 2)
                        cv2.putText(cv_image, f'BALL {conf:.2f}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 0), 2)
                        self.get_logger().info(
                            f'⚽ BALL pos=({self.smooth_ball_x:.2f},{self.smooth_ball_y:.2f}) conf={conf:.2f}')

            if not person_detected:
                self.get_logger().debug('⚠️ No person detected in this frame')
            if not ball_detected:
                self.get_logger().debug('⚠️ No ball detected in this frame')

            # 결과 이미지 발행
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.detection_pub.publish(out_msg)
            self.annotated_pub.publish(out_msg)

        except Exception as e:
            tb = traceback.format_exc()
            self.get_logger().error(f'Error processing image: {e}\n{tb}')

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
