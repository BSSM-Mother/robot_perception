# robot_perception - 비전 기반 인식

**카메라 영상에서 사람을 감지하고 위치를 계산하는 비전 처리 패키지**

## 📌 개요

YOLOv8/YOLOv11 기반 딥러닝 모델로 카메라 영상에서 사람을 감지하고, 감지된 사람의 위치를 ROS2 토픽으로 발행합니다.

## 🎯 주요 기능

- ✅ **실시간 사람 감지**: YOLO 기반 객체 감지
- ✅ **다양한 모델 지원**: ONNX, PyTorch 모델
- ✅ **위치 정보 발행**: 화면 좌표 및 신뢰도
- ✅ **시각화**: 감지 결과 이미지 발행

## 🚀 지원 모델

| 모델 | 파일 | 형식 | 성능 |
|------|------|------|------|
| YOLOv8 Nano | `yolov8n.pt` | PyTorch | 빠름 |
| YOLOv8 Nano ONNX | `yolov8n.onnx` | ONNX | 초고속 |
| YOLOv11 Nano | `yolo11n.pt` | PyTorch | 높은 정확도 |

**모델 위치**: 워크스페이스 루트 디렉토리
- `yolov8n.pt`
- `yolov8n.onnx`
- `yolo11n.pt`

## 📡 토픽 및 인터페이스

### 구독 토픽 (입력)
- **`/camera/image_raw`** (`sensor_msgs/Image`)
  - 카메라에서의 원본 영상 (RGB)

### 발행 토픽 (출력)
- **`person_position`** (`geometry_msgs/Point`)
  - `x`: 화면 중심 기준 수평 오차 (-1.0 ~ 1.0)
  - `y`: 화면 중심 기준 수직 오차 (-1.0 ~ 1.0)
  - `z`: 감지 신뢰도 (0.0 ~ 1.0)

- **`detection_image`** (`sensor_msgs/Image`) [선택]
  - 감지 결과 박스/라벨이 그려진 이미지

### 화면 좌표계

```
화면 (예: 640x480)
┌────────────────┐
│                │
│  (640/2, 240)  │ ← 화면 중심 (0, 0)
│       ●        │
│                │
└────────────────┘

person_position:
- x = (감지된_x - 320) / 320  → -1.0 ~ 1.0 범위
- y = (감지된_y - 240) / 240  → -1.0 ~ 1.0 범위
- z = 신뢰도                  → 0.0 ~ 1.0 범위
```

## ⚙️ 파라미터 설정

기본 설정 (robot_launch의 설정 파일):
```yaml
person_detector:
  ros__parameters:
    model_path: "yolov8n.pt"           # 모델 경로 (상대 또는 절대 경로)
    model_format: "pytorch"            # "pytorch" 또는 "onnx"
    confidence_threshold: 0.5          # 신뢰도 임계값 (0.0 ~ 1.0)
    iou_threshold: 0.45                # NMS IoU 임계값
    camera_topic: "/camera/image_raw"  # 입력 카메라 토픽
    output_topic: "person_position"    # 출력 토픽
    publish_detection_image: true      # 감지 이미지 발행 여부
    target_class: 0                    # YOLO 클래스 ID (0=사람)
```

## 💾 빌드

```bash
colcon build --packages-select robot_perception
```

## 🚀 실행

개별 실행:
```bash
ros2 run robot_perception person_detector
```

또는 런치 파일로 전체 실행:
```bash
ros2 launch robot_launch robot.launch.py
```

## 📊 감지 프로세스

```
카메라 입력 (sensor_msgs/Image)
          │
          ▼
   ┌─────────────┐
   │  이미지 변환 │ (OpenCV 처리)
   └─────────────┘
          │
          ▼
   ┌─────────────┐
   │ YOLO 추론   │ (딥러닝 모델)
   └─────────────┘
          │
          ▼
   ┌──────────────────┐
   │ 감지 결과 필터링  │ (신뢰도 >= threshold)
   └──────────────────┘
          │
          ▼
   ┌──────────────────────┐
   │ 좌표 정규화 변환     │ (-1.0 ~ 1.0 범위)
   │ + 신뢰도 계산        │
   └──────────────────────┘
          │
          ▼
   person_position 발행
```

## 🔧 환경 설정

### Python 의존성
```bash
pip install numpy opencv-python torch torchvision ultralytics onnx onnxruntime
```

또는 requirements.txt 사용:
```bash
pip install -r robot_perception/requirements.txt
```

### NVIDIA GPU 지원 (선택)
CUDA 가속을 사용하려면:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📝 소스 파일

```
src/robot_perception/robot_perception/
├── person_detector.py    # 메인 감지 노드
└── __init__.py
```

## 🎨 신뢰도 임계값 조정 가이드

| 임계값 | 특성 | 상황 |
|--------|------|------|
| 0.3 | 높은 감지율, 오탐지 많음 | 약한 조명, 원거리 |
| 0.5 | 균형잡힘 (기본값) | 대부분의 환경 |
| 0.7 | 높은 정확도, 감지 누락 | 명확한 환경, 근거리 |

## ⚠️ 주의사항

- **모델 파일 확인**: 지정된 모델 파일이 존재하고 읽을 수 있는지 확인
- **GPU 메모리**: 대규모 모델은 GPU 메모리 부족 가능 (Nano 모델 권장)
- **카메라 연결**: camera_ros 패키지가 먼저 실행되어야 함
- **첫 실행**: 모델 초기화에 시간 소요 가능 (1~3초)

## 🔗 연관 패키지

- **입력 출처**: `camera_ros` (카메라 영상)
- **출력 대상**: `robot_control` (제어 로직)
- **의존성**: rclpy, cv_bridge, sensor_msgs, geometry_msgs, torch, ultralytics

## 📊 성능 지표

| 모델 | 프레임레이트 | GPU 메모리 | 정확도 |
|------|-------------|-----------|--------|
| YOLOv8n (PyTorch) | 30 FPS | ~1GB | 중간 |
| YOLOv8n (ONNX) | 45 FPS | ~500MB | 중간 |
| YOLOv11n (PyTorch) | 25 FPS | ~1.5GB | 높음 |

*(Jetson Orin Nano 기준)*

## 🚀 성능 최적화

1. **ONNX 모델 사용**: PyTorch보다 빠름
2. **입력 해상도 감소**: 640x480 → 320x240
3. **배치 처리**: 여러 프레임 한 번에 처리
4. **GPU 가속**: NVIDIA GPU 활용
