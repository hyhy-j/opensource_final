# 낙상 및 비명 감지 시스템

이 프로젝트는 OpenCV와 PyTorch를 사용하여 낙상 및 비명을 감지하는 시스템입니다. 웹캠을 통해 낙상을 감지하고, 마이크를 통해 비명을 감지하는 기능을 포함하고 있습니다.

## 기능 설명

1. **낙상 감지 (fall_detection.py)**:
   - OpenCV를 사용하여 웹캠 영상을 실시간으로 처리합니다.
   - 사람의 낙상을 감지하고, 감지된 경우 이미지로 저장합니다.
   - 특정 음성을 인식하여 도움 요청을 감지합니다.

2. **비명 감지 (scream_detection_main.py)**:
   - PyTorch를 사용하여 비명 소리를 감지합니다.
   - 사전 훈련된 모델을 사용하여 마이크로 입력된 소리를 분석합니다.

## 기술적 구현

1. **낙상 감지 (fall_detection.py)**
   - OpenCV를 사용하여 웹캠에서 영상을 스트리밍하고, 배경 제거 기법인 MOG2를 이용하여 배경을 제거합니다.
   - 이후, 배경 제거된 이미지에서 contour를 찾아 넘어짐을 감지합니다.
   - 사람의 높이와 너비 비율을 통해 넘어짐을 판단하고, 일정 시간 동안 연속적인 넘어짐이 감지되면 "FALL" 메시지를 출력하고
     해당 프레임을 이미지 파일로 저장합니다.

2. **비명 감지 (scream_detection_main.py)**
   - PyQt5를 사용하여 GUI를 구현하고, 마이크 입력을 처리합니다.
   - 신경망 모델은 합성곱 신경망(Convolutional Neural Network)으로 구성되어 소리의 스펙트로그램을 분석하고,
     비명 패턴을 학습한 후 감지합니다.
   - 감지된 비명에 대한 경고를 화면에 출력하며, 필요시 추가적인 경고 메커니즘을 구현할 수 있습니다.
   
## 요구 사항
- Python 3.7 이상
- 필요한 라이브러리:
  **낙상감지**
   cv2 (OpenCV): 이미지 처리 및 컨투어 검출
   datetime: 파일 이름으로 현재 시간을 사용하여 이미지 저장
   time: 웹캠 초기화를 위한 대기 시간 제어

  **비명 감지**
   torch: 신경망 모델 관리 및 학습된 가중치 로드
   PyQt5: GUI 구현 및 이벤트 처리
   threading: 두 프로그램을 병렬로 실행하기 위한 멀티스레딩 지원


## 실행방법
두 프로그램은 각각 독립적으로 실행되며, 멀티스레딩을 통해 병렬로 실행됩니다.
(프로젝트를 실행하기 전에 웹캠과 마이크가 제대로 연결되어 있는지 확인해야 합니다.)
### 낙상감지
-  웹캠을 통해 사람의 넘어짐을 감지하고 실시간으로 화면에 표시합니다.
-  넘어짐이 감지되면 "FALL" 메시지를 표시하고 해당 시점의 이미지를 저장합니다.
### 비명감지
- 마이크를 통해 주변 소리를 수집하고, 신경망 모델을 사용하여 비명을 감지합니다.
- PyQt5를 사용하여 소리를 시각화하고, 비명 감지 시 사용자에게 경고를 출력합니다.


