# 낙상 및 비명 감지 시스템

이 프로젝트는 OpenCV와 PyTorch를 사용하여 낙상 및 비명을 감지하는 시스템입니다. 웹캠을 통해 낙상을 감지하고, 마이크를 통해 비명을 감지하는 기능을 포함하고 있습니다.

## 기능 설명

1. **낙상 감지 - fall_detection.py**:
   - OpenCV를 사용하여 웹캠 영상을 실시간으로 처리합니다.
   - 사람의 낙상을 감지하고, 감지된 경우 이미지로 저장합니다.
   - 특정 음성을 인식하여 도움 요청을 감지합니다.

2. **비명 감지 - scream_detection.ipynb**:
   - 비명 감지 목적으로 구성된 Jupyter 노트북입니다.
   - 소리 데이터의 처리와 분석을 중심으로 구성되어, 딥러닝을 사용해 비명을 감지하는 기능을 구현하는 것을 목표로 합니다.

   **비명 감지 - demo.py**:
   - 마이크에서 오디오 스트림을 읽어와 MyWindow 클래스로 전송됩니다.
   - 실시간으로 오디오를 분석하고 시각화합니다.
   - 바 차트를 확률에 따라 다르게 표시하며, 분류 결과에 따라 GUI에 이미지를 표시합니다.

   **비명 감지 - scream_fall_detection_main.py**:
   - PyTorch를 사용하여 비명 소리를 감지합니다.
   - 사전 훈련된 모델을 사용하여 마이크로 입력된 소리를 분석합니다.


## 기술적 구현

1. **낙상 감지 - fall_detection.py**
   - OpenCV를 사용하여 웹캠에서 영상을 스트리밍하고, 배경 제거 기법인 MOG2를 이용하여 배경을 제거합니다.
   - 이후, 배경 제거된 이미지에서 contour를 찾아 넘어짐을 감지합니다.
   - 사람의 높이와 너비 비율을 통해 넘어짐을 판단하고, 일정 시간 동안 연속적인 넘어짐이 감지되면 "FALL" 메시지를 출력하고
     해당 프레임을 이미지 파일로 저장합니다.

2. **비명 감지 - scream_detection.ipynb**
   - 소리 데이터를 프레임 단위로 나누어 처리합니다. 
   - 각 프레임에서 Mel 스펙트로그램을 추출하여
     주파수 변화의 패턴을 딥러닝 모델이 학습할 수 있는 형태로 데이터를 변환합니다.
   - 비명이 포함된 오디오 샘플을 사용하여 해당 프레임이 비명을 포함하는지 여부를 레이블로 지정합니다.
   - 데이터를 불러오고 필요한 경우 전처리를 수행합니다.
   - 딥러닝 모델을 정의하는 클래스를 구현합니다.
   - 정의된 모델 클래스를 사용하여 데이터를 학습시킵니다.

   **비명 감지 - demo.py**:
   - PyQt 및 UI를 연동시킵니다.
   - 실시간으로 오디오를 분석하고 적합하고 적합한 형대로 변환합니다.
   - PyTorch를 사용하 딥러닝 모델로 통합시킵니다.
   - 읽어온 데이터를 실시작으로 GUI에 반영, 업데이트합니다.

   **비명 감지 - scream_fall_detection_main.py**
   - PyQt5를 사용하여 GUI를 구현하고, 마이크 입력을 처리합니다.
   - 신경망 모델은 합성곱 신경망(Convolutional Neural Network)으로 구성되어 소리의 스펙트로그램을 분석하고,
     비명 패턴을 학습한 후 감지합니다.
   - 감지된 비명에 대한 경고를 화면에 출력하며, 필요시 추가적인 경고 메커니즘을 구현할 수 있습니다.

   
## 요구 사항
- Python 3.7 이상
- 필요한 라이브러리:

1. **낙상감지**
  - cv2 (OpenCV): 이미지 처리 및 컨투어 검출
  - datetime: 파일 이름으로 현재 시간을 사용하여 이미지 저장
  - time: 웹캠 초기화를 위한 대기 시간 제어

2. **비명 감지**
  - torch: 신경망 모델 관리 및 학습된 가중치 로드
  - PyQt5: GUI 구현 및 이벤트 처리
  - threading: 두 프로그램을 병렬로 실행하기 위한 멀티스레딩 지원


## 실행방법
두 프로그램은 각각 독립적으로 실행되며, 멀티스레딩을 통해 병렬로 실행됩니다.
(프로젝트를 실행하기 전에 웹캠과 마이크가 제대로 연결되어 있는지 확인해야 합니다.)
### 낙상감지
-  웹캠을 통해 사람의 넘어짐을 감지하고 실시간으로 화면에 표시합니다.
-  넘어짐이 감지되면 "FALL" 메시지를 표시하고 해당 시점의 이미지를 저장합니다.
### 비명감지
- 마이크를 통해 주변 소리를 수집하고, 신경망 모델을 사용하여 비명을 감지합니다.
- PyQt5를 사용하여 소리를 시각화하고, 비명 감지 시 사용자에게 경고를 출력합니다.

## 결과화면
### 낙상감지
![image](https://github.com/hyhy-j/opensource_final/assets/141477787/d5feea9c-6d92-49ef-8d3b-a035bef0b843)
### 비명감지
![image](https://github.com/hyhy-j/opensource_final/assets/141477787/46fe5e55-cdc3-4c20-a89c-79d349adeba1)

### 최종 결과화면(낙상+비명감지)
![image](https://github.com/hyhy-j/opensource_final/assets/141477787/8b4022f9-5710-4e48-963d-b8e01ee93158)
![image](https://github.com/hyhy-j/opensource_final/assets/141477787/1688cf9e-89fc-46dc-9f4a-e616ada516e1)


## 팀원 역할분담 :

### 팀장 - 권희재
1. **역할**
 - 프로젝트 리더십 및 조정, 낙상 모니터링 프로그램 개발 및 관리
2. **기여**
- webcam을 이용한 낙상 판별 알고리즘 설계 및 구현
- 낙상 사건 발생 시 데이터 캡처 및 관리 
- 최종 보고서 작성 및 문서화
  
### 팀원1 - 이현정
1. **역할**
 - 딥러닝 모델 학습 및 시스템 통합 관리
2. **기여**
 - 합성곱 신경망을 활용한 음향 패턴 분석 및 모델 최적화 
 - thread를 이용하여 학습된 모델과 fall_detection 프로그램을 통합하여 실행
 - GitHub 리포지토리 관리 및 팀 내 협업 지원
   
### 팀원2 - 이주원
1. **역할**
 - 비명 소리 인식 모델 학습 및 GUI 개발
2. **기여**
 - 합성곱 신경망을 활용한 비명 소리 인식 모델의 학습과 테스트
 - scream_detection 실행 시의 사용자 친화적인 GUI 구현
 - 프로젝트에 대한 Readme.md 작성
