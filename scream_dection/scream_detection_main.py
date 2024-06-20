import threading
import sys
import os

# 현재 스크립트의 디렉토리 경로를 얻습니다.
current_dir = os.path.dirname(__file__)

# 모델 파일의 절대 경로를 생성합니다.
model_dir = os.path.join(current_dir, 'mymodel.pth')

# fall_detection.py 파일이 있는 디렉토리를 sys.path에 추가
fall_detection_path = os.path.join(current_dir, '../fall_detection')
sys.path.append(fall_detection_path)

import torch
import torch.nn as nn
from PyQt5 import QtCore, QtWidgets
from demo import MyWindow, MicrophoneRecorder
from fall_detection import fall_detection  # 낙하 감지 함수 가져오기

def scream_detection():
    sampling_rate = 22050  # Hz
    chunk_size = 22050  # samples

    # 모델 정의
    model = nn.Sequential(
        nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(64, 1),
        ),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Dropout2d(p=0.3),
        nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(1, 9),
            stride=4
        ),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Dropout2d(p=0.3),
        nn.Flatten(),
        nn.Linear(64 * 1 * 9, 1),
    )

    # 모델 로드
    try:
        model.load_state_dict(torch.load(model_dir, map_location='cpu'))
        print("모델 로드 성공")
    except FileNotFoundError:
        print(f"모델 파일을 찾을 수 없습니다: {model_dir}")
        return
    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        return

    app = QtWidgets.QApplication(sys.argv)
    myWindow = MyWindow(model=model)
    mic = MicrophoneRecorder()
    mic.signal.connect(myWindow.read_collected)

    # 시간 간격(초) 계산
    interval = int(1000 * chunk_size / sampling_rate)  # ms 단위로 변경
    t = QtCore.QTimer()
    t.timeout.connect(mic.read)
    t.start(interval)  # 계산된 interval 사용

    myWindow.show()
    app.exec_()

if __name__ == "__main__":
    # 멀티스레딩을 사용하여 두 프로그램 병렬 실행
    scream_thread = threading.Thread(target=scream_detection)
    fall_thread = threading.Thread(target=fall_detection)

    scream_thread.start()
    fall_thread.start()

    scream_thread.join()
    fall_thread.join()
