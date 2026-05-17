# 🎮 LoL Win Prediction

리그 오브 레전드(League of Legends) 경기 데이터를 기반으로\
**게임 승패를 예측하는 머신러닝 프로젝트**입니다.

졸업 과제로 진행되었으며, 실제 게임 데이터를 활용하여\
팀의 승리 확률을 분석하고 예측 모델을 구축하는 것을 목표로 합니다.

1. 프로젝트 폴더로 이동
cd c:\Users\User\Desktop\a\lol_win_predict

2. 가상환경 생성
py -m venv venv

3. 가상환경 활성화
.\venv\Scripts\Activate.ps1

4. 패키지 설치
pip install -r requirements.txt

5. 서버 실행
python manage.py runserver

------------------------------------------------------------------------

## 📌 Overview

본 프로젝트는 LoL 경기에서 수집된 다양한 게임 지표(골드, 킬, 오브젝트
등)를 활용하여\
특정 시점에서의 **승리/패배 여부를 예측**합니다.

프로젝트는 다음 단계로 구성됩니다.

-   데이터 전처리
-   특징(Feature) 엔지니어링
-   머신러닝 모델 학습
-   모델 성능 평가
-   예측 결과 분석

------------------------------------------------------------------------

## 🚀 Features

-   📊 경기 데이터 기반 승패 예측
-   ⚙️ 다양한 피처를 활용한 모델 학습
-   📈 모델 성능 평가 (Accuracy, Precision, Recall 등)
-   🧠 머신러닝 알고리즘 적용 (예: Logistic Regression, Random Forest
    등)
-   🔍 데이터 분석 및 시각화

------------------------------------------------------------------------

## 📊 Dataset

LoL 경기 데이터를 기반으로 구성되었습니다.

주요 Feature 예시:

-   Gold Difference (골드 차이)
-   Kills (킬 수)
-   Towers Destroyed (타워 파괴 수)
-   Dragon / Baron Control (드래곤 및 바론 확보)

데이터는 프로젝트 목적에 맞게 전처리 및 가공되었습니다.

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Matplotlib
-   Seaborn

------------------------------------------------------------------------

## ⚙️ Installation

``` bash
git clone https://github.com/htaey/lol_win_predict.git
cd lol_win_predict
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Usage

### 모델 학습

``` bash
python src/train.py
```

### 예측 실행

``` bash
python src/predict.py
```

------------------------------------------------------------------------

## 📈 Results

예시 결과:

-   Model Accuracy: **72.8%**
-   주요 인사이트
    -   골드 차이가 승패에 가장 큰 영향을 미침
    -   오브젝트 확보가 승률에 유의미한 영향을 줌

-   2025.11. 9. T1 vs KT 승리 예측 결과 5경기 중 4경기의 예측이 적중하였음

------------------------------------------------------------------------

## 🎯 Future Work

-   딥러닝 모델 적용 (LSTM, Transformer 등)
-   실시간 경기 데이터 기반 예측 시스템
-   웹 기반 시각화 대시보드 개발

# 실제 페이지
<img width="779" height="923" alt="image" src="https://github.com/user-attachments/assets/9bc5c9ca-bc77-40d3-a253-6378486b7584" />

