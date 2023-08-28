# apt_kimbeomseong
아파트 전세가율 이진분류 모델입니다

### 데이터셋
data폴더에 두가지 데이터가 존재합니다

> region_inter.csv: 팀장님이 배포한 지역 가격 변화율 보간 데이터(noise.csv)입니다

> dataset.csv: preprocessing.ipynb를 통해 전처리된 데이터입니다. 모델 학습 및 평가에 이 데이터셋을 사용합니다

---

### 파일구성
1. apt_EDA.ipynb: region_inter.csv를 기반으로 EDA 및 시각화한 ipynb입니다
2. util.py: 조정대상지역, 투기과열지구 특성을 만드는 함수가 저장된 파일입니다. preprocessing.ipynb에서 사용됩니다
3. preprocessing.ipynb: region_inter.csv를 전처리해 dataset.csv로 저장합니다
4. eval_record.ipynb: optuna로 찾은 최적하이퍼파라미터와 각 모델의 성능 평가 결과가 담긴 ipynb이며 joblib으로 인코더와 모델을 저장할 때도 쓰였습니다. 시간이 꽤 오래 걸리므로 재실행은 권장하지 않습니다.
5. main.py: CLI로 학습 및 성능 평가 시 사용됩니다. 유지 보수용입니다. argparse로 받는 변수는 target, test_ratio, n_splits입니다.

CLI input 예시

  ```
  python main.py --target target60 --test_ratio 0.2 --n_splits 10
  ```
> n_splits는 TimeSeriesSplit에서 몇번 쪼갤지를 나타내는데 k-fold의 k와 거의 동일합니다

> target은 target60, target70, target80, target90 으로 4가지 입니다
   
6. feature_importance.ipynb: 모델 별 특성 중요도 확인하려고 만든 ipynb인데 혼자 확인하려고 쓰던거라 열람하실 필요는 없습니다

---
### save
1. eval_record.ipynb에서 쓰인 인코더와 최적 하이퍼파라미터 모델을 save 폴더에 저장했습니다
2. targetN.joblib과 sigungu.joblib, gwangyeok.joblib이 저장돼있습니다
3. sigungu(시군구), gwangyeok(광역) 특성은 dataset.csv 기반으로 인코딩됐습니다. 4개 모델 모두 동일한 인코더를 사용합니다.
---
### image 및 pdf
image 폴더는 eval_record.ipynb에 출력된 최적 모델 성능을 이미지로 저장한 폴더입니다

pdf는 ppt자료입니다
