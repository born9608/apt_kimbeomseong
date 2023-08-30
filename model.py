import argparse
import os
import joblib 
import time

import pandas as pd
import numpy as np
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score

from xgboost import XGBClassifier

def define_argparser():
    """
    config를 생성합니다
        target(str): 'target60', 'target70', 'target80', 'target90' 중 하나를 입력받아 저장합니다
        n_splits(int): train 시 TimeSeriesSplit의 fold 횟수 결정
        test_ratio(float): test 데이터 비율
    """
    
    p = argparse.ArgumentParser()

    p.add_argument('--n_splits', type=int, default=10)
    p.add_argument('--test_ratio', type=float, default=0.2)
    p.add_argument('--target', type=str, choices=['target60', 'target70', 'target80', 'target90'], default='target70')
    config = p.parse_args()

    return config

class Model:

    def __init__(self, config):
        self.params = {}
        self.model = None
        self.config = config

        # 모델 불러오기
        # 성공시 미리 저장한 모델.joblib을 불러와 예측 진행
        # 실패시 하이퍼파라미터 기본 값으로 학습 진행

        file_path = os.path.join(f'save/apt_{self.config.target}.joblib')
        if os.path.exists(file_path):
            self.model = joblib.load(file_path)
            self.params = self.model.get_params()
            print('모델 불러오기 성공')
        else:
            print('모델 불러오기 실패')
            self.model, train_time = self.train(self.params, self.config)
            self.params = self.model.get_params()
            
            # 학습 소요시간 print. 0.1초보다 작으면 ms단위로
            if train_time < 0.1:
                print(f'모델학습 완료. 소요시간 {train_time:.2f}')
            else:
                print(f'모델학습 완료. 소요시간 {train_time*1000:.2f}ms')

    def _load_data(self):
        """
        경로에서 파일을 가져와 인코딩 후 계약년월 순으로 배열합니다
        Args:
            file_path (str): 불러올 csv파일 경로
        Returns:
            pandas 데이터프레임 형태로 데이터를 변환
        """

        # 데이터 불러오기
        data_path = 'data/dataset.csv'
        data = pd.read_csv(data_path)

        # 인코딩
        le1 = LabelEncoder()
        data['광역'] = le1.fit_transform(data['광역'])
        le2 = LabelEncoder()
        data['시군구'] = le2.fit_transform(data['시군구'])
        
        # 계약년월 순 배열(시계열 데이터이기 때문)
        data = data.sort_values(['계약년월'])

        return data 
    
    def _split_data(self, data, config):
        """
        훈련과 평가 데이터를 나눕니다
        Args:
            data: 훈련/평가로 나눌 데이터셋
            config: 입력된 설정
        Returns:
            split을 진행함과 동시에 x, y 분리
        """
        # config에 저장된 target들 불러오고 변수선언
        target = config.target

        # 지정한 target을 제외하고 모두 drop
        target_list = ['target60', 'target70' ,'target80', 'target90']
        target_list.remove(target)
        data = data.drop(columns=target_list)

        if target in ['target80', 'target70']:
            # target80인 경우 면적당보증금을 log변환 시켜줘야함(이유는 짐작하지 못했으나 성능이 우세함)
            data['log_면적당보증금'] = np.log(data['면적당보증금'])
            data.drop(columns='면적당보증금', inplace=True)        

        # split
        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns = target), data[target], test_size=config.test_ratio, shuffle=False)

        return X_train, X_test, y_train, y_test

    def train(self, params, config):
        """
        load_data로 데이터셋을 불러오고 split_data로 x_train,x_test... 로 데이터를 분리한 뒤 타겟에 맞게 학습을 진행합니다
        Arg:
            config: 입력된 설정
        Returns:
            model: 학습한 모델
            train_time: 학습 소요 시간
        """

        # 데이터셋 불러오기
        data = self._load_data()
        
        # 데이터셋 split
        x_train, _, y_train, _ = self._split_data(data, config)

        # XGBClassifier로 모델 선언 및 학습
        model = XGBClassifier(**params)
    
        # 학습 소요시간
        t1 = time.time()
        model.fit(x_train, y_train)
        t2 = time.time()
        train_time = t2-t1

        return model, train_time

    def evaluate(self, model, x_test, y_test):
        """
        모델을 평가합니다
        Args:
            model: 평가할 모델
            X_test
            y_test
        Returns:
            recall
            f1
            accuracy
            precision
        """

        y_pred = model.predict(x_test)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        return recall, f1, accuracy, precision  
    
    def trainer(self, config):
        """
        앞서 정의된 4가지 사용자 정의함수를 이용해 학습을 진행하고 모델 성능을 평가함
        Args:
            config: 입력된 설정
        Returns:
        """

        # 데이터셋 불러오기
        data = self._load_data()
        
        # 데이터셋 split
        _, x_test, _, y_test = self._split_data(data, config)

        # __init__에서 모델 불러오기에 성공한 경우 학습 없이 성능평가 진행함
        if self.model != None:
            # 모델 성능 추출
            recall, f1, accuracy, precision = self.evaluate(self.model, x_test, y_test)
            train_time = None
            return recall, f1, accuracy, precision, train_time

        # 모델 불러오기에 실패한 경우 학습 진행 후 성능 평가
        else:
            model, train_time = self.train(self.params, config)
            recall, f1, accuracy, precision = self.evaluate(model, x_test, y_test)
            return recall, f1, accuracy, precision, train_time

# Main!!! 
if __name__ == '__main__':   
    config = define_argparser()
    model = Model(config)
    recall, f1, accuracy, precision, train_time = model.trainer(config)
    
    # config설정에 따른 모델 평가 성능 출력
    print('Setting', ' --- ' 'Target:', config.target, ' test_ratio:', config.test_ratio, " Timeseries's n_splits: ", config.n_splits)
    print('recall:', recall, ' f1 score:', f1, ' accuracy:', accuracy, ' precision', precision, ' train_time', train_time)        