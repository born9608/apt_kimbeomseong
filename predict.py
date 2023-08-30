import argparse
import os
import joblib 
import time
import warnings

import pandas as pd
import numpy as np
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score
from xgboost import XGBClassifier

# 경고 무시
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings(action='ignore', module='xgboost')



def get_user_input():
    while True:
        choice = input("한 행만 예측하시겠습니까?, [y/n]: ").lower()
        if choice in ['y', 'n']:
            return choice
        else:
            print("잘못된 입력입니다. 'y' 또는 'n'만 입력하세요.")

class Model:

    def __init__(self, target, train_model=0):
        # train_model: 학습 여부(0이면 joblib에서 불러온다, 1이면 새로 학습한다)
        # target: 사용할 target
        self.target = target
        self.train_model = train_model
        self.params = {}
        self.model = None

        # 모델 불러오기
        # train_model = 0이면 저장된 모델을 불러오고 로드에 실패하면 학습을 진행한다
        if self.train_model == 0:
            file_path = os.path.join(f'save/apt_{self.target}.joblib')
            if os.path.exists(file_path):
                self.model = joblib.load(file_path)
                self.params = self.model.get_params()
                print('모델 불러오기 성공')
            else:
                print('모델 불러오기 실패')
                self.model, train_time = self.train()
                self.params = self.model.get_params()
                
                # 학습 소요시간 print. 0.1초보다 작으면 ms단위로
                if train_time < 0.1:
                    print(f'모델학습 완료. 소요시간 {train_time:.2f}')
                else:
                    print(f'모델학습 완료. 소요시간 {train_time*1000:.2f}ms')

        # train_model = 1이면 학습을 진행한다
        else:
            self.model, train_time = self.train()
            self.params = self.model.get_params()
            
            # 학습 소요시간 print. 0.1초보다 작으면 ms단위로
            if train_time < 0.1:
                print(f'모델학습 완료. 소요시간 {train_time:.2f}')
            else:
                print(f'모델학습 완료. 소요시간 {train_time*1000:.2f}ms')

        super().__init__()

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
    
    def _split_data(self, data):
        """
        훈련과 평가 데이터를 나눕니다
        Args:
            data: 훈련/평가로 나눌 데이터셋
            config: 입력된 설정
        Returns:
            split을 진행함과 동시에 x, y 분리
        """
        
        # 지정한 target을 제외하고 모두 drop
        target_list = ['target60', 'target70' ,'target80', 'target90']
        target_list.remove(self.target)
        data = data.drop(columns=target_list)

        if self.target in ['target80', 'target70']:
            # target80인 경우 면적당보증금을 log변환 시켜줘야함(이유는 짐작하지 못했으나 성능이 우세함)
            data['log_면적당보증금'] = np.log(data['면적당보증금'])
            data.drop(columns='면적당보증금', inplace=True)        

        # split
        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns = self.target), data[self.target], test_size=0.2, shuffle=False)
        return X_train, X_test, y_train, y_test

    def train(self):
        """
        load_data로 데이터셋을 불러오고 split_data로 x_train,x_test... 로 데이터를 분리한 뒤 타겟에 맞게 학습을 진행합니다
        Arg:
            target: 사용할 target
        Returns:
            model: 학습한 모델
            train_time: 학습 소요 시간
        """

        # target변수 선언 
        target = self.target

        # default 하이퍼파라미터 설정
        if target == 'target60':
            default_params = {'n_estimators': 228, 'max_depth': 10, 'learning_rate': 0.0768583812353364, 
                        'subsample': 0.9662229301474877, 'colsample_bytree': 0.9785005736064512, 
                        'gamma': 0.30686503400500115, 'lambda': 3.522762852015552e-08, 
                        'alpha': 2.2796151107527633e-08, 'min_child_weight': 1}
        
        elif target == 'target70':
            default_params = {'n_estimators': 215, 'max_depth': 10, 'learning_rate': 0.16077919807440758,
                        'subsample': 0.9135532693912264, 'colsample_bytree': 0.7197754959056628,
                        'gamma': 0.32176155905630405, 'lambda': 8.057588898109088e-07,
                        'alpha': 0.0013400305230964816, 'min_child_weight': 1}
        
        elif target == 'target80':
            default_params = {'n_estimators': 225, 'max_depth': 12, 'learning_rate': 0.08434351267106448,
                        'subsample': 0.9603746784124344, 'colsample_bytree': 0.6839618527946499,
                        'gamma': 0.18450648730656335, 'lambda': 0.03916943898020069,
                        'alpha': 0.007893206304307562, 'min_child_weight': 1}
        
        elif target == 'target90':
            default_params = {'n_estimators': 128, 'max_depth': 4, 'learning_rate': 0.02698777567422039,
                        'subsample': 0.6662096314670163, 'colsample_bytree': 0.9217544827819781,
                        'gamma': 0.27038755382214374, 'lambda': 0.0002866567623589203,
                        'alpha': 0.0003088636857809953, 'min_child_weight': 1}

        # 데이터셋 불러오기
        data = self._load_data()

        # 데이터셋 split
        x_train, _, y_train, _ = self._split_data(data)

        # XGBClassifier로 모델 선언 및 학습
        model = XGBClassifier(**default_params)
    
        # 학습 소요시간
        t1 = time.time()
        model.fit(x_train, y_train)
        t2 = time.time()
        train_time = t2-t1

        return model, train_time

    def evaluate(self, model, x_test, y_test):
        """
        !!! 웹에서 쓰이지 않음 !!!
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
        !!! 웹에서 쓰이지 않음 !!!
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

    def _to_frame(self, data, target=False):
        """
        내부사용함수. 입력데이터를 모델에 넣을 수 있도록 pd.DataFrame으로 변환해준다.
        target==True이면 타겟데이터변환, False이면 feature변환 오류나면 None반환  

        args:
            data : 변환할 데이터 (array-like)
            target : 타겟데이터를 변환하는건인지 확인
        returns:
            DataFrame : 데이터프레임으로 변경된 데이터
        """
    
        # 이미 DataFrame이면 그대로 반환
        if isinstance(data, pd.DataFrame):
            return data

        def transform(data):
            # X가 DataFrame이 아니면 DataFrame으로 최종변환
            if not isinstance(data, pd.DataFrame):
                # 리스트나 튜플이면 np.ndarray로 변환
                if isinstance(data, (list|tuple)):
                    data = np.array(data)

                if isinstance(data, np.ndarray):
                    # [data1, data2, ..] 이면 [[data1, data2.. ]]로 변환
                    if data.ndim == 1:
                        data = data.reshape(1,-1)
                    elif data.ndim !=2:
                        print(f'입력데이터의 차원이 이상합니다. {data.ndim}차원이 아닌 1차원 혹은 2차원이 되어야합니다.')
                        return None
                    # DataFrame으로 변환
                    data = pd.DataFrame(data)

                if isinstance(data, pd.Series):
                    data = data.to_frame().T
            return data

        data = transform(data=data)
        if data is None:
            return None

    def predict_db(self, input_data, config):
        """
        다수의 행을 dataframe으로 입력받으면 사용 ---- 휴업 중 읽을 필요 없는 코드
        args:
            input_Data : 입력데이터 (pd.DataFrame or array-like)
            config 
        returns:
            pred : 모델 에측값 (np.ndarray)
        """
        # 데이터프레임을 입력값으로 받음
        # 모델이 사용하는 특성은 6개, 순서는 아래와 같음
        # 시군구, 계약년월, 조정대상지역, 투기과열지구, 광역, 면적당보증금 or log_면적당보증금
        
        # 입력값 특성 순서가 시군구, 계약년월, 조정대상지역, 투기과열지구, 전세가, 전용면적이라고 가정하고 진행함
        # 입력을 DF로 받을 때 xgboost는 학습, 예측 간 특성 이름과 순서가 정확히 일치해야함. 아니면 에러 발생
        if isinstance(input_data, pd.DataFrame):
            
            # 면적당보증금만들기
            input_data['면적당보증금'] = input_data['전세가'] / input_data['전용면적']
            input_data.drop(columns=['전세가', '전용면적'], inplace=True)

            # target70, target80 일 땐 면적당보증금을 log 변환해야함
            if config.target == 'target70' or config.target == 'target80':
                input_data['np_면적당보증금'] = np.log(input_data['면적당보증금'])
            
            # 광역 특성 만들기
            input_data['광역'] = input_data['시군구'].str.split().str[0]

            # 줄이기 위한 광역 키
            REGIONKEY = {'강원특별자치도': '강원', '경기도': '경기', '경상남도': '경남',
                        '경상북도': '경북', '광주광역시': '광주', '대구광역시': '대구', '대전광역시':'대전',
                        '부산광역시': '부산', '서울특별시': '서울', '세종특별자치시': '세종', '울산광역시':'울산',
                        '인천광역시': '인천', '전라남도':'전남', '전라북도':'전북', '제주특별자치도':'제주', '충청남도':'충남', '충청북도':'충북'}
            input_data['광역'] = input_data['광역'].replace(REGIONKEY)
        
            # 특성 재배열
            if config.target == 'target70' or config.target == 'target80':
                input_data = input_data[['시군구', '계약년월', '조정대상지역', '투기과열지구', '광역', 'log_면적당보증금']]
            else:
                input_data = input_data[['시군구', '계약년월', '조정대상지역', '투기과열지구', '광역', '면적당보증금']]

        # 라벨 인코더 불러오기
        le_sigungu = joblib.load('save/sigungu.joblib')
        le_gwangyeok = joblib.load('save/gwangyeok.joblib')
        input_data['시군구'] = le_sigungu.transform(input_data['시군구'])
        input_data['광역'] = le_gwangyeok.transform(input_data['광역'])

        pred = self.model.predict(input_data)
        return pred
    
    def predict(self, input):
        """
        데이터를 받아서 예측함
        Arg:
            input: 입력값

        Returns:
            pred: 예측값
        """

        # target70, target80 모델로 예측할 땐 input의 면적당보증금(deposit)을 로그변환해서 넣어야함
        if self.target in ['target70', 'target80']:
            input[-1] = np.log(input[-1])

        pred = self.model.predict(input)
        return pred
    
def input_data():

    # 입력값 받기
    sigungu = input("시군구를 입력하세요: ")
    contract_time = int(input("계약년월을 입력하세요: "))
    adjust = int(input("조정대상지역 여부를 입력하세요(0,1): "))
    overheat = int(input("투기과열지구 여부를 입력하세요(0,1) "))
    deposit = float(input("면적당보증금을 입력하세요: "))

    # 줄이기 위한 광역 키
    REGIONKEY = {'강원특별자치도': '강원', '경기도': '경기', '경상남도': '경남',
                '경상북도': '경북', '광주광역시': '광주', '대구광역시': '대구', '대전광역시':'대전',
                '부산광역시': '부산', '서울특별시': '서울', '세종특별자치시': '세종', '울산광역시':'울산',
                '인천광역시': '인천', '전라남도':'전남', '전라북도':'전북', '제주특별자치도':'제주', '충청남도':'충남', '충청북도':'충북'}
    
    if sigungu.split()[0] in REGIONKEY:
        gwangyeok = REGIONKEY[sigungu.split()[0]]

    # 라벨 인코더 불러오기
    le_sigungu = joblib.load('save/sigungu.joblib')
    le_gwangyeok = joblib.load('save/gwangyeok.joblib')
    sigungu_encoded = int(le_sigungu.transform([sigungu])[0])
    gwangyeok_encoded = int(le_gwangyeok.transform([gwangyeok])[0])

    input_list_encoded = [sigungu_encoded, contract_time, adjust, overheat, gwangyeok_encoded, deposit]
    input_data_encoded = np.array(input_list_encoded).reshape(1, -1)

    # pred = self.model.predict(input_data_encoded)
    return input_data_encoded

# Main!!! 
if __name__ == '__main__':   
    
    # 입력 받기
    response = get_user_input()

    # 새로 학습해서 예측할지->1, 저장된 모델로 예측할지 -> 0 결정
    train_model = 0
    
    if response == 'y':
        
        # 입력값 받기
        input = input_data()

        rate = 50
        for i in ['target60', 'target70', 'target80', 'target90']:
            model = Model(i, train_model)
            prediction = model.predict(input)
            if prediction == 1:
                if i == 'target60':
                    rate = 60
                    print(f'역전세 위험에서 안전합니다')

                elif i == 'target70':
                    rate = 70
                    print(f'역전세 가능성에 관심을 기울여야 합니다')

                elif i == 'target80':
                    rate = 80
                    print(f'역전세 가능성이 높으니 경계해야 합니다')

                else:
                    rate = 90
                    print(f'역전세 위험이 매우 높습니다')
                break    
        
        if rate is None:
            print(f'역전세 위험에서 매우 안전합니다')
        
        print(f'이 아파트는 예상 전세가율은 {rate}% 이상입니다')