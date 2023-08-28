import argparse

import pandas as pd
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
    p.add_argument('--target', type=str, choices=['target60', 'target70', 'target80', 'target90'], default='target80')
    config = p.parse_args()

    return config

def load_data(file_path):
    """
    경로에서 파일을 가져와 인코딩 후 계약년월 순으로 배열합니다
    Args:
        file_path (str): 불러올 csv파일 경로
    Returns:
        pandas 데이터프레임 형태로 데이터를 변환
    """

    # 데이터 불러오기
    data = pd.read_csv(file_path)

    # 인코딩
    le1 = LabelEncoder()
    data['광역'] = le1.fit_transform(data['광역'])
    le2 = LabelEncoder()
    data['시군구'] = le2.fit_transform(data['시군구'])
    
    # 계약년월 순 배열
    data = data.sort_values(['계약년월'])

    return data

def split_data(data, target, test_size=0.2):
    """
    훈련과 평가 데이터를 나눕니다
    Args:
        data: 훈련/평가로 나눌 데이터셋
        target: 사용할 타겟
        test_size: 평가 데이터의 비율
    Returns:
        split을 진행함과 동시에 x, y 분리
    """
    
    # 지정한 target을 제외하고 모두 drop
    target_list = ['target60', 'target70' ,'target80', 'target90']
    target_list.remove(target)
    data = data.drop(columns=target_list)

    # split
    X_train, X_test, y_train, y_test = train_test_split(data.drop(target, axis=1), data[target], test_size=test_size)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, target, n_splits):
    """
    Optuna로 찾은 최적 하이퍼파라미터를 넣은 xgboost로 Timeseriessplit을 통해 학습을 진행합니다
    Args:
        X_train
        y_train 
        target: 사용할 target
        n_splits: timeseriessplit에서 몇번 split할지 결정
    Returns:
        훈련된 모델을 반환
    """

    if target == 'target80':
        best_params = {'n_estimators': 225, 'max_depth': 12, 'learning_rate': 0.08434351267106448,
                      'subsample': 0.9603746784124344, 'colsample_bytree': 0.6839618527946499,
                      'gamma': 0.18450648730656335, 'lambda': 0.03916943898020069,
                      'alpha': 0.007893206304307562, 'min_child_weight': 1}

    elif target == 'target60':
        best_params = {'n_estimators': 228, 'max_depth': 10, 'learning_rate': 0.0768583812353364, 
                      'subsample': 0.9662229301474877, 'colsample_bytree': 0.9785005736064512, 
                      'gamma': 0.30686503400500115, 'lambda': 3.522762852015552e-08, 
                      'alpha': 2.2796151107527633e-08, 'min_child_weight': 1}
    
    elif target == 'target70':
        best_params = {'n_estimators': 215, 'max_depth': 10, 'learning_rate': 0.16077919807440758,
                      'subsample': 0.9135532693912264, 'colsample_bytree': 0.7197754959056628,
                      'gamma': 0.32176155905630405, 'lambda': 8.057588898109088e-07,
                      'alpha': 0.0013400305230964816, 'min_child_weight': 1}
    
    elif target == 'target90':
        best_params = {'n_estimators': 128, 'max_depth': 4, 'learning_rate': 0.02698777567422039,
                      'subsample': 0.6662096314670163, 'colsample_bytree': 0.9217544827819781,
                      'gamma': 0.27038755382214374, 'lambda': 0.0002866567623589203,
                      'alpha': 0.0003088636857809953, 'min_child_weight': 1}
    
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    모델을 평가합니다
    Args:
        model: 평가할 모델
        X_test
        y_test
    Returns:
        모델의 평가 성능을 반환
    """

    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return recall, f1, accuracy, precision

if __name__ == '__main__':
    config = define_argparser()

    data = load_data('data/dataset.csv')

    X_train, X_test, y_train, y_test = split_data(data, config.target, config.test_ratio)

    model = train_model(X_train, y_train, config.target, config.n_splits)

    recall, f1, accuracy, precision = evaluate_model(model, X_test, y_test)

    # config로 받은 값과 평가 성능 출력
    print('Setting', ' --- ' 'Target:', config.target, ' test_ratio:', config.test_ratio, " Timeseries's n_splits: ", config.n_splits)
    print('recall:', recall, ' f1 score:', f1, ' accuracy:', accuracy, ' precision', precision)