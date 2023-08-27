def help():
    print(
        """
        두 함수 모두 noise 데이터셋을 기준으로 합니다.
        단 noise 데이터셋 형태로 계약년월, 시군구, 시 특성이 있다면 어떤 데이터에도 적용할 수 있습니다
        ---------------------------------
        make_overheat
        1. 투기과열지구 특성을 추가합니다
        2. '계약년월', '시' 특성이 필수적으로 요구됩니다

        make_adjust
        1. 조정대상지역 특성을 추가합니다
        2.  '계약년월', '시', '시군구' 특성을 필수적으로 요구합니다 
        ---------------------------------
        
        공통 주의사항
        1. 두 함수 모두 말미에 모든 행의 결측치를 0으로 대체합니다. 필요에 따라 말미의 fillna함수를 수정하시기 바랍니다
        2. 계약년월의 기간은 201908~202307입니다, 기간이 다른 경우 문제가 발생할 수 있습니다. 코드 까보면 이해할 수 있지만 도움이 필요하면 바로 말씀해주세요.
        """
    )


def make_overheat(df_test):

    # 투기과열지구 특성을 추가합니다
    # noise 데이터셋 기준으로 만들었습니다
    # '계약년월'과 '시' 특성이 필수로 있어야 합니다

    # 주의사항
    # 데이터셋에 결측치가 없어야 합니다
    # 있을 경우 함수 최하단에 fillna 함수를 수정하세요


    # 계약년월을 순서대로 정렬해 리스트에 삽입
    list_time = sorted(df_test['계약년월'].unique().tolist())

    start_time = list_time[0] # 201908
    hot_1st = list_time[25] # 202109
    hot_2nd = list_time[35] # 202207
    hot_3rd = list_time[38] # 202210
    hot_4th = list_time[39] # 202211
    hot_5th = list_time[41] # 202301

    # 서울
    seoul_gu = df_test[df_test['시'].str.split().str[0] == '서울특별시'].시.unique().tolist()
    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < hot_5th) & (df_test['시'].isin(seoul_gu))
    df_test.loc[condition, '투기과열지구'] = 1
    condition = (df_test['계약년월'] >= hot_5th) & (df_test['시'].isin(['서울특별시 서초구', '서울특별시 강남구', '서울특별시 송파구', '서울특별시 용산구']))
    df_test.loc[condition, '투기과열지구'] = 1

    # 경기도 과천시
    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < hot_5th) & (df_test['시'] == '경기도 과천시')
    df_test.loc[condition, '투기과열지구'] = 1

    # 경기도 성남분당구 / 경기도 광명시, 하남시
    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < hot_5th) & (df_test['시'].isin(['경기도 성남분당구', '경기도 광명시', '경기도 하남시']))
    df_test.loc[condition, '투기과열지구'] = 1

    # 대구광역시 수성구
    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < hot_2nd) & (df_test['시'] == '대구광역시 수성구')
    df_test.loc[condition, '투기과열지구'] = 1

    # 경기도 수원시
    suwon_gu = ['경기도 수원장안구', '경기도 수원권선구', '경기도 수원팔달구', '경기도 수원영통구']
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_4th) & (df_test['시'].isin(suwon_gu))
    df_test.loc[condition, '투기과열지구'] = 1

    # 성남시 수정구
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_5th) & (df_test['시'] == '경기도 성남수정구')
    df_test.loc[condition, '투기과열지구'] = 1

    # 경기도 안양시
    anyang_gu = ['경기도 안양만안구', '경기도 안양동안구']
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_4th) & (df_test['시'].isin(anyang_gu))
    df_test.loc[condition, '투기과열지구'] = 1

    # 안산시 단원구
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_4th) & (df_test['시'] == '경기도 안산단원구')
    df_test.loc[condition, '투기과열지구'] = 1

    # 경기도 구리시
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_4th) & (df_test['시'] == '경기도 구리시')
    df_test.loc[condition, '투기과열지구'] = 1

    # 인천광역시 연수구, 남동구, 서구
    ic_gu = ['인천광역시 연수구', '인천광역시 남동구', '인천광역시 서구']
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_3rd) & (df_test['시'].isin(ic_gu))
    df_test.loc[condition, '투기과열지구'] = 1

    # 대전광역시 동구, 중구, 서구, 유성구
    dj_gu = ['대전광역시 동구', '대전광역시 중구', '대전광역시 서구', '대전광역시 유성구']
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_2nd) & (df_test['시'].isin(dj_gu))
    df_test.loc[condition, '투기과열지구']

    # 경기도 군포시
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_4th) & (df_test['시'] == '경기도 군포시')
    df_test.loc[condition, '투기과열지구'] = 1

    # 경기도 의왕시
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_4th) & (df_test['시'] == '경기도 의왕시')
    df_test.loc[condition, '투기과열지구'] = 1

    # 용인 시흥구, 수지구
    ls = ['경기도 용인기흥구', '경기도 용인수지구']
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_4th) & (df_test['시'].isin(ls))
    df_test.loc[condition, '투기과열지구'] = 1
    
    # 결측 없애기
    df_test.fillna(0, inplace=True)
    # 정수로 변환
    df_test = df_test.astype({'투기과열지구':'int'})

    return df_test


def make_adjust(df_test):
    # 조정대상지역 특성을 추가합니다.
    # noise 데이터셋을 기준으로 했습니다
    # 계약년월, 시, 시군구을 필수 특성으로 가집니다. 이 세개의 특성만 있으면 어떤 데이터에서든 정상 작동합니다

    # 주의사항
    # 결측치가 없어야 하며 있을 경우 함수 하단의 fillna 함수를 수정하시기 바랍니다

    list_time = sorted(df_test['계약년월'].unique().tolist())
    start_time = list_time[0] # 201908
    hot_1st = list_time[25] # 202109
    hot_2nd = list_time[35] # 202207
    hot_3rd = list_time[38] # 202210
    hot_4th = list_time[39] # 202211
    hot_5th = list_time[41] # 202301

    # 서울
    seoul_gu = df_test[df_test['시'].str.split().str[0] == '서울특별시'].시.unique().tolist()
    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < hot_5th) & (df_test['시'].isin(seoul_gu))
    df_test.loc[condition, '조정대상지역'] = 1
    condition = (df_test['계약년월'] >= hot_5th) & (df_test['시'].isin(['서울특별시 서초구', '서울특별시 강남구', '서울특별시 송파구', '서울특별시 용산구']))
    df_test.loc[condition, '조정대상지역'] = 1

    # 경기도 과천시
    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < hot_5th) & (df_test['시'] == '경기도 과천시')
    df_test.loc[condition, '조정대상지역'] = 1

    # 경기도 성남시 분당구, 수정구 / 경기도 광명시, 하남시
    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < hot_5th) & (df_test['시'].isin(['경기도 성남분당구', '경기도 성남수정구', '경기도 광명시', '경기도 하남시']))
    df_test.loc[condition, '조정대상지역'] = 1

    # 경기도 성남시 중원구
    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < hot_4th) & (df_test['시'] == '경기도 성남중원구')
    df_test.loc[condition, '조정대상지역'] = 1

    # 화성시 전지역 지정
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_4th) & (df_test['시'] == '경기도 화성시')
    df_test.loc[condition, '조정대상지역'] = 1

    # 고양시 전지역
    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < 201911) & (df_test['시'].isin(['경기도 고양일산서구', '경기도 고양일산동구', '경기도 고양덕양구']))
    df_test.loc[condition, '조정대상지역'] = 1

    # 고양시 재지정
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_4th) & (df_test['시'].isin(['경기도 고양일산서구', '경기도 고양일산동구', '경기도 고양덕양구']))
    df_test.loc[condition, '조정대상지역'] = 1

    # 남양주시
    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < 201911) & (df_test['시'] == '경기도 남양주시')
    df_test.loc[condition, '조정대상지역'] = 1

    # 남양주시 별내동, 다산동
    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < hot_4th) & (df_test['시군구'].isin(['경기도 남양주시 별내동', '경기도 남양주시 다산동']))
    df_test.loc[condition, '조정대상지역'] = 1

    # 남양주시 재지정
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_4th) & (df_test['시'] == '경기도 남양주시')
    df_test.loc[condition, '조정대상지역'] = 1

    # 경기도 구리시
    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < hot_4th) & (df_test['시'] == '경기도 구리시')
    df_test.loc[condition, '조정대상지역'] = 1

    # 경기도 용인시
    ls = ['경기도 용인수지구', '경기도 용인기흥구']
    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < hot_4th) & (df_test['시'].isin(ls))
    df_test.loc[condition, '조정대상지역'] = 1

    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_4th) & (df_test['시'] == '경기도 용인처인구')
    df_test.loc[condition, '조정대상지역'] = 1

    ls = ['경기도 용인처인구 포곡읍 둔전리', '경기도 용인처인구 모현읍 일산리']
    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < hot_5th) & (df_test['시군구'].isin(ls))
    df_test.loc[condition, '조정대상지역'] = 0

    # 경기도 수원시
    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < hot_4th) & (df_test['시'] == '경기도 수원팔달구')
    df_test.loc[condition, '조정대상지역'] = 1

    ls = ['경기도 수원영통구 이의동', '경기도 수원영통구 원천동', '경기도 수원영통구 하동', '경기도 수원영통구 매탄동', '경기도 수원팔달구 우만동', '경기도 수원장안구 연무동']
    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < hot_4th) & (df_test['시군구'].isin(ls))
    df_test.loc[condition, '조정대상지역'] = 1

    ls = ['경기도 수원영통구', '경기도 수원권선구', '경기도 수원장안구']
    condition = (df_test['계약년월'] >= 202003) & (df_test['계약년월'] < hot_4th) & (df_test['시'].isin(ls))
    df_test.loc[condition, '조정대상지역'] = 1

    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < hot_4th) & (df_test['시'] == '경기도 수원팔달구')
    df_test.loc[condition, '조정대상지역'] = 1

    # 경기도 안양시
    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < hot_4th) & (df_test['시'] == '경기도 안양동안구')
    df_test.loc[condition, '조정대상지역'] = 1

    condition = (df_test['계약년월'] >= 202003) & (df_test['계약년월'] < hot_4th) & (df_test['시'] == '경기도 안양만안구')
    df_test.loc[condition, '조정대상지역'] = 1

    # 경기도 의왕시
    condition = (df_test['계약년월'] >= 202003) & (df_test['계약년월'] < hot_4th) & (df_test['시'] == '경기도 의왕시')
    df_test.loc[condition, '조정대상지역'] = 1

    # 경기도 군포시, 부천시, 시흥시, 오산시, 의정부시, 광주시
    ls = ['경기도 군포시', '경기도 부천시', '경기도 시흥시', '경기도 오산시', '경기도 의정부시', '경기도 광주시']
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_4th) & (df_test['시'].isin(ls))
    df_test.loc[condition, '조정대상지역'] = 1

    # 경기도 광주시 부분 지정해제
    ls = ['경기도 광주시 초월읍 도평리', '경기도 광주시 초월읍 산이리', '경기도 광주시 초월읍 쌍동리', '경기도 광주시 곤지암읍 곤지암리', '경기도 광주시 곤지암읍 삼리', '경기도 광주시 도척면 진우리']
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_4th) & (df_test['시군구'].isin(ls))
    df_test.loc[condition, '조정대상지역'] = 0

    # 경기도 평택시
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_4th) & (df_test['시'] == '경기도 평택시')
    df_test.loc[condition, '조정대상지역'] = 1

    # 경기도 안산시 
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_4th) & (df_test['시'] == '경기도 안산시')
    df_test.loc[condition, '조정대상지역'] = 1

    # 경기도 안성시
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_3rd) & (df_test['시'] == '경기도 안성시')
    df_test.loc[condition, '조정대상지역'] = 1

    # 경기도 양주시
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_3rd) & (df_test['시'] == '경기도 양주시')
    df_test.loc[condition, '조정대상지역'] = 1

    ls = ['경기도 양주시 백석읍', '경기도 양주시 광적면']
    pattern = '|'.join(ls)
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < hot_3rd) & (df_test['시군구'].str.contains(pattern))
    df_test.loc[condition, '조정대상지역'] = 0

    # 경기도 김포시
    condition = (df_test['계약년월'] >= 202012) & (df_test['계약년월'] < hot_4th) & (df_test['시'] == '경기도 김포시')
    df_test.loc[condition, '조정대상지역'] = 1

    condition = (df_test['계약년월'] >= 202012) & (df_test['계약년월'] < hot_4th) & (df_test['시군구'].str.contains('경기도 김포시 통진읍'))
    df_test.loc[condition, '조정대상지역'] = 0

    # 경기도 파주시
    condition = (df_test['계약년월'] >= 202012) & (df_test['계약년월'] < hot_3rd) & (df_test['시'] == '경기도 파주시')
    df_test.loc[condition, '조정대상지역'] = 1

    ls = ['경기도 파주시 문산읍', '경기도 파주시 조리읍']
    pattern = '|'.join(ls)
    condition = (df_test['계약년월'] >= 202012) & (df_test['계약년월'] < hot_3rd) & (df_test['시군구'].str.contains(pattern))
    df_test.loc[condition, '조정대상지역'] = 0

    # 경기도 동두천시
    condition = (df_test['계약년월'] >= 202109) & (df_test['계약년월'] < hot_3rd) & (df_test['시'] == '경기도 동두천시')
    df_test.loc[condition, '조정대상지역'] = 1

    # 인천광역시
    incheon_gu = df_test[df_test['시'].str.split().str[0] == '인천광역시'].시.unique().tolist()
    incheon_gu.remove('인천광역시 강화군')
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_4th) & (df_test['시'].isin(incheon_gu))
    df_test.loc[condition, '조정대상지역'] = 1

    # 부산광역시 남구-연제구
    ls = ['부산광역시 남구', '부산광역시 연제구']
    condition = (df_test['계약년월'] >= 202012) & (df_test['계약년월'] < hot_3rd) & (df_test['시'].isin(ls))
    df_test.loc[condition, '조정대상지역'] = 1

    # 부산광역시 해운대구, 동래구, 수영구
    ls = ['부산광역시 해운대구', '부산광역시 동래구', '부산광역시 수영구']
    condition = (df_test['계약년월'] >= start_time) & (df_test['계약년월'] < 201911) & (df_test['시'].isin(ls))
    df_test.loc[condition, '조정대상지역'] = 1

    # 재지정
    condition = (df_test['계약년월'] >= 202012) & (df_test['계약년월'] < hot_3rd) & (df_test['시'].isin(ls))
    df_test.loc[condition, '조정대상지역'] = 1

    # 부산광역시 부산진구
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < hot_3rd) & (df_test['시'] == '부산광역시 부산진구')
    df_test.loc[condition, '조정대상지역'] = 1

    # 부산광역시 서구, 동구, 영도구, 금정구, 북구, 강서구, 사상구, 사하구
    ls = ['부산광역시 서구', '부산광역시 동구', '부산광역시 영도구', '부산광역시 금정구', 
                    '부산광역시 북구', '부산광역시 강서구', '부산광역시 사상구', '부산광역시 사하구']
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < hot_3rd) & (df_test['시'].isin(ls))
    df_test.loc[condition, '조정대상지역'] = 1

    # 대구광역시 수성구
    condition = (df_test['계약년월'] >= 202012) & (df_test['계약년월'] < hot_3rd) & (df_test['시'] == '대구광역시 수성구')
    df_test.loc[condition, '조정대상지역'] = 1

    # 대구광역시 중구, 동구, 서구, 남구, 북구, 달서구, 달성군
    ls = ['대구광역시 중구', '대구광역시 동구', '대구광역시 서구', '대구광역시 남구', 
                    '대구광역시 북구', '대구광역시 달서구', '대구광역시 달성군']
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < hot_3rd) & (df_test['시'].isin(ls))
    df_test.loc[condition, '조정대상지역'] = 1

    # 대구광역시 달성군 일부 지역 제외
    ls = ['대구광역시 달성군 구지면' '대구광역시 달성군 논공읍', '대구광역시 달성군 옥포읍', '대구광역시 달성군 유가읍', '대구광역시 달성군 현풍읍']
    pattern = '|'.join(ls)
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < hot_3rd) & (df_test['시군구'].str.contains(pattern))
    df_test.loc[condition, '조정대상지역'] = 0

    # 광주광역시 동구, 서구, 남구, 북구, 광산구
    ls = ['광주광역시 동구', '광주광역시 서구', '광주광역시 남구', '광주광역시 북구', 
                    '광주광역시 광산구']
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < hot_3rd) & (df_test['시'].isin(ls))
    df_test.loc[condition, '조정대상지역'] = 1

    # 대전광역시 전 지역
    ls = ['대전광역시 동구', '대전광역시 중구', '대전광역시 서구', '대전광역시 유성구', '대전광역시 대덕구']
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_3rd) & (df_test['시'].isin(ls))
    df_test.loc[condition, '조정대상지역'] = 1

    # 울산광역시 중구, 남구
    ls = ['울산광역시 중구', '울산광역시 남구']
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < hot_3rd) & (df_test['시'].isin(ls))
    df_test.loc[condition, '조정대상지역'] = 1

    # 충청북도 청주시(동 지역, 오창 오송읍만 지정)
    ls = ['충청북도 청주청원구', '충청북도 청주흥덕구', '충청북도 청주상당구', '충청북도 청주서원구']
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_3rd) & (df_test['시'].isin(ls))
    df_test.loc[condition, '조정대상지역'] = 1

    # 청주시 제외 지역 다시 0으로 변환
    ls = ['충청북도 청주서원구 남이면 가마리', '충청북도 청주서원구 남이면 척북리', '충청북도 청주청원구 내수읍 내수리', '충청북도 청주청원구 내수읍 도원리',
        '충청북도 청주청원구 내수읍 은곡리', '충청북도 청주흥덕구 강내면 월곡리', '충청북도 청주흥덕구 강내면 탑연리', '충청북도 청주흥덕구 옥산면 가락리',
        '충청북도 청주흥덕구 옥산면 오산리' ]
    condition = (df_test['계약년월'] >= 202007) & (df_test['계약년월'] < hot_3rd) & (df_test['시군구'].isin(ls))
    df_test.loc[condition, '조정대상지역'] = 0

    # 충청남도 천안 동남구, 천안 서북구
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < hot_3rd) & (df_test['시'].str.contains('충청남도 천안'))
    df_test.loc[condition, '조정대상지역'] = 1

    # 천안 제외지역 0으로 변환
    ls = ['충청남도 천안동남구 목천읍', '충청남도 천안동남구 병천면', '충청남도 천안동남구 북면', '충청남도 천안서북구 성환읍', '충청남도 천안서북구 성거읍',
        '충청남도 천안서북구 직산읍', '충청남도 천안서북구 입장면']
    pattern = '|'.join(ls)
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < hot_3rd) & (df_test['시군구'].str.contains(pattern))
    df_test.loc[condition, '조정대상지역'] = 0

    # 충청남도 논산시
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < hot_3rd) & (df_test['시'] == '충청남도 논산시')
    df_test.loc[condition, '조정대상지역'] = 1

    # 논산 제외지역 0으로 변환
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < hot_3rd) & (df_test['시군구'] == '충청남도 논산시 연무읍 안심리')
    df_test.loc[condition, '조정대상지역'] = 0

    # 충청남도 공주시
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < hot_3rd) & (df_test['시'] == '충청남도 공주시')
    df_test.loc[condition, '조정대상지역'] = 1

    # 전라북도 전주시 완산구, 덕진구
    ls = ['전라북도 전주완산구', '전라북도 전주덕진구']
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < hot_3rd) & (df_test['시'].isin(ls))
    df_test.loc[condition, '조정대상지역'] = 1

    # 전라남도 여수시
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < 202207) & (df_test['시'] == '전라남도 여수시')
    df_test.loc[condition, '조정대상지역'] = 1

    # 전남 여수 제외 지역 0으로 변환
    ls = ['전라남도 여수시 돌산읍', '전라남도 여수시 율촌면']
    pattern = '|'.join(ls)
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < 202207) & (df_test['시군구'].str.contains(pattern))
    df_test.loc[condition, '조정대상지역'] = 0

    # 전라남도 순천시
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < 202207) & (df_test['시'] == '전라남도 순천시')
    df_test.loc[condition, '조정대상지역'] = 1

    # 전라남도 광양시
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < 202207) & (df_test['시'] == '전라남도 광양시')
    df_test.loc[condition, '조정대상지역'] = 1

    # 경상북도 포항시 남구
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < hot_3rd) & (df_test['시'] == '경상북도 포항남구')
    df_test.loc[condition, '조정대상지역'] = 1

    # 포항시 남구 연일읍, 오천읍 0으로 변환
    ls = ['경상북도 포항남구 연일읍', '경상북도 포항남구 오천읍']
    pattern = '|'.join(ls)
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < hot_3rd) & (df_test['시군구'].str.contains(pattern))
    df_test.loc[condition, '조정대상지역'] = 0

    # 경상북도 경산시
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < 202207) & (df_test['시'] == '경상북도 경산시')
    df_test.loc[condition, '조정대상지역'] = 1

    # 경북 경산시 하양읍, 진량읍, 압량읍 0으로 변환
    ls = ['경상북도 경산시 하양읍', '경상북도 경산시 진량읍', '경상북도 경산시 압량읍']
    pattern = '|'.join(ls)
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < hot_3rd) & (df_test['시군구'].str.contains(pattern))
    df_test.loc[condition, '조정대상지역'] = 0

    # 경상남도 창원시 성산구
    condition = (df_test['계약년월'] >= 202101) & (df_test['계약년월'] < hot_3rd) & (df_test['시'] == '경상남도 창원성산구')
    df_test.loc[condition, '조정대상지역'] = 1

    # 결측 없애고 정수로 변환
    df_test.fillna(0, inplace=True)
    df_test = df_test.astype({'조정대상지역':'int'})

    return df_test