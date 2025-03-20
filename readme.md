# 🍿 영화 데이터 기반 흥행 분류 및 수익 예측 통합 모델
<br />
<table>
  <thead>
    <td align="center">
      <a href="https://github.com/DawnSurplus">
        <img src="https://github.com/DawnSurplus.png" width="150" alt="hojin"/><br /><hr/>
        성호진
      </a><br />
    </td>
    <td align="center">
      <a href="https://github.com/Ohjunghh">
        <img src="https://github.com/Ohjunghh.png" width="150" alt="junhee"/><br /><hr/>
        오정현
      </a><br />
    </td>
    <td align="center">
      <a href="https://github.com/haides8973">
        <img src="https://github.com/haides8973.png" width="150" alt="minjung"/><br /><hr/>
        이상준
      </a><br />
    </td>
    <td align="center">
      <a href="https://github.com/comet39">
        <img src="https://github.com/comet39.png" width="150" alt="yugyeong"/><br /><hr/>
        이혜성
      </a><br />    
  </thead>
</table>
<br /><br />



# 📊 EDA 리뷰
> | 제작비 - 수익 | 평점 - 수익 | 추천수 - 수익 |
> | --- | --- | --- |
> | ![image](./img/budget_revenue.png) | ![image](./img/vote_average_revenue.png) | ![image](./img/vote_count_revenue.png)

> | 추천수 - 인기지수 | 월별 수익 분포 |
> | --- | --- |
> | ![image](./img/pop_vote_count_scatter.png) | ![image](img/output3.png) |
- 어떤 요소들이 수익에 가장 큰 영향을 미치는지를 파악하여 흥행 요인을 기반으로 한 수익 예측 모델을 생성
- 필요성
   - 영화 제작자는 제한된 자원을 가장 효과적으로 활용하여 최대의 수익을 얻고자 함
   - 제작비, 평점, 추천수와 같은 요소들이 수익에 어떻게 기여하는지를 이해하여, 영화 제작 및 마케팅 전략을 최적화

<br /><br />



# 📍 프로젝트 개요
영화 제작사와 투자자들이 개봉 전 흥행 가능성과 수익성을 사전에 판단하고 의사결정에 도움이 될 수 있는 맞춤형 모델을 개발
<br /><br />


### 📍 모델 선정 배경
- 영화 산업에서는 개봉 전후로 수익을 예측하는 것이 매우 중요하므로 **수익을 예측하는 회귀 모델**을 구축
- 단순한 수익 예측뿐만 아니라, **손익분기점 기준으로 영화의 흥행 여부를 분류하는 모델**을 먼저 구축하여 흥행 여부를 사전 판단

하지만 단일 예측 모델만으로는 데이터 패턴을 충분히 반영하지 못할 가능성이 있음 
- 따라서, 더 정교한 예측을 위해 **클러스터링 모델을 구축하여 영화 데이터를 그룹화**한 후, 그룹별 특성을 고려한 개별 수익 예측 모델을 개발

---
<br /><br /><br />





# 📊 데이터셋 개요
- **출처**: Kaggle - The Ultimate 1Million Movies Dataset (TMDB + IMDb)

<br />

## 📍 데이터 전처리 과정

#### 1. 사용하지 않는 feature 제거
#### 2. 결측치 및 이상치 처리
  - 영화 ID(id) 기준으로 중복 행 제거
  - 결측치 제거
  - 이상치 제거
    - 0 값을 가지는 행
    - 예산(budget)과 수익(revenue)이 각각 1000 미만인 행

#### 3. 특성 컬럼 추가
  - 물가 반영 금액 추가
  - 개봉 연도/월월 추가
  - 순수익(profit) 추가
    - $profit = revenue - budget$
  - 수익률(ROI) 추가
    - $ROI = profit / budget * 100$

#### 4. 다중 레이블 인코딩
  - genres 데이터를 원-핫 인코딩 처리

#### 5. 수치형 변수 스케일링
  - StandardScaler를 사용하여 `budget`, `revenue`, `popularity`, `vote_average`, `vote_count`등을 표준화

<br />


## 📍 특성
- ### 주요 Column
  > | 컬럼명 | 설명 |
  > | --- | --- |
  > | `id` | 영화 id |
  > | `title` | 영화 제목 |
  > | `budget` | 영화 제작 비용 |
  > | `adjusted_budget` | 영화 제작 비용 (물가 반영) |  
  > | `revenue` | 영화 수익 |
  > | `adjusted_revenue` | 영화 수익 (물가 반영) |
  > | `genres` | 영화 장르 (',' 구분분) |
  > | `vote_average` | 평점 |
  > | `vote_count` | 추천 수 |
  > | `popularity` | 인기 지수 |
  > | `release_date` | 영화 개봉일자 |
  > | `release_year` | 영화 개봉년도 |
  > | `release_month` | 영화 개봉월 |

- ### 상관관계
  ![](img/common_heatmap.png)

---

<br/><br/>





# 🏆 모델 및 성능 평가


## 📍 흥행 여부 분류 모델

### 1. 모델 선정
   - 영화의 흥행 여부는 여러 요인이 복합적으로 작용<br/>
      - 복잡한 관계를 잘 다룰 수 있는 모델 필요
   1. #### `RandomForest`
       - 여러 개의 결정 트리를 학습하여 앙상블로 예측을 개선
   2. #### `XGBoost`
       - 부스팅 방식으로 성능을 극대화
<br/><br/>

### 2. 데이터 추가 전처리
```python
df['success'] = df['ROI'].apply(lambda x: 1 if x >= 100 else 0)
```
- 영화 성공 여부 분류
  - ROI 값이 100 이상이면 성공(1), 그렇지 않으면 실패(0)로 간주
  - 해당 기준은 사용자가 원하는 정도로 조정 가능 

```pyhon
df.drop(['id','title','ROI','cast','director','adjusted_revenue','adjusted_budget','revenue','release_date','profit','popularity'],axis=1,inplace=True) 
```
- 불필요한 특성들을 제거하여 모델 학습에 필요한 정보만 남김
<br/><br/>

### 3. 모델 튜닝
- #### RandomForest
  - **최적 파라미터** : `n_estimators = 100`,`max_depth = 7` 이 설정에서 성능이 가장 우수
    - `n_estimators`: **100,200,300,400,500,600** 으로 테스트  
    - `max_depth`: **3,4,5,6,7**로 테스트 </br> 
    
  - **특성 선택**:
    - `vote_average`, `vote_count`, `budget`, `genres`, `release_year`, `release_month`를 포함
      - 각 특성을 하나씩 제거해보았으나, 모델 성능이 하락하여 모든 특성을 포함하는 것이 더 나은 결과를 얻음
  	  
  - **성능 평가**:
    
    ![rdc_eval](img/resized_rdc_evaluation.png)
</br>

    > 최적화 과정에서 과적합을 줄이기 위해 하이퍼파라미터 튜닝을 수행
    > ➡️ 초기 모델에 비해 일반화 `성능이 향상`

- #### XGBoost 모델 튜닝
  - **최적 파라미터** : `subsample = 0.7`, `n_estimators = 700`, `max_depth = 3`, `learning_rate = 0.01`, `colsample_bytree = 0.8` 에서 성능이 가장 우수
    - `n_estimators`: 500부터 800까지 100 단위로 탐색
    - `max_depth`: 3부터 4까지 탐색
    - `learning_rate`: 0.01, 0.05, 0.1, 0.15, 0.2의 값을 실험
    - `subsample`:  0.7, 0.8, 0.9로 설정하여 샘플의 일부만 학습에 사용하여 과적합을 방지
    - `colsample_bytree`: 0.7, 0.8, 1.0. 각 트리를 학습할 때 사용할 특성(열)의 비율을 지정
</br>

  - **특성 선택**:
    - RandomForest와 동일하게 진행
     
  - **성능 평가**:

    ![xgb_eval](img/resized_xgb_evaluation.png)
</br> 

    > 최적화 과정에서 과적합을 줄이기 위해 하이퍼파라미터 튜닝을 수행<br/>
    > 초기 모델에 비해 일반화 '성능이 향상'

</br>

### 4. 최종 흥행 여부 분류 모델 선정
> | **모델**         | **Test F1 Score** | **5-fold 교차 검증 F1 Score** |
> |------------------|------------------|-----------------------------|
> | **RandomForest**  | 0.68             | 0.69                        |
> | **XGBoost**       | 0.69             | 0.67                        |

교차 검증에서 **F1-score**가 가장 높았던 **RandomForestClassifier**를 최종 모델로 선정

#### ✅ 평가 지표 선정 이유  
- **`투자자 관점(Precision)`** : 정밀도 중요
  - 수익성이 없는 영화를 성공으로 잘못 예측(False Positive)하면 큰 손실을 초래할 수 있음
- **`제작사 관점(Recall)`** : 재현율 중요
  - 성공 가능성이 있는 영화를 실패로 잘못 예측(False Negative)하면 기회를 놓칠 수 있음
- **`균형적 접근(F1-score)`** : 최종 평가 지표로 사용
  - Precision과 Recall 간 균형을 맞추기 위함
  
</br>

</br></br>



## 📍 수익 예측 모델
- 만약, 코로나가 발생하지 않았다면 연도별 수익은 어떻게 나올까?
  - covid19 이전 개봉한 영화의 데이터를 학습하여, 코로나 시기에 개봉한 영화들의 수익 잠재력을 예측

### 1. 모델 선정
   - #### Linear Regressor
     - EDA 리뷰 지표 기준 수익과 관련된 feature들은 선형성을 띔
   - #### Random Forest Regressor
     - 영화의 흥행 여부는 여러 요인이 복합적으로 작용<br/>
     - 복잡한 관계를 잘 다룰 수 있는 모델 필요

### 2. Feature 선정
 - profit/adjusted_revenue와 같이 revenue를 포함/조정하고 있는 데이터 제외
 - revenue와 상관관계가 높은/유의미한 feature 선정
   - `budget`, `popularity`, `vote_average`, `vote_count`
 - 연도/월 별 따른 패턴이나 장르가가수익에 영향을 미칠 수 있으므로 반영
   - `genres`, `release_year`, `release_month`

### 2. 성능 평가
- #### Linear Regressor : `test_size=0.25`, `random_state=4`의 경우가 과적합이 적고 Best
  - `test_size` : 0.1부터 0.3까지 0.05 단계씩 변경하면서 성능 확인
  - `random_state` : 0부터 50까지 변경하면서 성능 확인
    ![image](./img/linear_test.png)

- #### Random Forest Regressor
  - 
![image](./img/revenue_by_year_before_covid19.png)
- Random Forest Regressor 선정 결과

### 3. 예측 결과
![image](./img/covid19_result.png)
- 2021년까지 코로나의 영향으로 영화 시장이 직접적으로 타격을 입음 (추가설명필요)
- 2022년 부터 경기 회복으로 인해 영화 시장이 많이 회복되기 시작
  - '아바타: 물의 길', '탑건: 매버릭'과 같은 대형 영화들이 대거 개봉
</br></br>



### 📍 클러스터링 모델
1. ####  클러스터링 모델 개발 이유
	영화 데이터 분석 시에 feature 간의 관계가 비선형적이고 복잡한 것을 알 수 있었음.
	이때, 클러스터링으로 수익과 연관이 있는 특성별로 데이터를 분리한 후 개별 회귀 모델을 생성하면 예측 정확도가 높아질 가능성이 있음.
	
2. #### 모델 선정
 - **K-Means** : 영화 데이터를 비슷한 특징을 가진 영화 그룹(예: 고수익 영화, 저예산 독립 영화)을 식별하는 데 적합
      
- **DBSCAN** : 밀도가 높은 클러스터를 형성하고, 이상치를 분리하여 분석
        
3. #### Feature 선정
	수익예측 모델에 훈련시킬 클러스터를 구분하는게 목적이기에 수익성에 따른 그룹을 군집화 하기 위해, revenue와 관련된 핵심 특성만 선택하여 훈련
	- `budget`, `popularity`, `vote_average`, `vote_count`, `genres`

4. #### 성능 평가
- 클러스터링 모델 평가
	| **클러스터링 모델** | **평가 지표** | **결과** | 
	|--------------------------|---------------|---------|
	| **K-Means** | Silhouette Score | 0.31 | 
	| **DBSCAN** | Silhouette Score | 0.70 | 

- K-Means 클러스터링을 활용하여 수익 예측 회귀 모델의 성능을 평가

   *평가 지표는 MSE, R2*
| **LinearRegression** | **RandomForestRegressor** | 
	|--------------------------|--------------------|
	| ![image](./img/km+lr.png) |  ![image](./img/km+rf.png) |

- DBSCAN 클러스터링을 활용하여 수익 예측 회귀 모델의 성능을 평가

   *평가 지표는 MSE, R2*
	| **LinearRegression** | **RandomForestRegressor** | 
	|--------------------------|----------------------|
	| ![image](./img/db+lr.png)  |  ![image](./img/db+rf.png) | 


5. #### 성능 향상을 위해 노력한 점
- #### 특징 공학 : 차원 축소
  - 다중 레이블로 인코딩된 genres 특성을 포함한 데이터의 차원을 PCA를 통해 주요 3개의 차원으로 축소
  
	✔️ PCA 결과: 주요 차원이 전체 분산의 약 62%를 차지하므로, 데이터의 복잡성을 줄이는 데 성공

- **K-Means** 파라미터 튜닝 
  - 최적의 파라미터 (`n_clusters`)를 찾기 위해 **Elbow Method**와 **Silhouette Score**를 활용  

    ![n_clusters](./img/n_clusters.png)  

	1.  **엘보우 메서드 (왼쪽 그래프)**: k=3~6 근처에서 감소율이 완만해지는 경향이 보임
	2.  **실루엣 점수 (오른쪽 그래프)**:  k=2에서 가장 높은 점수를 보이지만,  k=6도 비교적 높은 점수를 유지

	✔️  최적의 클러스터 개수는  **6**

- **DBSCAN** 파라미터 튜닝
  - 최적의 파라미터 (`eps`, `min_samples`)를 찾기 위해 **k-거리 정렬**과 **Silhouette Score**를 활용  
	 
	 ![eps](./img/k-dist.png)  
	 ✔️ **eps = 0.9** : 모든  `min_samples`  값에서 가장 높은 실루엣 점수
 
     ![eps_min_samples](./img/eps_min_samples.png) 
	 <br/>✔️ **min_samples = 5~10** : 그래프에서 급격한 변화가 나타나는 지점은 약 **5~10** 근처

6. #### 예측 결과

- #### K-Means 클러스터링을 활용한 수익 예측  
  - **Linear Regression**  
  
    ![KMeans_LR](./img/km+lr_pred.png)  
  
  - **Random Forest Regressor**  
  
    ![KMeans_RF](./img/km+rf_pred.png)  

- #### DBSCAN 클러스터링을 활용한 수익 예측  
  - **Linear Regression**  
  
    ![DBSCAN_LR](./img/db+lr_pred.png)  
  
  - **Random Forest Regressor**  
  
    ![DBSCAN_RF](./img/db+rf_pred.png)  
	 </br>

7. 결론
- 클러스터링을 적용한 후 회귀모델을 학습시키면 더 나은 성능을 보일 것이라고 예측했으나, 실제 결과는 그렇지 않았음.
- 클러스터링 모델의 성능이 회귀모델의 성능에 비례할 것이라고 예상했으나, 이 역시 성립하지 않았음.

✔️ 회귀모델과 클러스터링 알고리즘 간의 상호작용을 고려하여 하이퍼파라미터 튜닝 및 피처 엔지니어링 개선이 필요하다고 판단
	 
---
<br /><br />



## 🚀 결론 및 향후 개선 방향
- (프로젝트에서 얻은 주요 인사이트)

### 프로젝트 개선점

- 개봉 전에 알 수 있는 특성만으로 학습하기
  - 프로젝트에서 사용한 `vote_count`, `popularity` 등의 특성은 개봉을 해야만 알 수 있음.
  - 따라서 프로젝트의 목표인 '개봉 전'을 상정한 모델이라면 그러한 것들이 없는 상태에서도 예측을 할 수 있어야 함.
 
- 특성 전처리 개선
  - `budget`, `revenue` 이 한쪽으로 치우친 경향을 보이기 때문에, 해당 특성들에 로그 변환을 가해주면 추가적인 성능 향상을 기대할 수 있음.
  - 학습에 사용된 특성 선택을 직관에 따라 극히 일부로 제한했는데, SelectKBest 같은 모듈을 사용하여 모든 특성에 대해 Label Data와 어느 정도의 연관성을 갖는지 수치화하여, 상위 특성들을 남기는 방법들을 사용할 수도 있음.
    
