# 🍿 영화 데이터 기반 흥행 분류 및 수익 예측 통합 모델

## 📖 개요
### 📍 프로젝트 주제
영화 제작사와 투자자들이 개봉 전 흥행 가능성과 수익성을 사전에 판단하고 의사결정에 도움이 될 수 있는 맞춤형 모델을 개발


### 📍 모델 선정 배경

- 영화 산업에서는 개봉 전후로 수익을 예측하는 것이 매우 중요하므로 **수익을 예측하는 회귀 모델**을 구축
- 단순한 수익 예측뿐만 아니라, **손익분기점 기준으로 영화의 흥행 여부를 분류하는 모델**을 먼저 구축하여 흥행 여부를 사전 판단

하지만 단일 예측 모델만으로는 데이터 패턴을 충분히 반영하지 못할 가능성이 있음 
- 따라서, 더 정교한 예측을 위해 **클러스터링 모델을 구축하여 영화 데이터를 그룹화**한 후, 그룹별 특성을 고려한 개별 수익 예측 모델을 개발

---

## 📊 데이터셋 개요
### 📍 데이터셋 소개
- **출처**: (데이터셋이 출처한 사이트 또는 기관)
- **특징**: (데이터셋의 주요 컬럼 및 특성)

### 📍 데이터 전처리 과정

1. 사용하지 않는 feature 제거

2. 결측치 및 이상치 처리
  - 영화 ID(id) 기준으로 중복 행 제거
  - 결측치 제거
  - 이상치 제거
    - 0 값을 가지는 행
    - 예산(budget)과 수익(revenue)이 각각 1000 미만인 행

3. 특성 컬럼 추가
  - 물가 반영 금액 추가
  - 개봉 연도 추가
  - 수익률(ROI) 추가
  - 순수익(profit) 추가

4. 다중 레이블 인코딩
  - genres 데이터를 원-핫 인코딩 처리

5. 수치형 변수 스케일링
  - StandardScaler를 사용하여 `budget`, `popularity`, `vote_average`, `vote_count`등을 표준화

---

## 🏆 모델 및 성능 평가

### 📍 흥행 여부 분류 모델
> #### 모델 선정
  #### RandomForest & XGBoost
RandomForest는 여러 개의 결정 트리를 학습하여 앙상블로 예측을 개선하고, XGBoost는 부스팅 방식으로 성능을 극대화합니다. 영화의 흥행 여부는 여러 요인이 복합적으로 작용하는데 두 모델 모두 복잡한 관계를 잘 다룰 수 있는 특징을 가지고 있어서 선택했습니다.
  
> #### 데이터 추가 전처리
```python
df['success'] = df['ROI'].apply(lambda x: 1 if x >= 100 else 0)
```
영화 성공 여부를 분류하기 위해 ROI 값이 100 이상이면 성공(1), 그렇지 않으면 실패(0)로 간주하여 success 열에 저장합니다.

```pyhon
df.drop(['id','title','ROI','cast','director','adjusted_revenue','adjusted_budget','revenue','release_date','profit','popularity'],axis=1,inplace=True) 
```
불필요한 특성들을 제거하여 모델 학습에 필요한 정보만 남깁니다.</br>
  #### RandomForest
  #### 모델 튜닝
  - **최적 파라미터**:
    - `n_estimators`: **100,200,300,400,500,600** 으로 테스트  
    - `max_depth`: **3,4,5,6,7**로 테스트 </br> 
    - `n_estimators = 100`,`max_depth = 7` 이 설정에서 성능이 가장 우수함
    
- **특성 선택**:
  - `vote_average`, `vote_count`, `budget`, `genres`, `release_year`, `release_month`를 포함
  - 각 특성을 하나씩 제거해보았으나, 제거 시 모델 성능이 하락하여 모든 특성을 포함하는 것이 더 나은 결과를 얻었습니다.
  
> #### 성능 평가
![rdc_eval](images/rdc_evaluation.png)

</br> 최적화 과정에서 과적합을 줄이기 위해 하이퍼파라미터 튜닝을 수행했습니다. 초기 모델에 비해 일반화 성능이 향상되었습니다.

 #### XGBoost
#### 모델 튜닝
  - **최적 파라미터**:
    - `n_estimators`: 500부터 800까지 100 단위로 탐색
    - `max_depth`: 3부터 4까지 탐색
    - `learning_rate`: 0.01, 0.05, 0.1, 0.15, 0.2의 값을 실험
    - `subsample`:  0.7, 0.8, 0.9로 설정하여 샘플의 일부만 학습에 사용하여 과적합을 방지
    - `colsample_bytree`: 0.7, 0.8, 1.0. 각 트리를 학습할 때 사용할 특성(열)의 비율을 지정
    - `subsample = 0.7`, `n_estimators = 700`, `max_depth = 3`, `learning_rate = 0.01`, `colsample_bytree = 0.8` 에서 성능이 가장 우수했습니다.</br>
  - **특성 선택**:
    - RandomForest와 동일하게 진행했습니다.
     
#### 모델 성능
![xgb_eval](images/xgb_evaluation.png)

</br> 최적화 과정에서 과적합을 줄이기 위해 하이퍼파라미터 튜닝을 수행했습니다. 초기 모델에 비해 일반화 성능이 향상되었습니다. 

### 최종 흥행 여부 분류 모델 선정  

| **모델**         | **Test F1 Score** | **5-fold 교차 검증 F1 Score** |
|------------------|------------------|-----------------------------|
| **RandomForest**  | 0.68             | 0.69                        |
| **XGBoost**       | 0.69             | 0.67                        |

교차 검증에서 **F1-score**가 가장 높았던 **RandomForestClassifier**를 최종 모델로 선정했습니다.  

#### ✅ 평가 지표 선정 이유  
- **투자자 관점(Precision)**: 수익성이 없는 영화를 성공으로 잘못 예측(False Positive)하면 큰 손실을 초래할 수 있어 정밀도가 중요합니다.  
- **제작사 관점(Recall)**: 성공 가능성이 있는 영화를 실패로 잘못 예측(False Negative)하면 기회를 놓칠 수 있어 재현율이 중요합니다.  
- **균형적 접근(F1-score)**: Precision과 Recall 간 균형을 맞추기 위해 최종 평가 지표로 사용했습니다.
  
</br>

### 📍 수익 예측 모델
> #### 데이터 추가 전처리

> #### 성능 평가

> #### 예측 결과

</br></br>

### 📍 클러스터링 모델
> 클러스터링 모델 개발 이유

> 모델 선정
  #### K-Means
  #### DBSCAN 

> Feature 선정
수익성에 따른 그룹을 군집화 하기 위해, revenue와 관련된 핵심 특성만 선택하여 훈련
- `budget`, `popularity`, `vote_average`, `vote_count`, `genres`

> #### 성능 평가
##### 1. K-Means 모델 성능 평가
##### 2. DBSCAN 모델 성능 평가
##### 3. K-Means 클러스터링을 활용하여 수익 예측 회귀 모델의 성능을 평가
- LinearRegression
- RandomForestRegressor
##### 4. DBSCAN 클러스터링을 활용하여 수익 예측 회귀 모델의 성능을 평가
- LinearRegression
- RandomForestRegressor

> #### 성능 향상을 위해 노력한 점
##### 특징 공학
- 차원 축소
  - 다중 레이블로 인코딩된 genres 특성을 포함한 데이터의 차원을 PCA를 통해 주요 3개의 차원으로 축소
  => PCA 결과: 주요 차원이 전체 분산의 약 62%를 설명하며, 데이터의 복잡성을 줄이는 데 성공했습니다.
##### 하이퍼파라미터 튜닝
- K-Means
최적의 파라미터(`n_clusters`)를 찾기 위해 Elbow Method와 Silhouette Score를 활용

- DBSCAN
최적의 파라미터(`eps`, `min_samples`)를 찾기 위해 k-거리 정렬과 Silhouette Score를 활용

> #### 예측 결과
##### 1. K-Means 클러스터링을 활용하여 수익 예측
- LinearRegression
- RandomForestRegressor
##### 2. DBSCAN 클러스터링을 활용하여 수익 예측
- LinearRegression
- RandomForestRegressor

## 🚀 결론 및 향후 개선 방향
- (프로젝트에서 얻은 주요 인사이트와 앞으로의 발전 방향)
