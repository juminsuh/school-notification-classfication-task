# school-notification-classfication-task

# 학교 공지사항 분류 프로젝트

## Experiment Results

*acc 기준

|  | Multilingual BERT | KoBERT | KoBERT with LLRD |
| --- | --- | --- | --- |
| Test | 0.90530 | 0.8976 | 0.8982 |

## #01 프로젝트 소개

💡 프로젝트의 필요성 및 개요

학교 공지사항에서 학사, 경력, 장학 등 카테고리의 경계가 모호하여 사용자 입장에서 원하는 정보를 빠르고 정확하게 얻지 못하는 점에서 불편을 느낌.

특히, 홈페이지의 주 사용층인 학생의 주요 관심사는 **스펙을 쌓을 수 있는 활동에 대한 공지사항**임.

➡️ 공지사항을 스펙이 되는 것(1)과 그렇지 않은 것(0)으로 분류할 수 있는 모델을 구축하고자 함.

## #02 데이터 크롤링

- title scrapping 함수를 정의 → BeautifulSoup 라이브러리를 사용해 원하는 페이지의 원하는 부분을 크롤링할 수 있음.
- 홈페이지의 개발자 모드를 통해 스크래핑하고자 하는 부분이 어떤 태그, 어떤 클래스인지 확인 후 원하는 부분을 find 또는 find_all 메서드를 통해 크롤링할 수 있음.

## #03 EDA

- 한글 폰트 (나눔 폰트 설치) - 설치 후 세션 다시 시작
- 결측치 확인 & 중복 제거
- class balance 확인: train data의 class imbalance는 smote / oversampling / undersampling과 같은 데이터 조작을 필요로 할 수 있기 때문에 확인해야 할 중요한 요소
- title length 파악: 클래스에 따른 텍스트의 길이가 크게 다르지 않음을 확인함
- word cloud로 클래스 별 어떤 단어가 dominant한 지 시각적으로 확인

## #04 전처리

- 출현 빈도가 낮은 단어 제거: Counter()를 사용해 모든 단어의 출현 빈도 수를 딕셔너리에 저장, 이후 빈도 수가 3번 이하인 단어를 제거
- 크롤링한 학교 명을 95% 제거 → 특정 학교명으로 학습되는 일을 방지
- title에 영어와 한국어만 남기고 숫자, 괄호, 특수 문자는 모두 제거
- custom words와 stop words 정의: ‘장학생’과 같은 하나의 단어이자 분류 테스크에서의 주요 키워드를 ‘장’, ‘학생’으로 잘못 인식하여 토큰화하는 경우 발생 → custom words를 정의해 사용자 지정 사전 제작
- 한 글자는 drop하되 ‘팀’, ‘랩’과 같은 주요 키워드는 남겨두기

## #05 모델링

### Embedding

---

- n-gram vectorization: 단어의 앞뒤 맥락 정보를 더 잘 파악하기 위한 작업
- Kobert Tokenizer: 한국어의 형태소 분석을 반영하여 더 의미 있는 임베딩을 형성할 수 있음.
- high priority words: ‘프로그램’, ‘서포터즈’, ‘인턴’과 같이 분류에 있어 중요도가 높은 단어들을 포함하는 title을 1, 그렇지 않으면 0으로 설정한 뒤 machine learning model을 실행할 때 그 가중치를 높임.

⇒ 최종적으로 임베딩 및 중요도를 높인 단어들의 특징을 X_combined 변수에 stacking

### Machine Learning Model

---

*acc 기준 

|  | Random Forest | Multinomial Naive Bayes | Bernoulli Naive Bayes | Logistic Regression (ver. plain)  | Logistic Regression (ver. optimal hyperparameters with GridSearch) |
| --- | --- | --- | --- | --- | --- |
| 일반 버전 | 0.8456 | 0.8844 | 0.8942 | 0.7052 | 0.9131 |
| 특정 단어 중요도 높인 버전  | 0.8456 | 0.8850 | 0.8942 | 0.7041 | 0.9131 |

### Deep Learning Model

---

### Multilingual BERT

- 다양한 단어로 이루어진 텍스트를 이해할 수 있도록 설계된 BERT → 일반화된 언어 표현을 학습
- trained with self-supervised fashion (no ground-truth label)
- Pretrained objectives
    - Masked Language Modeling(MLM): randomly masks 15% of the input words → learn bidirectional represenation of word
    - Next Sentence Prediction(NSP): concatenates two masked sentences → predict whether two sentences are successive or not
- In this way, M-BERT learns an inner represenation of languages that can be useful in extracting features for each downstream task

### **Process**

1. title의 시작에 [CLS], 끝에 [SEP] 토큰을 붙임. [CLS]는 classification의 약자로, 문장의 시작에 붙임으로써 이 위치에서 특징 벡터를 추출하도록 모델에게 알림. [SEP]는 separator의 약자로, 문장의 끝에 붙임으로써 서로 다른 문장들을 구분함. 
2. 문장의 단어를 tokenzing: tokenizer
    
    ```python
    tokenizer = BertTokenizer.from_pretrained('bert-base-multillingual-cased', do_lower_case = False)
    tokenized_X_train = [tokenizer.tokenize(fixed) for fixed in X_train_fixed]
    tokenized_X_test = [tokenizer.tokenize(fixed) for fixed in X_test_fixed]
    ```
    
3. token을 각각 대응하는 id로 나타냄
4. pad sequence를 통해 시퀀스 데이터를 지정한 길이인 max_len으로 맞춰줌. 빈 부분은 0으로 패딩. 
5. 문장의 attention mask를 만듦 - id가 존재하면 1.0, 0으로 패딩됐으면 0.0으로 mask를 만듦
6. hyperparameter setting: 배치 사이즈, 에폭 수, learning rate scheduler 조절을 통해 최적의 accuracy를 가지는 hyperparameter를 찾음

| lr scheduler 사용 여부 | batch size | epochs | step size | gamma  | learning scheduler | learning rate | optimzier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| o | 32 | 10 | 3 | 0.1 | StepLR | 3e-5 | AdamW |
1. 10번의 에폭 동안 overfitting을 막기 위해 early stopping 기법 적용: eval loss가 세 번 동안이나 개선되지 않으면 overfitting이라고 판단하고 훈련 종료 → 7번의 epoch에서 훈련이 종료됨

### KoBERT

- 기존 BERT의 한국어 성능을 극복하기 위해 SKT Brain에서 개발한 모델
- 위키피디아와 뉴스 등에서 수집한 수백만 개의 한국어 문장의 대규모 말뭉치(Corpus)를 통해 학습됨
- output layer를 추가함으로써 언어 특화 모델을 customize할 수 있다는 장점 존재

### **Process**

1. Bert Classifier Architecture 정의: 
2. get_cosine_schedule_with_warmup을 활용한 learning rate scheduling: 학습 초기 단계에서 학습율을 점진적으로 증가시키는 warming-up step을 적용한 후, cosine 형태로 학습율을 점차 감소시키는 방식 
3. Layer-wise Learning Rate Decay(LLRD) 적용: 모델의 각 레이어마다 다른 학습율을 적용시키는 기법으로, 상위 레이어는 보다 구체적인 의미 / 테스크 특화 정보를 담고 있고 하위 레이어는 기본적은 문법, 어휘, 구조적 정보 등을 담고 있음. 따라서 하위 레이어에서는 작은 learning rate로 일반적이고 기본적인 구조를 학습한 상태를 유지하고 상위 레이어에서는 높은 learning rate로 테스크에 특화되도록 파라미터가 신속하게 업데이트되어야 함. → 하위 6개 layer는 learning rate로 0.1, 상위 6개 layer는 learning rate로 0.5를 가짐
4. LLRD를 적용한 모델이 그렇지 않은 모델보다 outperformed
