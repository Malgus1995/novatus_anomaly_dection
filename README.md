# 경남도 - UNIST, AI-Novatus 1 기 (2022 년)

## PBL (Problem-Based Learning)   
<img src="https://user-images.githubusercontent.com/62151520/167406557-5a82c282-f3ca-4b0f-b250-529a6c26f14e.png" width="300" height="250"/> <img src="https://miro.medium.com/max/1400/0*ILUJj0eMAFtrSblq" width="300" height="250"/>  

<!-- Quote -->
<!-- Table --> 
> |주제|**AI 기반 PHM 기술 (설비 건전성 상태 감지)**|
> |:--|:--|     
> |팀원|김영민 (DN Solutions (전) 두산 공작 기계), Solo (역할 : Do Everything)|   
> |기간|4/29 ~ 7/15|    
> |지도 교수님|**UNIST 권상진 교수님 (https://or.unist.ac.kr/)**|   
> |조교님|**정성욱** 석사 과정님|  
> |과제 목표|ML 의 장점을 효과적으로 활용하는 분야에서 크게 뒤처진 제조업에서, General 하게 적용 할 수 있는 강건한 시계열 예측 알고리즘 구축 (시계열 데이터가 가장 일반적이고 보편화된 도메인 영역임.)|      
> |문제 정의|수명을 물리적 신뢰성 테스트로만 의존하는 현재 상황에서, Big Data 를 활용한 RUL (Remaining-Useful-Life) 예측 (해결해야 하는 문제라기보단 AI 를 활용해 더 높은 기술 차원으로 도약)|  
> |해결 방안|Pre-Processing 완료 ☞ 가장 중요도 높은 센서 랭킹 도출 완료 ☞ **Feature Extraction (Extension) (~ing)** ☞ **시계열 예측 알고리즘 (LSTM-Based VAE-GAN Model)**|       
> |**예상되는 기술적 기대 성과**|**고장 시기 사전 예측으로 설비 다운 타임 최소화 및 고객과 분쟁 예방**, 고장 발생 패턴 파악으로 유닛 취약점 clustering, 데이터 취득 위한 Cloud / Sensing 기술 개발 촉진 **(경쟁사 대비 차별적인 기업 Digitalization 부여)**|               

<!-- Quote -->
<!-- Table -->
  
> |**Industry**|Manufacturing (lagged far behind in the area of effectively taking advantage of ML techniques)|  
> |:--|:--|   
> |**Problem Description**|failure of equipment (unplanned downtime, unnecessary maintenance..etc) often results in production loss 
> |**Data**|The resulting time-series data can be used in predictive modeling to determine the status of the machine or components|
> |**Strategy**|Bottom-Up (Base) + Top-down (additional)|  
> |**Goal**|**Build a predictor that performs well on unseen time series data (RUL, Remaining-Useful-Life)**|   
> |Raw data|https://www.kaggle.com/datasets/nphantawee/pump-sensor-data  https://www.kaggle.com/datasets/kimalpha/pump-sensor-data-preprocess-missing-values|  
___
> **Master Plan** :  

  * ~week 01 (4/29) : 교수님 및 석 / 박사 분들께 과제 브리핑~
  * ~week 02 (5/6) : Problem Definitions / EDA (Exploratory Data Analysis)~
  * ~week 03 (5/13) : EDA, **briefing**~ 
<!-- Table -->
> |**first meeting**||  
> |:--|:--|  
> |**Algorithm**|recommend the use of VAE (Auto-Encoding Variational Bayes), GAN (Generative Adversarial Networks), Boosting Algorithm, RandomForest|
> |**Missing values**|Determine the correlation of missing values against labeling, recommend the use of median values|
  * ~week 04 (5/20) : Pre-Processing~ ☞ [Pre-Processing Code](https://github.com/min0355/ai-novatus/blob/main/PBL%20main%20code/No.1_prepro.missing%20values.ipynb)    
      
    - **[After Pre-Processing]**  
    <img src="https://github.com/min0355/ai-novatus/blob/main/visualization/after%20preprocessing_msno.png" width="990" height="200"/>  
    <img src="https://github.com/min0355/ai-novatus/blob/main/visualization/after%20preprocessing.png" width="990" height="2000"/>
<!-- Table -->
> |**second meeting**||  
> |:--|:--|  
> |**Recommend the Algorithm**|statistics based approach : EWMA first and then CorGAN review ([EWMA_1](https://blog.minitab.com/ko/detect-small-shifts-in-the-process-mean-with-exponentially-weighted-moving-average-ewma-charts), [EWMA_2](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=61stu01&logNo=221282667120), [Pandas_EWMA](https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.ewma.html)), [CorGAN](https://github.com/astorfi/cor-gan)|
  * ~week 05 (5/27) : Feature Importance~ ☞ [Feature Importance Code](https://github.com/min0355/ai-novatus/blob/main/PBL%20main%20code/No.2_Feature%20importance.ipynb)      
     
    - **[Metrics and Scoring (using all (49) sensor signal)]**   
              
    <img src="https://github.com/min0355/ai-novatus/blob/main/visualization/results%20of%20training.gif" width="500" height="400"/>  
      
    - **[Feature importance by Predictive algorithm]**       
    <img src="https://github.com/min0355/ai-novatus/blob/main/visualization/LGB%20Feature%20importance.png" width="990" height="600"/>  
    <img src="https://github.com/min0355/ai-novatus/blob/main/visualization/XGBoost%20feature%20importance.png" width="400" height="600"/>  
      
    - **[Permutation importance by boosting and Tree model]**  
    - XGBoost, Catboost, RandomForest (no max_depth, max_depth=6, 10, 15)    
    <img src="https://github.com/min0355/ai-novatus/blob/main/visualization/permutation%20importance_xgb.PNG" width="220" height="457"/>  <img src="https://github.com/min0355/ai-novatus/blob/main/visualization/permutation%20importance_cat.PNG" width="220" height="457"/>  <img src="https://github.com/min0355/ai-novatus/blob/main/visualization/permutation%20importance_rf_no%20max.PNG" width="220" height="457"/>  <img src="https://github.com/min0355/ai-novatus/blob/main/visualization/permutation%20importance_rf_max%20dep%206.PNG" width="220" height="457"/>  <img src="https://github.com/min0355/ai-novatus/blob/main/visualization/permutation%20importance_rf_max%20dep%2010.PNG" width="220" height="457"/>  <img src="https://github.com/min0355/ai-novatus/blob/main/visualization/permutation%20importance_rf_max%20dep%2015.PNG" width="220" height="457"/>  
    
    - **Results**  
    <img src="https://github.com/min0355/ai-novatus/blob/main/visualization/feature%20importance_final%20results.PNG" width="600" height="300"/>  

<!-- Table -->
> |**third meeting**||  
> |:--|:--|  
> |**Description of Scenario**|Add feature importance ([95 % feature importance](https://towardsdatascience.com/improving-random-forest-in-python-part-1-893916666cd)), Neural Network Paper Review : [LSTM-Based VAE-GAN](https://anencore94.github.io/2020/10/28/lstm-based-vae-gan.html),  [Fire detection](https://ieeexplore.ieee.org/abstract/document/9357405), feature extension using FFT (Fast Fourier Transform)  
 
  * ~week 06 (6/3) : Feature Selection, Feature Extension~ ☞ [Feature Selection & Extension Code](https://github.com/min0355/ai-novatus/blob/main/PBL%20main%20code/No.3_Feature%20selection%20and%20Feature%20Extension.ipynb)    
    - **Feature Selection** 
    <img src="https://github.com/min0355/ai-novatus/blob/main/visualization/result_remove%20corr.png" width="400" height="220"/>   
      
    - **Importance Ranking**  
    <img src="https://github.com/min0355/ai-novatus/blob/main/visualization/univariate%20feature%20selection.png" width="300" height="250"/>   

  * ~week 07 (6/10) : algorithm, **interim findings**~ ☞ [Algorithm Code](https://github.com/min0355/ai-novatus/blob/main/PBL%20main%20code/No.4_Algorithm.ipynb)  
  * ~week 08 (6/17) : algorithm (LSTM, GAN, VAE - Individual Code Review, think construction of frame)~ ☞ reading paper, basic algorithms  
  * ~week 09 (6/24) : Evaluation metric (with test set, normal | noisy version)~  
  * week 10 (7/1) : algorithm (LSTM-VAE Model, Hyperparameter tune)   
  * week 11 (7/8) : algorithm (LSTM-VAE Model, Hyperparameter tune), Visualize results    
  * week 12 (7/15) : **Final Presentation**

> **Points to Note** :
  * Sensor signal in real world    
  * Sensor attachment position  
  * Sensor output signal type      
  * **Obtain abnormal labels** (mechanical engineering design domainknowledge)  
  * Test methods and order to ensure correct labels  
  * Define the time we want to predict  
  * Utilize the analyzed results data   
  * **Data Representation Issues**  

> **Cold Start** 
  * Employees and organizations need to understand data driven AI      

> **Engineering Design Consideration**  
> (The essence of engineering is the commercialization of technology)  
  * feasibility :    
  * performance : cost incurred by definitions of service   
  * constraints : sensor price, unchanging unit design    
  * creativity and novelty :         

### REVIEW
___
  * **https://www.analyticsvidhya.com/blog/2021/12/time-series-forecasting-with-extreme-learning-machines/**  
      - Statistical method : ARIMA, AR, MA, ARMA, ARIMA, ARIMAX, SARIMAX (they fail to capture seasonality, they can only be applied to stationary time series)   
      - NN models : ELMs (single hidden layer FFNN, Extreme learning machines, 2004, Huang et., advantage : take less training time, more efficiency, generalization
                   performance, universal approximation capabilities, [paper 1](https://www.sciencedirect.com/science/article/abs/pii/S0925231206000385), [paper 2](https://ieeexplore.ieee.org/document/6035797), [paper 3](https://www.researchgate.net/publication/6928613_Universal_Approximation_Using_Incremental_Constructive_Feedforward_Networks_With_Random_Hidden_Nodes))  
        
      <img src="https://editor.analyticsvidhya.com/uploads/98262Capture4.PNG" width="600" height="400"/>   
       
      - NN models can capture the non-linearity in data due to external factors.  
      - ELMs disadvantages : The main disadvantage of ELMs is that the stochasticity involved in the algorithm might sometimes lead to sub-optimal performance in terms of accuracy. To overcome this, one must properly tune the hyperparameters involved like the number of hidden nodes, the type of activation function, etc.  
      - **ELM Results** 
          - best conditions : 14 lag size, 110 hidden units, relu  
                                                   
       <img src="https://github.com/min0355/ai-novatus/blob/main/visualization/demand%20forcasting.png" width="600" height="400"/>  

  * **Everything you can do with a time series (kernel, Kaggle Master)** :    
    https://www.kaggle.com/code/thebrownviking20/everything-you-can-do-with-a-time-series#2.-Finance-and-statistics   
      - **Stationarity** : the behavior where the mean and standard deviation of the data changes over time, the data with such behavior is considered not stationary,[Kaggle notebook, refer to 3.4](https://www.kaggle.com/code/thebrownviking20/everything-you-can-do-with-a-time-series#2.-Finance-and-statistics)     
      - **Autocorrelation (ACF)** : behavior of the data where the data is correlated with itself in a different time period, [Kaggle notebook, refer to 2.8](https://www.kaggle.com/code/thebrownviking20/everything-you-can-do-with-a-time-series#2.-Finance-and-statistics), [ARIMA, Partial ACF](https://www.quora.com/What-is-the-difference-among-auto-correlation-partial-auto-correlation-and-inverse-auto-correlation-while-modelling-an-ARIMA-series)  

  * **LSTM-Based VAE-GAN Model** :  
      - paper : https://www.mdpi.com/1424-8220/20/13/3738/htm  
      - blog : https://anencore94.github.io/2020/10/28/lstm-based-vae-gan.html  
    <Img src="https://github.com/min0355/ai-novatus/blob/main/visualization/LSTM-BASED%20VAE-GAN.png" weight="855" height="779"/>      

  * **LSTM-Based Encoder-Decoder for Multi-sensor Anomaly Detection** :  
      - paper : https://paperswithcode.com/paper/lstm-based-encoder-decoder-for-multi-sensor  
      - git : https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection, https://github.com/KDD-OpenSource/DeepADoTS

  * **VAE-LSTM-for-Anomaly-Detection** :  
      - git : https://github.com/lin-shuyu/VAE-LSTM-for-anomaly-detection  

```source : @INPROCEEDINGS{VAE-LSTM-AD, author={S. {Lin} and R. {Clark} and R. {Birke} and S. {Schönborn} and N. {Trigoni} and S. {Roberts}}, booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, title={Anomaly Detection for Time Series Using VAE-LSTM Hybrid Model}, year={2020}}```      
   
### [reference] 
___
<!-- Quote -->
<!-- Bullet list -->
1. Machine Tools Movie (Applicable to all of these industrial facility) : https://www.youtube.com/watch?v=_Phpr92pqcw  
2. PHM : http://www.phm.or.kr/asso/greeting.php  
3. Mechanical Engineering and AI : https://drive.google.com/file/d/1kW5-q5ukxKtmzh4g63Xw-HoYIBfkdhPG/view   
4. INEEJI (KAIST Professor.choi, Start up, Industry automation / advanced, time series AI-based process optimization) : http://www.ineeji.com/   
5. Machine Tools, Adding Digital Value : https://www.arcweb.com/market-studies/machine-tools  
    
    ![](https://www.arcweb.com/sites/default/files/Images/research-images/machine-tools-market-2020-2025.jpg)  
    
5. Activation functions : https://deeesp.github.io/deep%20learning/DL-Activation-Functions/  
6. Anomaly Detection in time series sensor data : https://towardsdatascience.com/anomaly-detection-in-time-series-sensor-data-86fd52e62538   
7. Deep Learning Tutorials with PyTorch : https://pseudo-lab.github.io/Tutorial-Book/chapters/time-series/intro.html  
8. transformer (chapter.16) : https://wikidocs.net/book/2155  
9. Math for ML (Foreword) : https://github.com/min0355/postech_math  
     - Fledgling Composer    
     As ML(Machine learning) is applied to new domains, developers of ML need to develop new methods and extend exisiting algorithms. They are often researchers who need to understand the mathematical basis of ML and uncover relationships between different tasks.  
     This is similar to composers of music who, within the rules and structure of musical theory, create new and amazing pieces.   
     **There is a great need in society for new researchers who are able to propose and explore novel approaches for attacking the many chanllenges of learning from data.**  
10. TCN (Temporal Convolutional Networks) : https://hongl.tistory.com/253 ([Paper](https://arxiv.org/pdf/1803.01271.pdf)), https://github.com/NervanaSystems/aidc-2018-timeseries    
11. N-beats : https://joungheekim.github.io/2020/09/09/paper-review/    
12. Darts (colab) : https://colab.research.google.com/drive/1X-y0xkyo5nq72Q-prCwXdoTvLXC7x1Ae?usp=sharing  
13. VAE (Auto-Encoding Variational Bayes) : https://taeu.github.io/paper/deeplearning-paper-vae/, [kaggle_VAE](https://www.kaggle.com/code/irwanjunardi/pump-sensor-data-lstm-vae), [Reparameterization Trick](https://hulk89.github.io/machine%20learning/2017/11/20/reparametrization-trick/)  
14. GAN (Generative Adversarial Networks) : https://www.kaggle.com/code/ohseokkim/gan-how-does-gan-work   
15. Feature Importance : https://soohee410.github.io/iml_permutation_importance  
16. IAIpostech : [ML](https://iai.postech.ac.kr/teaching/machine-learning), [Deep Learning](https://iai.postech.ac.kr/teaching/deep-learning)  
17. AI + ME : https://sites.google.com/view/aiksme/ai-me   
18. tsfresh (package) : https://github.com/blue-yonder/tsfresh, https://tsfresh.readthedocs.io/en/latest/text/forecasting.html  
19. STFT : https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html, https://pinkwink.kr/1370, https://www.kaggle.com/code/amanooo/ingv-volcanic-basic-solution-stft  
20. model compiler for tree ensembles : [Treelite](https://treelite.readthedocs.io/en/latest/)    
21. Dimensionality Reduction : https://www.kaggle.com/code/ohseokkim/the-curse-of-dimensionality-dimension-reduction/notebook    


