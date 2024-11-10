# 汽車保險詐欺預測儀表板

## 專案概述
詐欺案件每年為保險公司帶來巨大的經濟損失。本專案的目的是建立一個汽車險詐欺預測儀表板，協助公司提升調查效率並減少經濟損失。透過模型評估每個案件的風險等級，低風險的案件自動進入核賠流程，理賠人員則可以把時間花在高風險的案件，從而提升整體理賠作業的效率與成效。  

此專案中有三個主要資料夾: data, notebooks 與 app
## data資料夾
raw資料夾裡有原始資料，processed資料夾裡則是 Data Preprocessing 處理完的資料

資料來源：https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection/data

## notebooks 資料夾
notebooks裡有 3 個檔案，
1. EDA_insurance.ipynb 
2. Preprocessing.ipynb 
3. Modeling .ipynb

## app 資料夾
app.py 將Modeling.ipynb裡訓練好的模型及Plotly圖透過 Streamlit 製作成一個 dashboard
