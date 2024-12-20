import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import os
from pathlib import Path

# 在最開始設置頁面配置
st.set_page_config(
    page_title="Vehicle Insurance Fraud Analysis",
    page_icon="🚗",
    layout="wide",  
    initial_sidebar_state="expanded"
)

# 添加自定義 CSS
css = """
<style>
    /* 主要背景 */
    .stApp {
        background-color: #0E1117;
    }
    
    /* 文字和標題 */
    .stMarkdown, .stText, h1, h2, h3, p {
        color: white !important;
    }
    
    /* 側邊欄背景 */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    /* 側邊欄內容 */
    [data-testid="stSidebar"] > div:first-child {
        background-color: #262730;
    }
    
    /* 側邊欄文字 */
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* 卡片背景 */
    .css-12w0qpk {
        background-color: #262730;
    }
    
    /* 按鈕 */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    
    /* 輸入框 */
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    
    /* 選擇框 */
    .stSelectbox>div>div>select {
        background-color: #262730;
        color: white;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)


def plot_fraud_ratio(df, category_col, plot_type='bar'):
    df_fraud = df[df['FraudFound_P'] == 1]
    df_fraud_count = df_fraud[category_col].value_counts()
    df_count = df[category_col].value_counts()

    # 如果是月份相關的欄位，使用特定排序
    if category_col in ['Month', 'MonthClaimed']:
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        categories = month_order
    # 如果是年齡相關的欄位，使用特定排序    
    elif category_col == 'AgeOfPolicyHolder':
        age_order = ['under 30', '31 to 35', '36 to 40', '41 to 50', '51 and above']
        categories = age_order
    elif category_col == 'PastNumberOfClaims':
        NClaims_order = ['none', '1', '2 to 4', 'more than 4']
        categories = NClaims_order
    elif category_col == 'AgeOfVehicle':
        AgeV_order = ['4 years or less', '5 years', '6 years', '7 years', 'more than 7']
        categories = AgeV_order
    else:
        categories = df_count.index

    # 重新索引並填充缺失值
    df_fraud_count = df_fraud_count.reindex(categories).fillna(0)
    df_count = df_count.reindex(categories).fillna(0)
    df_fraud_ratio = round(df_fraud_count/df_count, 2)

    # 設定圖表樣式
    template = dict(
        layout=dict(
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117',
            font=dict(color='white')
        )
    )

    # 根據plot_type選擇不同的圖表類型
    if plot_type == 'pie':
        fig = px.pie(
            values=df_fraud_ratio.values,
            names=df_fraud_ratio.index,
            height=500,
            width=800,
            template='plotly_dark'
        )
        fig.update_traces(textposition='inside', textinfo='label+percent', textfont=dict(color='white'))
        fig.update_layout(
            showlegend=True,
            margin=dict(t=50, l=50, r=50, b=50),
            **template['layout']
        )

    elif plot_type == 'line':
        fig = px.line(
            x=df_fraud_ratio.index,
            y=df_fraud_ratio.values,
            labels={'x': category_col, 'y': 'Fraud Ratio'},
            markers=True,
            height=500,
            width=800,
            template='plotly_dark'
        )
        fig.update_traces(line=dict(width=6))  
        fig.update_traces(textposition='top center')
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False,
            margin=dict(t=50, l=50, r=50, b=100),
            yaxis=dict(range=[0, max(df_fraud_ratio.values) * 1.1]),
            **template['layout']
        )

    elif plot_type == 'hbar':  
        fig = px.bar(
            x=df_fraud_ratio.values,
            y=df_fraud_ratio.index,
            labels={'x': 'Fraud Ratio', 'y': category_col},
            text=df_fraud_ratio.values,
            height=500,
            width=800,
            orientation='h',
            template='plotly_dark'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            showlegend=False,
            margin=dict(t=50, l=150, r=50, b=50),
            **template['layout']
        )

    else:  # default bar plot
        fig = px.bar(
            x=df_fraud_ratio.index,
            y=df_fraud_ratio.values,
            labels={'x': category_col, 'y': 'Fraud Ratio'},
            text=df_fraud_ratio.values,
            height=500,
            width=800,
            template='plotly_dark'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False,
            margin=dict(t=50, l=50, r=50, b=100),
            **template['layout']
        )

    return fig

# Set page title
st.title('Vehicle Insurance Fraud Analysis')

# Load data
# 獲取當前文件的目錄
current_dir = os.path.dirname(os.path.abspath(__file__))

# 讀取數據
df = pd.read_csv(os.path.join(current_dir, 'data', 'df_for_visualisation.csv'))

# Add sidebar for filters
with st.sidebar:
    st.header("Filters")
    
    selected_year = st.selectbox(
        "Select Year",
        options=['All'] + sorted(df['Year'].unique().tolist()),
        index=0
    )
    
    selected_repnumber = st.selectbox(
        "Select RepNumber", 
        options=['All'] + sorted(df['RepNumber'].unique().tolist()),
        index=0
    )
    
    selected_make = st.multiselect(
        "Select Brand",
        options=['All'] + sorted(df['Make'].unique().tolist()),
        default=['All']
    )
    
    selected_fault = st.selectbox(
        "Select Fault",
        options=['All'] + sorted(df['Fault'].unique().tolist()),
        index=0
    )

# Filter data based on selected options
filtered_df = df.copy()
if selected_year != 'All':
    filtered_df = filtered_df[filtered_df['Year'] == selected_year]
if selected_repnumber != 'All':
    filtered_df = filtered_df[filtered_df['RepNumber'] == selected_repnumber]
if 'All' not in selected_make:  # If 'All' is not selected
    filtered_df = filtered_df[filtered_df['Make'].isin(selected_make)]
if selected_fault != 'All':
    filtered_df = filtered_df[filtered_df['Fault'] == selected_fault]

# Create KPI metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <style>
        [data-testid="stMetric"] {
            border: 1px solid white;
            border-radius: 5px;
            padding: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
    st.metric(
        label="Total Claim Number",
        value=f"{len(filtered_df):,}",
        delta=None,
        label_visibility="visible",
        help=None,
        delta_color="normal"
    )

with col2:
    st.markdown("""
        <style>
        [data-testid="stMetric"] {
            border: 1px solid white;
            border-radius: 5px;
            padding: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
    fraud_count = len(filtered_df[filtered_df['FraudFound_P'] == 1])
    st.metric(
        label="Fraud Number", 
        value=f"{fraud_count:,}",
        delta=None,
        label_visibility="visible",
        help=None,
        delta_color="normal"
    )

with col3:
    st.markdown("""
        <style>
        [data-testid="stMetric"] {
            border: 1px solid white;
            border-radius: 5px;
            padding: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
    fraud_rate = fraud_count / len(filtered_df)
    st.metric(
        label="Fraud Rate",
        value=f"{fraud_rate:.1%}",
        delta=None,
        label_visibility="visible",
        help=None,
        delta_color="normal"
    )

st.markdown("""
<style>
[data-testid="stMetricLabel"] {
    color: white !important;
}

[data-testid="stMetricValue"] {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("---")  # Add a horizontal line for separation

# Create tabs
tab1, tab2 = st.tabs(["Data Visualization", "Model Prediction"])

with tab1:
    # Create three columns for the plots
    col4, col5, col6 = st.columns([1.5,1,1])
    
    with col4:
        st.subheader('MonthClaimed')
        fig2 = plot_fraud_ratio(filtered_df, 'MonthClaimed', 'line')
        st.plotly_chart(fig2, use_container_width=True)
        
    with col5:
        st.subheader('VehiclePrice')
        fig4 = plot_fraud_ratio(filtered_df, 'VehiclePrice', 'bar')
        st.plotly_chart(fig4, use_container_width=True)

    with col6:
        st.subheader('Vehicle Age')
        fig5 = plot_fraud_ratio(filtered_df, 'AgeOfVehicle', 'bar')
        st.plotly_chart(fig5, use_container_width=True)

    # Create second row with four columns
    col8, col9, col10 = st.columns([2,1,1])

    with col8:
        st.subheader('Policy Holder Age')
        fig6 = plot_fraud_ratio(filtered_df, 'AgeOfPolicyHolder', 'hbar')
        st.plotly_chart(fig6, use_container_width=True)

    with col9:
        st.subheader('Number of Past Claims')
        fig7 = plot_fraud_ratio(filtered_df, 'PastNumberOfClaims', 'bar')
        st.plotly_chart(fig7, use_container_width=True)

    with col10:
        st.subheader('BasePolicy')
        fig8 = plot_fraud_ratio(filtered_df, 'BasePolicy', 'bar')
        st.plotly_chart(fig8, use_container_width=True)

    # Create risk groups table
    st.subheader('高風險組合')
    risk_groups = filtered_df.groupby(['AgeOfPolicyHolder', 'BasePolicy', 'VehiclePrice']).agg(
        total_claims=('FraudFound_P', 'count'),
        fraud_claims=('FraudFound_P', 'sum')
    ).reset_index()

    # Calculate fraud rate
    risk_groups['Fraud_Rate'] = risk_groups['fraud_claims'] / risk_groups['total_claims']

    # Sort by fraud rate in descending order
    risk_groups = risk_groups.sort_values('Fraud_Rate', ascending=False)

    # Format fraud rate as percentage
    risk_groups['Fraud_Rate'] = risk_groups['Fraud_Rate'].map('{:.1%}'.format)

    # Display the table without index
    st.dataframe(risk_groups.head(10).set_index('AgeOfPolicyHolder').reset_index(), use_container_width=True, hide_index=True)

with tab2:
    # Load the trained model
    import pickle

    # 獲取當前文件的目錄
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 載入模型
    model_path = os.path.join(current_dir, 'models', 'xgboost_new2')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Add single prediction section
    st.subheader("Single Prediction")
    
    # Create input fields for features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fault = st.selectbox('Fault', ['Policy Holder', 'Third Party'])
        police_report = st.selectbox('Police Report Filed', ['Yes', 'No'])
        policyholder_age = st.selectbox('Age of Policy Holder', ['under 30', '31 to 35', '36 to 40', '41 to 50', '51 and above'])
        policy_type = st.selectbox('Policy Type', ['Sport - Liability', 'Sport - Collision', 'Sport - All Perils', 'Sedan - Liability',
                                                   'Sedan - Collision', 'Sedan - All Perils', 'Utility - Liability',
                                                   'Utility - Collision', 'Utility - All Perils'])
    
    with col2:
        deductible = st.selectbox('Deductible', [300, 400, 500, 600, 700])
        month = st.selectbox('Month', list(range(1, 13)))
        week_of_month = st.selectbox('Week of Month', list(range(1, 6)))
        day_of_week = st.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                                   'Friday', 'Saturday', 'Sunday'])
    
    with col3:
        year = st.selectbox('Year', [1994, 1995, 1996])
        month_claimed = st.selectbox('Month Claimed', list(range(1, 13)))
        week_of_month_claimed = st.selectbox('Week of Month Claimed', list(range(1, 6)))

    if st.button('Predict'):
        # Create a dataframe from inputs
        input_dict = {
            'AgeOfPolicyHolder_41 to 50': [1 if policyholder_age == '41 to 50' else 0],
            'Fault': [0 if fault == 'Policy Holder' else 1],
            'PoliceReportFiled': [1 if police_report == 'Yes' else 0],
            'PolicyType_Sedan - Collision': [1 if policy_type == 'Sedan - Collision' else 0],
            'PolicyType_Sedan - Liability': [1 if policy_type == 'Sedan - Liability' else 0],
            'HiskRisk_DeductibleAMT': [1 if deductible in [300, 500] else 0],
            'Month_Dec': [1 if month == 12 else 0],
            'Month_Mar': [1 if month == 3 else 0],
            'WeekOfMonth_2': [1 if week_of_month == 2 else 0],
            'WeekOfMonth_3': [1 if week_of_month == 3 else 0],
            'WeekOfMonth_4': [1 if week_of_month == 4 else 0],
            'DayOfWeek_Sunday': [1 if day_of_week == 'Sunday' else 0],
            'Year_1995': [1 if year == 1995 else 0],
            'Year_1996': [1 if year == 1996 else 0],
            'MonthClaimed_Aug': [1 if month_claimed == 8 else 0],
            'MonthClaimed_Dec': [1 if month_claimed == 12 else 0],
            'MonthClaimed_Feb': [1 if month_claimed == 2 else 0],
            'MonthClaimed_Jan': [1 if month_claimed == 1 else 0],
            'MonthClaimed_Jun': [1 if month_claimed == 6 else 0],
            'MonthClaimed_Nov': [1 if month_claimed == 11 else 0],
            'WeekOfMonthClaimed_2': [1 if week_of_month_claimed == 2 else 0]
        }
        
        single_input = pd.DataFrame(input_dict)
        
        # Make prediction
        prediction = model.predict(single_input)
        probability = model.predict_proba(single_input)
        
        # Display results
        st.write("Prediction Result:")
        if prediction[0] == 1:
            st.error(f"Fraud Detected! Probability: {probability[0][1]:.2%}")
        else:
            st.success(f"No Fraud Detected. Probability: {probability[0][1]:.2%}")

    # Add batch prediction section
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file for prediction", type="csv")

    if uploaded_file is not None:
        # Read and display the uploaded data
        input_data = pd.read_csv(uploaded_file)
        
        # Rename 'Unnamed: 0' to 'ID' if it exists
        if 'Unnamed: 0' in input_data.columns:
            input_data = input_data.rename(columns={'Unnamed: 0': 'ID'})
            input_data.set_index('ID', inplace=True)
        
        st.write("Preview of uploaded data:")
        st.write(input_data.head())
        
        if st.button('Generate Predictions'):
            # Make predictions
            predictions = model.predict(input_data)
            probabilities = model.predict_proba(input_data)
            
            # Add predictions and probabilities to the dataframe
            input_data['Predicted_Fraud'] = predictions
            input_data['Fraud_Probability'] = probabilities[:, 1]  # Probability of class 1 (Fraud)
            
            # Sort by fraud probability in descending order
            input_data = input_data.sort_values('Fraud_Probability', ascending=False)
            
            st.write("Prediction Results:")
            st.write(input_data)
            
            # Add download button for results
            csv = input_data.to_csv(index=False)
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv",
                type="primary"  # This sets the button color to green
            )
