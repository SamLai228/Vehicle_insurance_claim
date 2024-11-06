import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# 在最開始設置頁面配置
st.set_page_config(
    page_title="Vehicle Insurance Fraud Analysis",
    page_icon="🚗",
    layout="wide",  # 使用寬屏布局
    initial_sidebar_state="expanded"
)

def plot_fraud_ratio_bar(df, category_col):
    df_fraud = df[df['FraudFound_P'] == 1]
    df_fraud_count = df_fraud[category_col].value_counts()
    df_count = df[category_col].value_counts()

    # 如果是月份相關的欄位，使用特定排序
    if category_col in ['Month', 'MonthClaimed']:
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        categories = month_order  
    else:
        categories = df_count.index  

    # 重新索引並填充缺失值
    df_fraud_count = df_fraud_count.reindex(categories).fillna(0)
    df_count = df_count.reindex(categories).fillna(0)
    df_fraud_ratio = round(df_fraud_count/df_count, 2)  

    # 創建圖表
    fig = px.bar(
        x=df_fraud_ratio.index,
        y=df_fraud_ratio.values,
        title=f'Fraud Ratio by {category_col}',
        labels={'x': category_col, 'y': 'Fraud Ratio'},
        text=df_fraud_ratio.values,
        height=500,  # 增加高度
        width=800    # 增加寬度
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(
        title_x=0.5,
        xaxis_tickangle=-45,
        showlegend=False,
        margin=dict(t=50, l=50, r=50, b=100)  # 調整邊距
    )

    return fig

def plot_fraud_ratio_pie(df, category_col):
    df_fraud = df[df['FraudFound_P'] == 1]
    df_fraud_count = df_fraud[category_col].value_counts()
    df_count = df[category_col].value_counts()

    # 計算詐欺比率
    df_fraud_ratio = round(df_fraud_count/df_count, 2)

    # 創建圖表
    fig = px.pie(
        values=df_fraud_ratio.values,
        names=df_fraud_ratio.index,
        title=f'Fraud Ratio by {category_col}',
        height=500,  # 增加高度
        width=800    # 增加寬度
    )

    fig.update_traces(textposition='outside', textinfo='label+percent')
    fig.update_layout(
        title_x=0.5,
        showlegend=True,
        margin=dict(t=50, l=50, r=50, b=50)  # 調整邊距
    )

    return fig

def plot_fraud_ratio_line(df, category_col):
    df_fraud = df[df['FraudFound_P'] == 1]
    df_fraud_count = df_fraud[category_col].value_counts()
    df_count = df[category_col].value_counts()

    # 如果是月份相關的欄位，使用特定排序
    if category_col in ['Month', 'MonthClaimed']:
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        categories = month_order
    else:
        categories = df_count.index

    # 重新索引並填充缺失值
    df_fraud_count = df_fraud_count.reindex(categories).fillna(0)
    df_count = df_count.reindex(categories).fillna(0)
    df_fraud_ratio = round(df_fraud_count/df_count, 2)

    # 創建圖表
    fig = px.line(
        x=df_fraud_ratio.index,
        y=df_fraud_ratio.values,
        title=f'Fraud Ratio Trend by {category_col}',
        labels={'x': category_col, 'y': 'Fraud Ratio'},
        markers=True,
        height=500,  # 增加高度
        width=800    # 增加寬度
    )

    fig.update_traces(textposition='top center')
    fig.update_layout(
        title_x=0.5,
        xaxis_tickangle=-45,
        showlegend=False,
        margin=dict(t=50, l=50, r=50, b=100),  # 調整邊距
        yaxis=dict(range=[0, max(df_fraud_ratio.values) * 1.1])  # 設置y軸從0開始
    )

    return fig

# Set page title
st.title('Vehicle Insurance Fraud Analysis')

# Load data
df = pd.read_csv('../data/processed/df_for_visualisation.csv')

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
    
    selected_make = st.selectbox(
        "Select Make",
        options=['All'] + sorted(df['Make'].unique().tolist()),
        index=0
    )

# Filter data based on selected options
filtered_df = df.copy()
if selected_year != 'All':
    filtered_df = filtered_df[filtered_df['Year'] == selected_year]
if selected_repnumber != 'All':
    filtered_df = filtered_df[filtered_df['RepNumber'] == selected_repnumber]
if selected_make != 'All':
    filtered_df = filtered_df[filtered_df['Make'] == selected_make]

# Create tabs
tab1, tab2 = st.tabs(["Data Visualization", "Model Prediction"])

with tab1:
    # Create four columns for the plots
    col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     # Create fraud distribution plot
    #     st.subheader('Distribution of Fraud Cases')
    #     fraud_dist = filtered_df['FraudFound_P'].value_counts(normalize=True)
    #     fig = px.pie(
    #         values=fraud_dist.values,
    #         names=fraud_dist.index,
    #         labels={'names': 'Fraud Found', 'values': 'Percentage'},
    #         title='Distribution of Fraud Cases'
    #     )

    #     fig.update_traces(textposition='inside', textinfo='percent+label')
    #     fig.update_layout(
    #         title_x=0.5,
    #         showlegend=True
    #     )

    #     # Display plot in Streamlit
    #     st.plotly_chart(fig)
    
    with col1:
        st.subheader('Fraud Ratio by Month')
        fig2 = plot_fraud_ratio_line(filtered_df, 'Month')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader('Fraud Ratio by Vehicle Age')
        fig3 = plot_fraud_ratio_bar(filtered_df, 'AgeOfVehicle')
        st.plotly_chart(fig3, use_container_width=True)
        
    with col3:
        st.subheader('Fraud Ratio by Policy Holder Age')
        fig4 = plot_fraud_ratio_bar(filtered_df, 'AgeOfPolicyHolder')
        st.plotly_chart(fig4, use_container_width=True)

    # Create second row with four columns
    col4, col5, col6 = st.columns(3)

    with col4:
        st.subheader('Fraud Ratio by Fault')
        fig5 = plot_fraud_ratio_pie(filtered_df, 'Fault')
        st.plotly_chart(fig5, use_container_width=True)

    with col5:
        st.subheader('Fraud Ratio by Past Claims')
        fig7 = plot_fraud_ratio_bar(filtered_df, 'PastNumberOfClaims')
        st.plotly_chart(fig7, use_container_width=True)

    with col6:
        st.subheader('Fraud Ratio by Driver Rating')
        fig8 = plot_fraud_ratio_bar(filtered_df, 'DriverRating')
        st.plotly_chart(fig8, use_container_width=True)

with tab2:
    # Load the trained model
    import pickle

    with open('../models/balanced_random_forest.pkl', 'rb') as f:
        model = pickle.load(f)

    # Add file uploader
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
                mime="text/csv"
            )
