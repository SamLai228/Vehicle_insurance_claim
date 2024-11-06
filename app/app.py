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
        age_order = ['16 to 17', '18 to 20', '21 to 25', '26 to 30', 
                    '31 to 35', '36 to 40', '41 to 50', '51 to 65', 'over 65']
        categories = age_order
    else:
        categories = df_count.index

    # 重新索引並填充缺失值
    df_fraud_count = df_fraud_count.reindex(categories).fillna(0)
    df_count = df_count.reindex(categories).fillna(0)
    df_fraud_ratio = round(df_fraud_count/df_count, 2)

    # 設置通用的圖表樣式
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
        fig.update_traces(textposition='top center')
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False,
            margin=dict(t=50, l=50, r=50, b=100),
            yaxis=dict(range=[0, max(df_fraud_ratio.values) * 1.1]),
            **template['layout']
        )

    elif plot_type == 'hbar':  # horizontal bar plot
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
    # Create four columns for the plots
    col4, col5, col6 = st.columns(3)
    
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
    
    with col4:
        st.subheader('Month')
        fig2 = plot_fraud_ratio(filtered_df, 'Month', 'line')
        st.plotly_chart(fig2, use_container_width=True)

    with col5:
        st.subheader('Vehicle Age')
        fig3 = plot_fraud_ratio(filtered_df, 'AgeOfVehicle', 'bar')
        st.plotly_chart(fig3, use_container_width=True)
        
    with col6:
        st.subheader('Driver Rating')
        fig4 = plot_fraud_ratio(filtered_df, 'DriverRating', 'pie')
        st.plotly_chart(fig4, use_container_width=True)

    # Create second row with four columns
    col7, col8, col9 = st.columns(3)

    with col7:
        st.subheader('MaritalStatus')
        fig5 = plot_fraud_ratio(filtered_df, 'MaritalStatus', 'bar')
        st.plotly_chart(fig5, use_container_width=True)

    with col8:
        st.subheader('Number of Past Claims')
        fig7 = plot_fraud_ratio(filtered_df, 'PastNumberOfClaims', 'bar')
        st.plotly_chart(fig7, use_container_width=True)

    with col9:
        st.subheader('Policy Holder Age')
        fig8 = plot_fraud_ratio(filtered_df, 'AgeOfPolicyHolder', 'hbar')
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