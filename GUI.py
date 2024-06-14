import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import squarify
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def createDf(df_product, df_transaction):

    # Adjust the main df
    df = pd.merge(df_transaction, df_product, on='productId', how='left')
    df['order_value'] = df['items'] * df['price']
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

    # Create the latest month df
    df_latest_month = df[df.Date > datetime(2015, 11, 30)].sort_values(by='Date')
    return df, df_latest_month

# Some constants
df_prod = pd.read_csv('Products_with_Prices.csv')
df_trans = pd.read_csv('Transactions.csv')

# Using menu
st.title("Hỗ trợ phân cụm khách hàng")
st.caption("Version 1.0")
st.markdown("---")

menu = ["Home", "Nhập dữ liệu", "Thống kê chung"]
menu.insert(2, menu.pop(2))
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':    
    st.subheader("Nếu có câu hỏi hay thắc mắc nào: [Link](https://csc.edu.vn)")
    st.subheader("Giới thiệu: ")
    st.write("Dựa vào dự án \"Phân cụm khách hàng\", trang web này sẽ hỗ trợ các chủ cửa hàng: ")
    st.markdown("- Tiện lợi theo dõi các giao dịch của cửa hàng")
    st.markdown("- Thông kê và đưa ra một góc nhìn tổng quát về giao dịch")
    st.markdown("- Phân cụm các nhóm khách hàng")
    st.markdown("---")
    st.caption("Phát triển bởi: Nguyễn Đức Nhật Minh và Nguyễn Dương Minh Hoàng")

elif choice == 'Nhập dữ liệu':
    st.subheader("Gợi ý điều khiển project 1: Sentiment Analysis")
    # Cho người dùng chọn nhập dữ liệu hoặc upload file
    type = st.radio("Chọn cách nhập liệu", options=["Upload file", "Đang phát triển..."])
    # Nếu người dùng chọn nhập dữ liệu vào text area
    if type == "Đang phát triển...":
        st.markdown("Xin thử lại ở các phiên bản sau...")

    elif type == "Upload file":
        st.subheader("Upload file")
        # Upload file
        product_csv = st.file_uploader("Dữ liệu hàng hóa: ")
        if product_csv is not None:
            # Đọc file dữ liệu
            df_prod = pd.read_csv(product_csv)
            st.write(df_prod)

        trans_csv = st.file_uploader("Dữ liệu giao dịch: ", type=["csv", "txt"])
        if trans_csv is not None:
            # Đọc file dữ liệu
            df_trans = pd.read_csv(trans_csv)
            st.write(df_trans)

elif choice == 'Thống kê chung':
    st.warning("Phần thống kê này cần được nhập đầy đủ dữ liệu.")
    st.header("Insight")

    # Code
    df, df_latest_month = createDf(df_prod, df_trans)

    tab1, tab2 = st.tabs(["Doanh thu", "Khách hàng"])
    
    with tab1:
        st.header("Doanh thu")

        # Code
        df_spending = pd.DataFrame(df.groupby(by='Member_number')['order_value'].sum().sort_values(ascending=False)).reset_index()
        top_3 = df_spending.head(3)

        st.subheader('Top 3 những khách hàng tiêu tiền nhiều nhất: ')
        st.markdown("- Mã Member: {} - Chi tiêu: {}".format(top_3.Member_number.iloc[0], top_3.order_value.iloc[0]))
        st.markdown("- Mã Member: {} - Chi tiêu: {}".format(top_3.Member_number.iloc[1], top_3.order_value.iloc[1]))
        st.markdown("- Mã Member: {} - Chi tiêu: {}".format(top_3.Member_number.iloc[2], top_3.order_value.iloc[2]))

        st.markdown('---')

        st.subheader('Theo tổng thể: ')
        st.markdown('Trung bình một người chi: {}'.format(round(df_spending['order_value'].mean(),2)))
        st.markdown('Tổng thu nhập: {}'.format(round(df.order_value.sum(),2)))

        st.markdown('---')
        
        st.subheader('Trong tháng gần đây nhất (12/2015): ')
        st.markdown('Trung bình một người chi: {}'.format(round(df_spending['order_value'].mean(),2)))
        st.markdown('Tổng thu nhập: {}'.format(round(df.order_value.sum(),2)))

        st.markdown('---')

        st.subheader("Biểu đồ thể hiện doanh thu:")
        # Code
        df['year_month'] = df['Date'].dt.to_period('M')
        # Group by year_month and sum the order values
        monthly_sums = df.groupby('year_month')['order_value'].sum().reset_index()
        monthly_sums['year_month'] = monthly_sums['year_month'].astype(str)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plotting using Seaborn
        sns.barplot(x='year_month', y='order_value', data=monthly_sums, palette='Blues_d', ax=ax)

        # Customize the plot
        ax.set_title('Tổng doanh thu mỗi tháng')
        ax.set_xlabel('Năm-Tháng')
        ax.set_ylabel('Doanh thu')
        ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability

        # Display the plot using st.pyplot()
        st.pyplot(fig)

    with tab2:
        st.header("Khách hàng - Tần suất")

        #Code
        df_freq = pd.DataFrame(df['Member_number'].value_counts().reset_index())
        df_freq.columns = ['Member_number', 'Count']

        st.subheader('Top 4 những khách hàng mua nhiều nhất: ')
        st.markdown("- Mã Member: {} - Số lần: {}".format(df_freq.Member_number.iloc[0], df_freq.Count.iloc[0]))
        st.markdown("- Mã Member: {} - Số lần: {}".format(df_freq.Member_number.iloc[1], df_freq.Count.iloc[1]))
        st.markdown("- Mã Member: {} - Số lần: {}".format(df_freq.Member_number.iloc[2], df_freq.Count.iloc[2]))


        st.subheader('Tổng cộng có: {} members'.format(df_freq.shape[0]))
        st.subheader('Trung bình một người mua: {} lần'.format(round(df_freq['Count'].mean())))

        st.markdown("---")
        st.header("Khách hàng - Ngừng thực hiện giao dịch")
        st.subheader("Biểu đồ thể hiện thời điểm ngừng giao dịch của khách hàng:")

        # Code
        df_inactive = pd.DataFrame(df['Member_number'].value_counts().reset_index())
        df_inactive.columns = ['Member_number', 'Count']
        df_inactive.sort_values(by = 'Member_number', inplace=True)
        df_inactive.reset_index(drop=True, inplace=True)
        
        inactive_since = []
        for i in df_inactive['Member_number']:
            df_ = df[df['Member_number']==i]
            inactive_since.append(df_['Date'].max())
        df_inactive['Inactive Since'] = inactive_since

        df_inactive['year_month'] = df_inactive['Inactive Since'].dt.to_period('M')
        # Group by year_month and sum the order values
        date_count = df_inactive.groupby('year_month')['Count'].count().reset_index()
        date_count['year_month'] = date_count['year_month'].astype(str)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plotting using Seaborn
        sns.barplot(x='year_month', y='Count', data=date_count, palette='Blues_d', ax=ax)

        # Customize the plot
        ax.set_title('Thời điểm mà khách hàng ngừng hoạt động, cùng với số lượng')
        ax.set_xlabel('Năm-Tháng')
        ax.set_ylabel('Số lượng khách hàng ngừng hoạt động')
        ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability

        # Display the plot using st.pyplot()
        st.pyplot(fig)
        