import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# chuẩn hóa dữ liệu trong file app
def prepare_data(data, features):
    data_scaled = data.copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled[features] = scaler.fit_transform(data[features])
    return data_scaled, scaler
# tạo  ra các chuỗi dữ liệu cho việc huấn luyện các mô hình học máy( chuỗi thời gian)
def create_sequences(data, seq_length, feature_col):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        Y.append(data[i + seq_length][feature_col])
    return np.array(X), np.array(Y)

# Đọc dữ liệu và tải mô hình
data = pd.read_csv('dulieuthunho.csv')
data['Ngày'] = pd.to_datetime(data['Ngày'])
features = ['Mở cửa', 'Đóng cửa', 'Cao nhất', 'Thấp nhất', 'Trung bình', 'GD khớp lệnh KL']
data = data[['Ngày', 'Mã CK'] + features]
data = data.sort_values(by=['Mã CK', 'Ngày'])
    
with open('models.pkl', 'rb') as file:
    models = pickle.load(file)

with open('scalers.pkl', 'rb') as file:
    scalers = pickle.load(file)
    

# Dự đoán giá đóng cửa tiếp theo
def predict_next_close(stock_data, seq_length, model, features, scaler):
    last_sequence = stock_data[features].values[-seq_length:]# Lấy chuỗi dữ liệu cuối cùng từ dữ liệu cổ phiếu, dựa trên seq_length
    last_sequence = scaler.transform(last_sequence))# Chuyển đổi dữ liệu cuối cùng về phạm vi chuẩn hóa bằng scaler
    last_sequence = np.expand_dims(last_sequence[:-1], axis=0) # Mở rộng chiều của chuỗi dữ liệu cuối cùng để phù hợp với đầu vào của mô hình LSTM
    predicted_price = model.predict(last_sequence)# Dự đoán giá đóng cửa tiếp theo bằng mô hình LSTM
    predicted_price = np.concatenate([predicted_price, np.zeros((predicted_price.shape[0], len(features)-1))], axis=1)# Tạo mảng trống với cùng số hàng như dự đoán, nhưng với số cột bằng số đặc trưng
    predicted_price = scaler.inverse_transform(predicted_price)# Chuyển đổi ngược dự đoán về phạm vi gốc bằng scaler
    return predicted_price[0][features.index('Đóng cửa')] # Trả về giá đóng cửa dự đoán tiếp theo

#  Tên ứng dụng Streamlit
st.title('Stock Price Prediction and Profit Calculation')

# Nhập số tiền đầu tư
investment = st.number_input('Enter the investment amount:', min_value=0.0, value=1000.0, step=100.0)
seq_length=20
# Tính toán lợi nhuận
profits = {}
for stock in data['Mã CK'].unique():
    stock_data = data[data['Mã CK'] == stock]
    current_price = stock_data['Đóng cửa'].values[-1]
    predicted_price = predict_next_close(stock_data, seq_length, models[stock], features, scalers[stock])# dự đoán theo hàm predict bên trên
    profit = (predicted_price - current_price) / current_price
    profits[stock] = profit

# Hiển thị 3 mã CK có lợi nhuận cao nhất
sorted_profits = sorted(profits.items(), key=lambda x: x[1], reverse=True)[:3]

st.write('Top 3 stocks with highest predicted profit:')
for stock, profit in sorted_profits:
    st.write(f'{stock}: {profit:.2%}')

# Hiển thị lợi nhuận dự kiến cho số tiền đầu tư
for stock, profit in sorted_profits:
    st.write(f'{stock}: Expected profit for {investment} VND is {investment * profit:.2f} VND')

# Hiển thị biểu đồ dự đoán cho 10 mã CK
for stock, _ in sorted_profits:
    stock_data = data[data['Mã CK'] == stock]
    stock_data_scaled, scaler = prepare_data(stock_data, features)

    X, Y = create_sequences(stock_data_scaled[features].values, seq_length, features.index('Đóng cửa'))
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

    model = models[stock]
    train_predict = model.predict(X)
    
    train_predict_transformed = np.zeros((train_predict.shape[0], len(features))) # Khởi tạo một mảng với kích thước (số mẫu, số đặc trưng) để chứa các dự đoán của mô hình.
    train_predict_transformed[:, features.index('Đóng cửa')] = train_predict.flatten()#Điền các dự đoán của mô hình vào cột tương ứng với giá đóng cửa trong mảng train_predict_transformed
    train_predict_transformed = scaler.inverse_transform(train_predict_transformed)[:, features.index('Đóng cửa')]#Chuyển đổi dự đoán từ phạm vi chuẩn hóa trở lại phạm vi giá gốc bằng cách sử dụng scaler, và chỉ giữ cột giá đóng cửa.

    plt.figure(figsize=(10, 6))
    actual_transformed = np.zeros((Y.shape[0], len(features))) #Khởi tạo một mảng với kích thước (số mẫu, số đặc trưng) để chứa giá thực tế.
    actual_transformed[:, features.index('Đóng cửa')] = Y.flatten()#Điền giá thực tế vào cột tương ứng với giá đóng cửa trong mảng actual_transformed.
    actual_transformed = scaler.inverse_transform(actual_transformed)[:, features.index('Đóng cửa')]#Chuyển đổi giá thực tế từ phạm vi chuẩn hóa trở lại phạm vi giá gốc bằng cách sử dụng scaler, và chỉ giữ cột giá đóng cửa.
    plt.plot(actual_transformed, label=f'Actual ({stock})', color='blue', alpha=0.6, linewidth=2)
    plt.plot(train_predict_transformed, label=f'Predicted ({stock})', color='orange', alpha=0.8, linewidth=2)
    plt.title(f'Predicted vs Actual Stock Prices ({stock})', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Stock Price', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    st.pyplot(plt)
    plt.clf()
