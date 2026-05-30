import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Настройка страницы Streamlit
st.set_page_config(page_title="Мониторинги иқлими гармхона", layout="wide")

st.title("📊 Таҳлил ва пешгӯии ҳарорати ҳаво дар гармхона (Tair)")
st.markdown("Дар ин замима имконияти пешгӯии ҳарорат **бо назардошти вақти интихобшуда** нишон дода шудааст.")

# Функсия барои боркунӣ ва омодасозии маълумот
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('GreenhouseClimate1.csv')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    
    # 1. Специфические для времени признаки (Важно для диплома!)
    df['Hour'] = df['time'].dt.hour
    df['Minute'] = df['time'].dt.minute
    
    # 2. Интерполяцияи сенсорҳои бефосила
    continuous_cols = ['Tair', 'Tot_PAR', 'CO2air', 'VentLee']
    df[continuous_cols] = df[continuous_cols].interpolate(method='linear', limit_direction='both')
    
    # 3. Коркарди уставкаҳо (сутунҳои категориалӣ/зинагӣ)
    categorical_cols = [col for col in df.columns if '_sp' in col or '_vip' in col]
    if 't_heat_sp' not in categorical_cols and 't_heat_sp' in df.columns:
        categorical_cols.append('t_heat_sp')
        
    df[categorical_cols] = df[categorical_cols].fillna(method='ffill').fillna(method='bfill')
    
    # Тозакунии боқимондаҳо бо медиана
    df = df.fillna(df.median(numeric_only=True))
    
    # Сохтани лаг (қимати қаблӣ)
    df['Tair_lag1'] = df['Tair'].shift(1)
    df.dropna(subset=['Tair_lag1'], inplace=True)
    
    return df

try:
    df = load_and_clean_data()
    st.sidebar.success("Маълумот бомуваффақият бор карда шуд!")
    
    # ОБНОВЛЕННЫЙ СПИСОК ПРИЗНАКОВ: теперь сюда входят Hour и Minute
    features_final = ['Tair_lag1', 'Tot_PAR', 'CO2air', 'VentLee', 't_heat_sp', 'Hour', 'Minute']
    X = df[features_final]
    y = df['Tair']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # --- БЛОКИ 1: Омӯзиши модел ---
    st.sidebar.header("Танзимоти Random Forest")
    n_trees = st.sidebar.slider("Шумораи дарахтон (n_estimators)", min_value=10, max_value=200, value=100, step=10)
    
    @st.cache_resource
    def train_rf_model(X_tr, y_tr, trees):
        model = RandomForestRegressor(n_estimators=trees, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)
        return model
    
    with st.spinner("Модели Random Forest омӯзонида шуда истодааст... Лутфан мунтазир шавед."):
        model = train_rf_model(X_train, y_train, n_trees)
        y_pred = model.predict(X_test)
    
    # --- ИНТЕРАКТИВНОЕ ПРЕДСКАЗАНИЕ С ВЫБОРОМ ВРЕМЕНИ ---
    st.header("🔮 Пешгӯии фаврии ҳарорат бо назардошти вақт")
    st.markdown("Нишондиҳандаҳои ҷории сенсорҳо ва **вақти пешгӯиро** ворид кунед:")
    
    # Создаем 6 колонок для ввода параметров (добавили время)
    p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns(6)
    
    with p_col1:
        # Виджет выбора времени (выбираем часы и минуты)
        input_time = st.time_input("Вақти пешгӯӣ (Время)", datetime.time(12, 00))
        selected_hour = input_time.hour
        selected_minute = input_time.minute
        
    with p_col2:
        input_tair_lag = st.number_input("Ҳарорати қаблӣ (°C)", 
                                         min_value=0.0, max_value=50.0, 
                                         value=float(df['Tair'].mean()), step=0.5)
    with p_col3:
        input_par = st.slider("Равшанӣ (Tot_PAR)", 
                              min_value=0, max_value=1200, 
                              value=int(df['Tot_PAR'].median()))
    with p_col4:
        input_co2 = st.slider("Сатҳи CO2 (CO2air)", 
                              min_value=300, max_value=1200, 
                              value=int(df['CO2air'].median()))
    with p_col5:
        input_vent = st.slider("Вентилятсия % (VentLee)", 
                               min_value=0, max_value=100, 
                               value=int(df['VentLee'].median()))
    with p_col6:
        input_heat = st.number_input("Уставкаи гармидиҳӣ (°C)", 
                                     min_value=5.0, max_value=35.0, 
                                     value=float(df['t_heat_sp'].median()), step=0.5)
        
    # Сборка введенных данных (порядок строго как в features_final!)
    user_features = pd.DataFrame([[
        input_tair_lag, input_par, input_co2, input_vent, input_heat, selected_hour, selected_minute
    ]], columns=features_final)
    
    # Выполнение предсказания
    predicted_tair = model.predict(user_features)[0]
    
    # Красивый вывод
    st.subheader("Натиҷаи пешгӯии модел:")
    st.success(f"🌡️ **Ҳарорати пешгӯишаванда барои соати {input_time.strftime('%H:%M')}: {predicted_tair:.2f} °C**")
    
    st.markdown("---")
    
    # --- БЛОК 2: Намоиши метрикаҳои сифат ---
    st.header("1. Метрикаҳои самаранокии модел дар интихоби тестӣ")
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="MAE (Хатогии миёнаи мутлақ)", value=f"{mae:.3f} °C")
    col2.metric(label="RMSE (Хатогии квадратии миёна)", value=f"{rmse:.3f} °C")
    col3.metric(label="R² (Коэффисиенти детерминатсия)", value=f"{r2:.4f}")
    
    st.markdown("---")
    
    # --- БЛОК 3: Визуализатсияи натиҷаҳо ---
    st.header("2. Графики муқоисавии пешгӯӣ ва қиматҳои аслӣ")
    
    num_points = st.slider("Миқдори нуқтаҳо дар график барои намоиш", min_value=100, max_value=2000, value=500)
    num_points = min(num_points, len(y_test))
    
    time_test = df['time'].loc[y_test.index]
    
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time_test.iloc[-num_points:], y_test.values[-num_points:], label='Қиматҳои аслӣ (Actual)', color='black', alpha=0.6)
    ax.plot(time_test.iloc[-num_points:], y_pred[-num_points:], label='Пешгӯии модел (Predicted)', color='red', linestyle='--')
    ax.set_xlabel('Вақт')
    ax.set_ylabel('Ҳарорат, °C')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    st.pyplot(fig)
    
    # --- БЛОК 4: Муҳимияти нишонаҳо ---
    st.markdown("---")
    st.header("3. Муҳимияти нишонаҳо (Feature Importance)")
    
    features_tj = {
        'Tair_lag1': 'Ҳарорати қаблӣ (Tair_lag1)',
        'Tot_PAR': 'Равшанӣ/Радиатсия (Tot_PAR)',
        'CO2air': 'Сатҳи CO2 (CO2air)',
        'VentLee': 'Вентилятсия (VentLee)',
        't_heat_sp': 'Уставкаи гармидиҳӣ (t_heat_sp)',
        'Hour': 'Соати рӯз (Hour)',
        'Minute': 'Дақиқа (Minute)'
    }
    
    feat_importances = pd.Series(model.feature_importances_, index=features_final).sort_values(ascending=True)
    feat_importances.index = [features_tj.get(x, x) for x in feat_importances.index]
    
    fig_imp, ax_imp = plt.subplots(figsize=(10, 4))
    feat_importances.plot(kind='barh', color='teal', ax=ax_imp)
    ax_imp.set_title("Таъсири омилҳо ба пешгӯии ҳарорат")
    ax_imp.set_xlabel("Аҳамиятнокӣ")
    
    st.pyplot(fig_imp)
    
except FileNotFoundError:
    st.error("Хатогӣ: Файли 'GreenhouseClimate1.csv' ёфт нашуд.")