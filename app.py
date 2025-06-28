import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve
)

# Đặt tên file cố định (cùng cấp app.py)
LABEL_ENCODER_PATH = "label_encoder.pkl"
ONEHOT_ENCODER_PATH = "onehot_encoder.pkl"
TFIDF_VECTORIZER_PATH = "tfidf_vectorizer.pkl"
XGB_MODEL_PATH = "xgb_model.pkl"
COMPANY_FILE = "Overview_Companies.xlsx"
REVIEW_FILE = "Reviews.xlsx"
OVERVIEW_REVIEW_FILE = "Overview_Reviews.xlsx"

# PAGE CONFIG (nên để đầu file)
st.set_page_config(page_title="Company Recommendation & Candidate Classification", layout="wide")

# ================== LOAD MODEL & ENCODER ==================
@st.cache_resource
def load_all_models():
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    onehot_encoder = joblib.load(ONEHOT_ENCODER_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    return label_encoder, onehot_encoder, tfidf_vectorizer, xgb_model

label_encoder, onehot_encoder, tfidf_vectorizer, xgb_model = load_all_models()

# ================== LOAD DATA ==================
@st.cache_data
def load_data():
    df_companies = pd.read_excel(COMPANY_FILE)
    df_reviews = pd.read_excel(REVIEW_FILE)
    df_overview_reviews = pd.read_excel(OVERVIEW_REVIEW_FILE)
    return df_companies, df_reviews, df_overview_reviews

df_companies, df_reviews, df_overview_reviews = load_data()

# ================== SIDEBAR MENU ==================
st.sidebar.title("Project 02")
st.sidebar.caption("Team: Nguyen Quynh Oanh Thao - Nguyen Le Minh Quang")
menu = ["Trang chủ", "Dự đoán công ty", "Phân tích mô hình"]
choice = st.sidebar.radio("Menu", menu)

# ================== HOME ==================
if choice == "Trang chủ":
    st.title("Project 02 - Company Recommendation & Candidate Classification")
    st.markdown("""
    Ứng dụng hỗ trợ:
    - Đề xuất công ty phù hợp dựa vào nội dung và dữ liệu đánh giá.
    - Phân tích khả năng "Recommend" của nhân viên/ứng viên cho công ty.
    """)
    st.info("Chọn các tab bên trái để trải nghiệm các chức năng dự đoán!")

# ================== DỰ ĐOÁN CÔNG TY ==================
elif choice == "Dự đoán công ty":
    st.header("📝 Dự đoán Recommend cho tên công ty")
    company_name_list = df_companies["Company Name"].dropna().unique().tolist()
    selected_company = st.selectbox("Chọn tên công ty", sorted(company_name_list))
    if st.button("Dự đoán Recommend?"):
        try:
            selected_info = df_companies[df_companies["Company Name"] == selected_company].iloc[0]
            company_type = selected_info["Company Type"]
            company_size = selected_info["Company size"]

            # One-hot encode categorical
            cat_features = pd.DataFrame([[company_type, company_size]], columns=["Company Type", "Company size"])
            cat_encoded = onehot_encoder.transform(cat_features)

            # Chuẩn bị input cho model (vector 1010 chiều, cat features nằm cuối)
            num_total_features = 1010
            num_cat_features = cat_encoded.shape[1]
            final_features = np.zeros((1, num_total_features))
            final_features[0, -num_cat_features:] = cat_encoded

            # Predict
            proba = xgb_model.predict_proba(final_features)[0][1]
            prediction = label_encoder.inverse_transform([int(proba >= 0.5)])[0]
            st.subheader("🔍 Kết quả")
            st.write(f"**Xác suất Recommend:** {proba:.2%}")
            if prediction == "Yes":
                st.success(f"✨ Nhân viên/ứng viên có xu hướng **RECOMMEND** công ty này.")
            else:
                st.warning(f"⚠️ Nhân viên/ứng viên có xu hướng **KHÔNG RECOMMEND** công ty này.")
        except Exception as e:
            st.error(f"Lỗi: {e}")

# ================== PHÂN TÍCH MÔ HÌNH ==================
elif choice == "Phân tích mô hình":
    st.header("📊 Phân tích hiệu suất mô hình XGBoost")
    # Kết hợp review với các thông tin cần thiết
    df = df_reviews.merge(
        df_overview_reviews[["id", "Overall rating"]].rename(columns={"id": "company_id"}),
        left_on="id", right_on="company_id", how="left"
    ).merge(
        df_companies[["company_id", "Company Name", "Company Type", "Company size"]],
        on="company_id", how="left"
    )
    # Chuẩn bị dữ liệu cho model
    categorical_cols = ['Company Type', 'Company size']
    features = df[['What I liked', 'Rating', 'Company Type', 'Company size', 'Overall rating']].copy()
    features = features.fillna("")
    # Kết hợp text
    features['combined_text'] = features['What I liked'].astype(str)
    text_features = tfidf_vectorizer.transform(features['combined_text'])
    # One-hot encode
    cat_features = features[categorical_cols].fillna("Khác")
    cat_encoded = onehot_encoder.transform(cat_features)
    numeric_features = features[['Rating', 'Overall rating']].fillna(0).to_numpy()
    # Kết hợp all features
    from scipy.sparse import hstack
    X = hstack([text_features, numeric_features, cat_encoded])
    y = df['Recommend?'].fillna("No").to_numpy()

    # Chia test-train
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predict và đánh giá
    y_pred = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="Yes")
    rec = recall_score(y_test, y_pred, pos_label="Yes")
    f1 = f1_score(y_test, y_pred, pos_label="Yes")
    auc = roc_auc_score((y_test == "Yes").astype(int), y_proba)

    st.write("### 📈 Hiệu suất mô hình trên tập kiểm thử:")
    st.write(f"- **Accuracy:** {acc:.4f}")
    st.write(f"- **Precision:** {prec:.4f}")
    st.write(f"- **Recall:** {rec:.4f}")
    st.write(f"- **F1-score:** {f1:.4f}")
    st.write(f"- **AUC:** {auc:.4f}")

    # Hiển thị confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=["Yes", "No"])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Yes", "No"], yticklabels=["Yes", "No"], ax=ax)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    st.pyplot(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve((y_test == "Yes").astype(int), y_proba)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    st.pyplot(fig2)

    st.markdown("> Dữ liệu và model sử dụng hoàn toàn offline, không gửi dữ liệu đi đâu.")

# ========== HẾT ==========
