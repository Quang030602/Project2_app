import os
os.system('pip install underthesea')
# Step 2: Write the Streamlit app to a Python file
import streamlit as st

st.title("Project 02 - Company Recommendation & Candidate Classification")
st.caption("Team: Nguyen Quynh Oanh Thao - Nguyen Le Minh Quang")

menu = ["Home", "About"]
choice = st.sidebar.selectbox('Menu', menu)


if choice == 'Home':
   st.subheader("Streamlit From Colab")
elif choice == 'About':
   st.subheader("Requirements info")

st.set_page_config(page_title="Project 02", layout="wide")

# Tabs for Topic 1 and Topic 2
tab1, tab2 = st.tabs(["🔍 Topic 1: Company Recommendation", "🧠 Topic 2: Candidate Classification"])

with tab1:
    st.header("Topic 1: Content-Based Company Recommendation System")


with tab1:
    import pandas as pd
    import numpy as np
    import streamlit as st
    import re
    import gensim
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics.pairwise import cosine_similarity
    from gensim import models as gensim_models, corpora, similarities
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.sparse import hstack
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    # --- Text Preprocessing ---
    def clean_tokens(tokens):
        cleaned = [re.sub(r'\d+', '', word) for word in tokens]
        return [word.lower() for word in cleaned if word not in ['', ' ', ',', '.', '-', ':', '?', '%', '(', ')', '+', '/', 'g', 'ml']]

    stop_words = set([
        "a", "an", "the", "in", "on", "at", "to", "from", "by", "of", "with", "and", "but", "or", "for", "nor", "so", "yet",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "be", "have", "do", "does", "did",
        "was", "were", "will", "would", "shall", "should", "may", "might", "can", "could", "must",
        "that", "this", "which", "what", "their", "these", "those", "https", "www"
    ])

    def remove_stopwords(tokens):
        return [word for word in tokens if word not in stop_words]

    # --- Load and Prepare Data ---
    @st.cache_data
    def load_and_process_data():
        df = pd.read_excel("Overview_Companies.xlsx")
        df = df[['Company Name', 'Company overview']].dropna().copy()
        df['tokens'] = df['Company overview'].apply(lambda x: gensim.utils.simple_preprocess(x))
        df['tokens_cleaned'] = df['tokens'].apply(clean_tokens)
        df['tokens_final'] = df['tokens_cleaned'].apply(remove_stopwords)
        df = df[df['tokens_final'].str.len() > 0].copy()
        df['joined_tokens'] = df['tokens_final'].apply(lambda tokens: ' '.join(tokens))
        return df

    df = load_and_process_data()

    # --- ML Classification ---
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), sublinear_tf=True, stop_words='english', min_df=2, max_df=0.8, norm='l2')
    X = vectorizer.fit_transform(df['joined_tokens'])

    # Dummy label for now: use KMeans or known labels
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['label'] = kmeans.fit_predict(X)
    label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    df['label'] = df['label'].map(label_map)

    # Encode & split
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # --- Gensim Similarity Setup ---
    dictionary = corpora.Dictionary(df['tokens_final'])
    corpus = [dictionary.doc2bow(text) for text in df['tokens_final']]
    tfidf_model = gensim_models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]
    index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=len(dictionary))

    # --- Charts -----
    # ---- Define models BEFORE using them ----
    models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced'),
    "Support Vector Machine": SVC(probability=True, class_weight='balanced')}
    # ---- Recalculate cosine feature and split ----
    cosine_sim = cosine_similarity(X, X)
    ref_sim = cosine_sim[0].reshape(-1, 1)
    X_with_sim = hstack([X, ref_sim])
    X_train, X_test, y_train, y_test = train_test_split(
    X_with_sim, y, test_size=0.5, random_state=42, stratify=y)

    # ---- Evaluate All Models ----
    results = {}
    for name, model in models.items():
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
      recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
      f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
      accuracy = accuracy_score(y_test, y_pred)
      results[name] = {
          "Accuracy": accuracy,
          "Precision": precision,
          "Recall": recall,
          "F1-score": f1
    }

    # ---- Display Performance Table ----
    st.write("## 📊 Model Performance Summary")
    st.dataframe(pd.DataFrame(results).T.sort_values(by="F1-score", ascending=False))

    # ---- Confusion Matrix Visualization ----
    st.write("## 🔍 Confusion Matrices")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
      y_pred = model.predict(X_test)
      cm = confusion_matrix(y_test, y_pred)
      sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
      axes[idx].set_title(f"{name}")
      axes[idx].set_xlabel("Predicted")
      axes[idx].set_ylabel("Actual")

    # Remove extra axes
    for j in range(len(models), len(axes)):
      fig.delaxes(axes[j])

    st.pyplot(fig)

    # --- Show selected model ---
    st.markdown("## ✅ Selected Model for Recommendation")
    st.write("""We are using **Logistic Regression** as the primary model for predicting company fit and driving the similarity search logic below, based on its strong F1-score and overall performance.
""")


    # --- Streamlit UI ---
    st.subheader("🔍 Explore 'High' Fit Companies (Gensim TF-IDF Similarity)")

    input_text = st.text_area("Enter your company description or summary:")
    if st.button("Find similar 'High' fit companies"):
        if not input_text.strip():
            st.warning("Please enter some text.")
        else:
            input_tokens = gensim.utils.simple_preprocess(input_text)
            input_tokens_clean = remove_stopwords(clean_tokens(input_tokens))
            input_bow = dictionary.doc2bow(input_tokens_clean)

            sims = index[tfidf_model[input_bow]]
            ranked = sorted(enumerate(sims), key=lambda x: -x[1])

            st.write("### Top Similar Companies (label = High)")
            count = 0
            for idx, score in ranked:
                if df.iloc[idx]['label'] == 'High':
                    st.markdown(f"#### 🏷️ {df.iloc[idx]['Company Name']}")
                    st.markdown(f"- **Similarity Score:** `{score:.2f}`")
                    st.markdown(f"> {df.iloc[idx]['Company overview']}")
                    st.markdown("---")
                    count += 1
                if count >= 5:
                    break

with tab2:
    st.header("Topic 2: Candidate Fit Classification")


    import pandas as pd
    import numpy as np
    import streamlit as st
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re
    import openpyxl
    from underthesea import word_tokenize

    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, roc_auc_score
    from imblearn.over_sampling import SMOTE
    from collections import Counter


    # --- Load data from Google Drive local paths ---
    @st.cache_data
    def load_review_data():
        reviews = pd.read_excel("Reviews.xlsx")
        overview_reviews = pd.read_excel("Overview_Reviews.xlsx")
        overview_companies = pd.read_excel("Overview_Companies.xlsx")

        overview_reviews = overview_reviews.rename(columns={"id": "company_id"})
        overview_companies = overview_companies.rename(columns={"id": "company_id"})

        data = reviews.merge(overview_reviews[["company_id", "Overall rating"]], left_on="id", right_on="company_id", how="left")
        data = data.merge(overview_companies[["company_id", "Company Name", "Company Type", "Company size"]], on="company_id", how="left")

        st.write("📎 Available columns:", data.columns.tolist())
        return data

    # Load data
    df_reviews = load_review_data()

    # Validate required columns
    if 'What I liked' not in df_reviews.columns or 'Suggestions for improvement' not in df_reviews.columns:
        st.error("❌ Required columns 'What I liked' or 'Suggestions for improvement' are missing in the dataset.")
        st.stop()

    # Load stopwords and wrong words
    with open("vietnamese-stopwords.txt", encoding="utf-8") as f:
        stopwords = set(f.read().splitlines())

    with open("wrong-word.txt", encoding="utf-8") as f:
        wrong_words = set(f.read().splitlines())

    # Clean text function
    def clean_text(text):
        if pd.isnull(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = word_tokenize(text, format="text")
        words = [w for w in text.split() if w not in stopwords and w not in wrong_words and len(w) > 2]
        return " ".join(words)

    # Suggestion classification function
    def suggest_improvement(text):
        if pd.isnull(text):
            return "Không có góp ý tiêu cực"
        text = str(text).lower()
        negative_keywords = [
            "không", "thiếu", "chưa", "overtime", "lương_thấp",
            "áp_lực", "chậm", "tệ", "bất_công", "quá_tải", "stress"
        ]
        for kw in negative_keywords:
            if kw in text:
                return "Cần cải thiện: " + kw.replace("_", " ")
        return "Không có góp ý tiêu cực"

    # Apply cleaning
    df_reviews['What I liked_clean'] = df_reviews['What I liked'].apply(clean_text)
    df_reviews['Suggestions_clean'] = df_reviews['Suggestions for improvement'].apply(suggest_improvement)
    df_reviews['text_combined'] = df_reviews['What I liked_clean'] + ' ' + df_reviews['Suggestions_clean']

    # Feature engineering
    features = df_reviews[['text_combined', 'Rating', 'Company Type', 'Company size', 'Overall rating']]
    target = df_reviews['Recommend?']

    # Encode categorical variables
    categorical_cols = ['Company Type', 'Company size']
    features_processed = pd.get_dummies(features, columns=categorical_cols, drop_first=True)


    # Chuyển đổi text thành TF-IDF features
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
    text_features = tfidf.fit_transform(features['text_combined'])
    # Kết hợp text features và các features số khác
    # Drop 'text_combined' and convert to float (after ensuring all are numeric)
    numeric_features = features_processed.drop('text_combined', axis=1).apply(pd.to_numeric, errors='coerce').fillna(0).values.astype('float64')
    X = hstack((text_features, numeric_features))
    y = target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)


    #----------------
    # Only keeping Logistic Regression, SVM, and XGBoost
    # Removing Random Forest, Spark models, and related evaluation logic
    # The model training and evaluation will focus on these three models

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from sklearn.metrics import (
        classification_report, roc_auc_score, roc_curve,
        accuracy_score, precision_score, recall_score, f1_score
    )
    from sklearn.preprocessing import LabelEncoder

    # --- Model Training ---

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_balanced, y_train_balanced)
    lr_pred = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test == "Yes", lr_proba)
    lr_fpr, lr_tpr, _ = roc_curve(y_test == "Yes", lr_proba)

    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train_balanced, y_train_balanced)
    svm_pred = svm_model.predict(X_test)
    svm_proba = svm_model.predict_proba(X_test)[:, 1]
    svm_auc = roc_auc_score(y_test == "Yes", svm_proba)
    svm_fpr, svm_tpr, _ = roc_curve(y_test == "Yes", svm_proba)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train_balanced)
    y_test_enc = label_encoder.transform(y_test)
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train_balanced, y_train_enc)
    xgb_pred_enc = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = label_encoder.inverse_transform(xgb_pred_enc)
    xgb_auc = roc_auc_score(y_test == "Yes", xgb_proba)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test == "Yes", xgb_proba)

    # --- Visualization ---

    plt.figure(figsize=(10, 8))
    plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC = {lr_auc:.4f})")
    plt.plot(svm_fpr, svm_tpr, label=f"SVM (AUC = {svm_auc:.4f})")
    plt.plot(xgb_fpr, xgb_tpr, label=f"XGBoost (AUC = {xgb_auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Selected Models")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Metrics Comparison ---

    models_comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "SVM", "XGBoost"],
        "AUC": [lr_auc, svm_auc, xgb_auc],
        "Accuracy": [
            accuracy_score(y_test, lr_pred),
            accuracy_score(y_test, svm_pred),
            accuracy_score(y_test, xgb_pred)
        ],
        "Precision": [
            precision_score(y_test, lr_pred, pos_label="Yes"),
            precision_score(y_test, svm_pred, pos_label="Yes"),
            precision_score(y_test, xgb_pred, pos_label="Yes")
        ],
        "Recall": [
            recall_score(y_test, lr_pred, pos_label="Yes"),
            recall_score(y_test, svm_pred, pos_label="Yes"),
            recall_score(y_test, xgb_pred, pos_label="Yes")
        ],
        "F1-Score": [
            f1_score(y_test, lr_pred, pos_label="Yes"),
            f1_score(y_test, svm_pred, pos_label="Yes"),
            f1_score(y_test, xgb_pred, pos_label="Yes")
        ]
    })

    print("\nModel Performance Comparison:")
    print(models_comparison.sort_values("AUC", ascending=False))

    # --- ROC Curve Display inside Streamlit ---
    st.subheader("📉 ROC Curve - Logistic Regression vs SVM vs XGBoost")

    fig_roc, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC = {lr_auc:.4f})")
    ax.plot(svm_fpr, svm_tpr, label=f"SVM (AUC = {svm_auc:.4f})")
    ax.plot(xgb_fpr, xgb_tpr, label=f"XGBoost (AUC = {xgb_auc:.4f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Selected Models")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig_roc)

    # --- Metrics Comparison Table ---
    st.subheader("📊 Model Performance Summary")
    st.dataframe(models_comparison.sort_values("AUC", ascending=False).round(4))

    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib

    from sklearn.preprocessing import OneHotEncoder

    # --- App Settings ---
    st.set_page_config(page_title="ITViec Review Recommendation Predictor")
    st.title("🔍 Predict 'Recommend' from Employee Review")
    st.markdown("Dựa trên thông tin đánh giá từ nhân viên đã review trên ITViec, dự đoán xem họ có recommend công ty hay không.")

    # --- Load Pre-trained Models and Encoders ---
    xgb_model = joblib.load("xgb_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    onehot_encoder = joblib.load("onehot_encoder.pkl")

    # --- Load Company Data ---
    overview_companies = pd.read_excel(
        "Overview_Companies.xlsx"
    )
    overview_companies = overview_companies.rename(columns={"Company Name": "company_name"})

    # --- Prediction UI ---
    st.subheader("📝 Dự đoán theo tên công ty")

    company_name_list = overview_companies["company_name"].dropna().unique().tolist()
    company_name = st.selectbox("Chọn tên công ty", sorted(company_name_list))

    if st.button("Dự đoán"):
        try:
            # Lấy thông tin công ty từ tên đã chọn
            selected_info = overview_companies[overview_companies["company_name"] == company_name].iloc[0]
            company_type = selected_info["Company Type"]
            company_size = selected_info["Company size"]

            # Encode categorical features
            cat_features = pd.DataFrame([[company_type, company_size]], columns=["Company Type", "Company size"])
            cat_encoded = onehot_encoder.transform(cat_features)  # Do not use .toarray() if sparse_output=False

            # Tạo mảng 1010 chiều, chèn cat_encoded vào cuối
            num_total_features = 1010  # số chiều model yêu cầu
            num_cat_features = cat_encoded.shape[1]

            # Khởi tạo mảng toàn 0
            final_features = np.zeros((1, num_total_features))

            # Gán cat_encoded vào phần cuối
            final_features[0, -num_cat_features:] = cat_encoded

            # Dự đoán
            proba = xgb_model.predict_proba(final_features)[0][1]
            prediction = label_encoder.inverse_transform([int(proba >= 0.5)])[0]



            # Output
            st.subheader("🔍 Kết quả")
            st.write(f"**Xác suất Recommend:** {proba:.2%}")
            st.success(f"✨ Dự đoán: **{prediction}**")

        except IndexError:
            st.error("❌ Không tìm thấy thông tin công ty. Vui lòng thử lại.")
        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")






    import streamlit as st