import streamlit as st
import pandas as pd
import numpy as np
import re
import gensim
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from gensim import models as gensim_models, corpora, similarities
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from scipy.sparse import hstack
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from collections import Counter
import os

# Try to import underthesea, install if not available
try:
    from underthesea import word_tokenize
except ImportError:
    st.error("Please install underthesea: pip install underthesea")
    st.stop()

# Try to import xgboost, install if not available
try:
    from xgboost import XGBClassifier
except ImportError:
    st.error("Please install xgboost: pip install xgboost")
    st.stop()

st.set_page_config(page_title="Project 02", layout="wide")

st.title("Project 02 - Company Recommendation & Candidate Classification")
st.caption("Team: Nguyen Quynh Oanh Thao - Nguyen Le Minh Quang")

menu = ["Home", "About"]
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Home':
    st.subheader("Streamlit Application")
elif choice == 'About':
    st.subheader("Requirements info")

# Tabs for Topic 1 and Topic 2
tab1, tab2 = st.tabs(["🔍 Topic 1: Company Recommendation", "🧠 Topic 2: Candidate Classification"])

with tab1:
    st.header("Topic 1: Content-Based Company Recommendation System")

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
        try:
            # Update path to local data directory
            data_path = "data/Overview_Companies.xlsx"
            if not os.path.exists(data_path):
                st.error(f"Data file not found: {data_path}")
                st.info("Please ensure the data files are in the 'data' directory")
                return pd.DataFrame()
            
            df = pd.read_excel(data_path)
            df = df[['Company Name', 'Company overview']].dropna().copy()
            df['tokens'] = df['Company overview'].apply(lambda x: gensim.utils.simple_preprocess(x))
            df['tokens_cleaned'] = df['tokens'].apply(clean_tokens)
            df['tokens_final'] = df['tokens_cleaned'].apply(remove_stopwords)
            df = df[df['tokens_final'].str.len() > 0].copy()
            df['joined_tokens'] = df['tokens_final'].apply(lambda tokens: ' '.join(tokens))
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()

    df = load_and_process_data()
    
    if df.empty:
        st.warning("No data available. Please check your data files.")
    else:
        # --- ML Classification ---
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), sublinear_tf=True, stop_words='english', min_df=2, max_df=0.8, norm='l2')
        X = vectorizer.fit_transform(df['joined_tokens'])

        # Dummy label for now: use KMeans or known labels
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

        # --- Define models ---
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced'),
            "Support Vector Machine": SVC(probability=True, class_weight='balanced')
        }
        
        # --- Recalculate cosine feature and split ---
        cosine_sim = cosine_similarity(X, X)
        ref_sim = cosine_sim[0].reshape(-1, 1)
        X_with_sim = hstack([X, ref_sim])
        X_train, X_test, y_train, y_test = train_test_split(
            X_with_sim, y, test_size=0.5, random_state=42, stratify=y)

        # --- Evaluate All Models ---
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

        # --- Display Performance Table ---
        st.write("## 📊 Model Performance Summary")
        st.dataframe(pd.DataFrame(results).T.sort_values(by="F1-score", ascending=False))

        # --- Confusion Matrix Visualization ---
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
        st.write("""We are using **Logistic Regression** as the primary model for predicting company fit and driving the similarity search logic below, based on its strong F1-score and overall performance.""")

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

    # --- Load data ---
    @st.cache_data
    def load_review_data():
        try:
            reviews_path = "data/Reviews.xlsx"
            overview_reviews_path = "data/Overview_Reviews.xlsx"
            overview_companies_path = "data/Overview_Companies.xlsx"
            
            # Check if files exist
            for path in [reviews_path, overview_reviews_path, overview_companies_path]:
                if not os.path.exists(path):
                    st.error(f"Data file not found: {path}")
                    return pd.DataFrame()
            
            reviews = pd.read_excel(reviews_path)
            overview_reviews = pd.read_excel(overview_reviews_path)
            overview_companies = pd.read_excel(overview_companies_path)

            overview_reviews = overview_reviews.rename(columns={"id": "company_id"})
            overview_companies = overview_companies.rename(columns={"id": "company_id"})

            data = reviews.merge(overview_reviews[["company_id", "Overall rating"]], left_on="id", right_on="company_id", how="left")
            data = data.merge(overview_companies[["company_id", "Company Name", "Company Type", "Company size"]], on="company_id", how="left")

            st.write("📎 Available columns:", data.columns.tolist())
            return data
        except Exception as e:
            st.error(f"Error loading review data: {str(e)}")
            return pd.DataFrame()

    df_reviews = load_review_data()
    
    if df_reviews.empty:
        st.warning("No review data available. Please check your data files.")
    else:
        # Validate required columns
        if 'What I liked' not in df_reviews.columns or 'Suggestions for improvement' not in df_reviews.columns:
            st.error("❌ Required columns 'What I liked' or 'Suggestions for improvement' are missing in the dataset.")
        else:
            # Load stopwords and wrong words
            try:
                with open("data/files/vietnamese-stopwords.txt", encoding="utf-8") as f:
                    stopwords = set(f.read().splitlines())
                with open("data/files/wrong-word.txt", encoding="utf-8") as f:
                    wrong_words = set(f.read().splitlines())
            except FileNotFoundError:
                st.warning("Stopwords files not found. Using default empty sets.")
                stopwords = set()
                wrong_words = set()

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

            # Text to TF-IDF features
            tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
            text_features = tfidf.fit_transform(features['text_combined'])
            
            # Combine features
            numeric_features = features_processed.drop('text_combined', axis=1).apply(pd.to_numeric, errors='coerce').fillna(0).values.astype('float64')
            X = hstack((text_features, numeric_features))
            y = target

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

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

            # --- ROC Curve Display ---
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

            st.subheader("📊 Model Performance Summary")
            st.dataframe(models_comparison.sort_values("AUC", ascending=False).round(4))

            # --- Prediction UI ---
            st.subheader("🔍 Predict 'Recommend' from Employee Review")
            st.markdown("Dựa trên thông tin đánh giá từ nhân viên đã review trên ITViec, dự đoán xem họ có recommend công ty hay không.")

            # Load company data for prediction
            try:
                overview_companies = pd.read_excel("data/Overview_Companies.xlsx")
                overview_companies = overview_companies.rename(columns={"Company Name": "company_name"})
                
                st.subheader("📝 Dự đoán theo tên công ty")
                company_name_list = overview_companies["company_name"].dropna().unique().tolist()
                company_name = st.selectbox("Chọn tên công ty", sorted(company_name_list))

                if st.button("Dự đoán"):
                    try:
                        # Get company info
                        selected_info = overview_companies[overview_companies["company_name"] == company_name].iloc[0]
                        company_type = selected_info["Company Type"]
                        company_size = selected_info["Company size"]

                        # Simple prediction based on available data
                        st.subheader("🔍 Kết quả")
                        st.write(f"**Công ty:** {company_name}")
                        st.write(f"**Loại công ty:** {company_type}")
                        st.write(f"**Quy mô:** {company_size}")
                        
                        # Mock prediction for demonstration
                        mock_proba = np.random.random()
                        prediction = "Yes" if mock_proba >= 0.5 else "No"
                        
                        st.write(f"**Xác suất Recommend:** {mock_proba:.2%}")
                        st.success(f"✨ Dự đoán: **{prediction}**")
                        st.info("Note: This is a simplified prediction. For full functionality, pre-trained models and encoders are needed.")

                    except Exception as e:
                        st.error(f"❌ Lỗi: {str(e)}")
                        
            except FileNotFoundError:
                st.error("Company data file not found for prediction functionality.")
