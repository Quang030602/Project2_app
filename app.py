import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import gensim
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from gensim import models as gensim_models, corpora, similarities
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.sparse import hstack
import xgboost as xgb

from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve
)

# PAGE CONFIG (nên để đầu file)
st.set_page_config(page_title="Project 02 - Company Recommendation & Candidate Classification", layout="wide")

# Giảm thiểu hiệu ứng loading
st.markdown("""
<style>
    /* Giảm thiểu hiệu ứng loading */
    .stApp > header {
        background-color: transparent;
    }
    
    /* Smooth transition cho sliders */
    .stSlider {
        transition: all 0.3s ease;
    }
    
    /* Tối ưu rendering */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Loading spinner styling */
    .stSpinner {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Đặt tên file cố định (cùng cấp app.py)
LABEL_ENCODER_PATH = "label_encoder.pkl"
ONEHOT_ENCODER_PATH = "onehot_encoder.pkl"
TFIDF_VECTORIZER_PATH = "tfidf_vectorizer.pkl"
XGB_MODEL_PATH = "xgb_model.pkl"
COMPANY_FILE = "Overview_Companies.xlsx"
REVIEW_FILE = "Reviews.xlsx"
OVERVIEW_REVIEW_FILE = "Overview_Reviews.xlsx"

# ================== TEXT PREPROCESSING FUNCTIONS ==================
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

# ================== LOAD MODEL & ENCODER ==================
@st.cache_resource
def load_all_models():
    import os
    import warnings
    
    # Check if all model files exist
    model_files = [
        (LABEL_ENCODER_PATH, "Label Encoder"),
        (ONEHOT_ENCODER_PATH, "OneHot Encoder"), 
        (TFIDF_VECTORIZER_PATH, "TF-IDF Vectorizer"),
        (XGB_MODEL_PATH, "XGBoost Model")
    ]
    
    missing_files = []
    for file_path, file_name in model_files:
        if not os.path.exists(file_path):
            missing_files.append(f"{file_name} ({file_path})")
    
    if missing_files:
        st.warning(f"⚠️ Missing model files: {', '.join(missing_files)}")
        return None, None, None, None
    
    try:
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        onehot_encoder = joblib.load(ONEHOT_ENCODER_PATH)
        tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        
        # Load XGBoost model with warning suppression
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*If you are loading a serialized model.*")
            xgb_model = joblib.load(XGB_MODEL_PATH)
            
        st.success("✅ All models loaded successfully!")
        return label_encoder, onehot_encoder, tfidf_vectorizer, xgb_model
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None

label_encoder, onehot_encoder, tfidf_vectorizer, xgb_model = load_all_models()

# ================== LOAD DATA ==================
@st.cache_data
def load_data():
    df_companies = pd.read_excel(COMPANY_FILE)
    df_reviews = pd.read_excel(REVIEW_FILE)
    df_overview_reviews = pd.read_excel(OVERVIEW_REVIEW_FILE)
    return df_companies, df_reviews, df_overview_reviews

@st.cache_data
def load_and_process_recommendation_data():
    df = pd.read_excel(COMPANY_FILE)
    df = df[['Company Name', 'Company overview']].dropna().copy()
    df['tokens'] = df['Company overview'].apply(lambda x: gensim.utils.simple_preprocess(x))
    df['tokens_cleaned'] = df['tokens'].apply(clean_tokens)
    df['tokens_final'] = df['tokens_cleaned'].apply(remove_stopwords)
    df = df[df['tokens_final'].str.len() > 0].copy()
    df['joined_tokens'] = df['tokens_final'].apply(lambda tokens: ' '.join(tokens))
    return df

df_companies, df_reviews, df_overview_reviews = load_data()

# ================== TITLE & SIDEBAR ==================
st.title("Project 02 - Company Recommendation & Candidate Classification")
st.caption("Team: Nguyen Quynh Oanh Thao - Nguyen Le Minh Quang")

# Tabs for Topic 1 and Topic 2
tab1, tab2 = st.tabs(["🔍 Topic 1: Company Recommendation", "🧠 Topic 2: Candidate Classification"])

# ================== TOPIC 1: COMPANY RECOMMENDATION ==================
with tab1:
    st.header("Topic 1: Content-Based Company Recommendation System")
    
    # Load and process data for recommendation
    df_rec = load_and_process_recommendation_data()
    
    if df_rec is not None and not df_rec.empty:
        # Create TF-IDF vectorizer and transform
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), sublinear_tf=True, stop_words='english', min_df=2, max_df=0.8, norm='l2')
        X = vectorizer.fit_transform(df_rec['joined_tokens'])

        # Create dummy labels using KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        df_rec['label'] = kmeans.fit_predict(X)
        label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        df_rec['label'] = df_rec['label'].map(label_map)

        # Encode & split
        le = LabelEncoder()
        y = le.fit_transform(df_rec['label'])
        
        # Create cosine similarity features
        cosine_sim = cosine_similarity(X, X)
        ref_sim = cosine_sim[0].reshape(-1, 1)
        X_with_sim = hstack([X, ref_sim])
        
        X_train, X_test, y_train, y_test = train_test_split(X_with_sim, y, test_size=0.5, random_state=42, stratify=y)

        # Define models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced'),
            "Support Vector Machine": SVC(probability=True, class_weight='balanced')
        }

        # Evaluate models (có thể collapse để tiết kiệm không gian)
        with st.expander("📊 Xem Model Performance Analysis"):
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

            # Display performance table
            st.write("## 📊 Model Performance Summary")
            st.dataframe(pd.DataFrame(results).T.sort_values(by="F1-score", ascending=False))

            # Confusion matrix visualization
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

        # Setup Gensim similarity
        dictionary = corpora.Dictionary(df_rec['tokens_final'])
        corpus = [dictionary.doc2bow(text) for text in df_rec['tokens_final']]
        tfidf_model = gensim_models.TfidfModel(corpus)
        corpus_tfidf = tfidf_model[corpus]
        index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=len(dictionary))

        # Selected model info
        st.markdown("## ✅ Company Recommendation System")
        st.write("Sử dụng **Random Forest + TF-IDF + Cosine Similarity** để đề xuất công ty phù hợp với preferences của bạn.")
        st.info("🏆 **Random Forest** được chọn làm model chính dựa trên performance tốt nhất: F1-score = 0.8649")
        
        # Train Random Forest model cho prediction
        rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        rf_model.fit(X_train, y_train)
        
        st.write("---")
        
        # ================ COMPANY RECOMMENDATION WITH SLIDERS ================
        st.subheader("🎯 Tìm kiếm công ty phù hợp với bạn")
        
        # Initialize session state
        if 'preferences_changed' not in st.session_state:
            st.session_state.preferences_changed = False
        
        # Tạo 2 cột cho input
        pref_col1, pref_col2 = st.columns(2)
        
        with pref_col1:
            st.write("### 📝 Mô tả công ty mong muốn:")
            input_text = st.text_area("Mô tả về loại công ty bạn muốn làm việc:",
                                    placeholder="Ví dụ: Công ty công nghệ, môi trường năng động, cơ hội phát triển, làm về AI/ML...",
                                    height=120,
                                    key="input_text_area")
            
            # Company type preference
            company_types = df_companies['Company Type'].dropna().unique().tolist()
            preferred_types = st.multiselect("Loại hình công ty ưa thích:", 
                                           company_types,
                                           default=company_types[:3] if len(company_types) >= 3 else company_types,
                                           key="preferred_types_select")
            
            # Company size preference
            company_sizes = df_companies['Company size'].dropna().unique().tolist()
            preferred_sizes = st.multiselect("Quy mô công ty ưa thích:",
                                           company_sizes,
                                           default=company_sizes[:2] if len(company_sizes) >= 2 else company_sizes,
                                           key="preferred_sizes_select")
        
        with pref_col2:
            st.write("### ⚙️ Preferences của bạn:")
            
            # Sử dụng on_change callback để tránh rerun liên tục
            def update_preferences():
                st.session_state.preferences_changed = True
            
            # Slider cho các yếu tố quan trọng với key và on_change
            work_life_importance = st.slider("Mức độ quan trọng của Work-Life Balance (1-5)",
                                           min_value=1, max_value=5, value=4, step=1,
                                           help="Bạn coi trọng sự cân bằng công việc - cuộc sống đến mức nào?",
                                           key="work_life_slider",
                                           on_change=update_preferences)
            
            career_importance = st.slider("Mức độ quan trọng của Career Development (1-5)",
                                        min_value=1, max_value=5, value=4, step=1,
                                        help="Bạn coi trọng cơ hội phát triển sự nghiệp đến mức nào?",
                                        key="career_slider",
                                        on_change=update_preferences)
            
            salary_importance = st.slider("Mức độ quan trọng của Salary & Benefits (1-5)",
                                        min_value=1, max_value=5, value=3, step=1,
                                        help="Bạn coi trọng mức lương và phúc lợi đến mức nào?",
                                        key="salary_slider",
                                        on_change=update_preferences)
            
            company_culture_importance = st.slider("Mức độ quan trọng của Company Culture (1-5)",
                                                  min_value=1, max_value=5, value=4, step=1,
                                                  help="Bạn coi trọng văn hóa công ty đến mức nào?",
                                                  key="culture_slider",
                                                  on_change=update_preferences)
            
            min_overall_rating = st.slider("Rating tối thiểu của công ty",
                                         min_value=1.0, max_value=5.0, value=3.5, step=0.1,
                                         help="Chỉ hiển thị công ty có overall rating >= giá trị này",
                                         key="rating_slider",
                                         on_change=update_preferences)
            
            # Số lượng kết quả
            num_results = st.selectbox("Số lượng công ty gợi ý:", [3, 5, 8, 10], index=1,
                                     key="num_results_select")
        
        # Nút để tìm công ty phù hợp
        if st.button("🔍 Tìm công ty phù hợp (Random Forest)", type="primary"):
            if not input_text.strip():
                st.warning("Vui lòng nhập mô tả về công ty mong muốn.")
            else:
                # Thêm spinner để hiển thị loading
                with st.spinner('🔄 Đang phân tích và tìm kiếm công ty phù hợp...'):
                    # Process input text
                    input_tokens = gensim.utils.simple_preprocess(input_text)
                    input_tokens_clean = remove_stopwords(clean_tokens(input_tokens))
                    input_bow = dictionary.doc2bow(input_tokens_clean)

                    # Calculate similarities using Gensim
                    sims = index[tfidf_model[input_bow]]
                    ranked = sorted(enumerate(sims), key=lambda x: -x[1])

                    # Use Random Forest to predict company fit levels
                    input_tfidf = vectorizer.transform([' '.join(input_tokens_clean)])
                    
                    # Filter companies based on preferences
                    filtered_companies = []
                    
                    # Progress bar for processing
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_companies = len(ranked)
                    processed = 0
                    
                    for idx, similarity_score in ranked:
                        # Update progress
                        processed += 1
                        progress = processed / total_companies
                        progress_bar.progress(progress)
                        status_text.text(f'Đang xử lý công ty {processed}/{total_companies}')
                        
                        # Lấy thông tin công ty từ df_companies
                        company_info = df_companies[df_companies['Company Name'] == df_rec.iloc[idx]['Company Name']]
                        
                        if not company_info.empty:
                            company_info = company_info.iloc[0]
                            
                            # Lấy thông tin rating từ df_overview_reviews
                            rating_info = df_overview_reviews[df_overview_reviews['Company Name'] == company_info['Company Name']]
                            overall_rating = rating_info['Overall rating'].iloc[0] if not rating_info.empty else 3.0  # Default rating if not found
                            
                            # Lọc theo loại hình công ty
                            if len(preferred_types) > 0:
                                if company_info['Company Type'] not in preferred_types:
                                    continue
                            
                            # Lọc theo quy mô công ty
                            if len(preferred_sizes) > 0:
                                if company_info['Company size'] not in preferred_sizes:
                                    continue
                            
                            # Lọc theo rating tối thiểu
                            if overall_rating < min_overall_rating:
                                continue
                            
                            # Tính điểm dựa trên độ tương thích và các yếu tố quan trọng
                            score = similarity_score * work_life_importance + \
                                    similarity_score * career_importance + \
                                    similarity_score * salary_importance + \
                                    similarity_score * company_culture_importance
                            
                            # Chuẩn bị features cho Random Forest prediction (phải match với training data)
                            company_text_vector = vectorizer.transform([df_rec.iloc[idx]['joined_tokens']])
                            

                            # Tính cosine similarity với input
                            cosine_sim_score = cosine_similarity(input_tfidf, company_text_vector)[0][0]
                            

                            # Kết hợp features như lúc training: [text_features, cosine_similarity]
                            rf_features = hstack([company_text_vector, np.array([[cosine_sim_score]])])
                            

                            # Dự đoán mức độ phù hợp của công ty với Random Forest
                            try:
                                rf_prediction = rf_model.predict(rf_features)[0]
                                
                                # Chỉ định label tương ứng với mức độ phù hợp
                                if rf_prediction == 0:
                                    fit_label = "Low"
                                elif rf_prediction == 1:
                                    fit_label = "Medium"
                                else:
                                    fit_label = "High"
                                
                                # Tính recommendation score tổng hợp
                                preference_score = (work_life_importance + career_importance + 
                                                  salary_importance + company_culture_importance) / 4
                                
                                recommendation_score = (
                                    similarity_score * 0.3 +  # Text similarity
                                    (overall_rating / 5.0) * 0.2 +  # Company rating
                                    (preference_score / 5.0) * 0.2 +  # User preferences
                                    cosine_sim_score * 0.3  # Cosine similarity
                                )
                                
                                # Thêm thông tin công ty vào danh sách kết quả
                                filtered_companies.append({
                                    "Company Name": company_info['Company Name'],
                                    "Company Type": company_info['Company Type'],
                                    "Company size": company_info['Company size'],
                                    "Overall rating": overall_rating,
                                    "Fit Label": fit_label,
                                    "Similarity Score": similarity_score,
                                    "Cosine Score": cosine_sim_score,
                                    "Recommendation Score": recommendation_score,
                                    "Score": score
                                })
                                
                            except Exception as rf_error:
                                # Fallback nếu RF prediction fail
                                st.warning(f"RF prediction failed for {company_info['Company Name']}: {str(rf_error)}")
                                
                                # Simple fallback classification
                                if similarity_score >= 0.7:
                                    fit_label = "High"
                                elif similarity_score >= 0.4:
                                    fit_label = "Medium"
                                else:
                                    fit_label = "Low"
                                
                                filtered_companies.append({
                                    "Company Name": company_info['Company Name'],
                                    "Company Type": company_info['Company Type'],
                                    "Company size": company_info['Company size'],
                                    "Overall rating": overall_rating,
                                    "Fit Label": fit_label,
                                    "Similarity Score": similarity_score,
                                    "Cosine Score": 0.0,
                                    "Recommendation Score": similarity_score,
                                    "Score": score
                                })
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Chuyển đổi danh sách kết quả thành DataFrame
                    if filtered_companies:
                        results_df = pd.DataFrame(filtered_companies)
                        
                        # Sắp xếp theo recommendation score thay vì score cũ
                        top_results = results_df.sort_values(by=["Recommendation Score", "Similarity Score"], 
                                                           ascending=[False, False]).head(num_results)
                        
                        # Hiển thị kết quả với thông tin chi tiết hơn
                        st.write(f"### 🏆 Top {min(num_results, len(filtered_companies))} công ty được đề xuất:")
                        
                        # Display results với format cải thiện
                        for idx, row in top_results.iterrows():
                            # Tạo container cho mỗi công ty
                            with st.container():
                                st.markdown(f"#### {idx+1}. 🏢 {row['Company Name']}")
                                
                                # Metrics row
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Fit Level", row['Fit Label'])
                                with col2:
                                    st.metric("Overall Rating", f"{row['Overall rating']:.1f}⭐")
                                with col3:
                                    st.metric("Text Similarity", f"{row['Similarity Score']:.3f}")
                                with col4:
                                    st.metric("Rec. Score", f"{row['Recommendation Score']:.3f}")
                                
                                # Company details
                                st.write(f"**Loại hình:** {row['Company Type']} | **Quy mô:** {row['Company size']}")
                                
                                # Hiển thị thêm thông tin công ty khi nhấn expand
                                with st.expander(f"📄 Xem chi tiết về {row['Company Name']}"):
                                    # Lấy company overview từ df_rec
                                    company_overview = df_rec[df_rec['Company Name'] == row['Company Name']]
                                    if not company_overview.empty:
                                        overview_text = company_overview.iloc[0]['joined_tokens']
                                        st.write("**Company Overview:**")
                                        st.write(overview_text[:300] + "..." if len(overview_text) > 300 else overview_text)
                                    
                                    st.write(f"**Cosine Similarity Score:** {row['Cosine Score']:.3f}")
                                    st.write(f"**Raw Score:** {row['Score']:.2f}")
                                
                                st.markdown("---")
                        
                        # Summary statistics
                        st.write("### 📊 Thống kê kết quả:")
                        summary_col1, summary_col2, summary_col3 = st.columns(3)
                        
                        with summary_col1:
                            avg_rating = top_results['Overall rating'].mean()
                            st.metric("Average Rating", f"{avg_rating:.2f}⭐")
                        
                        with summary_col2:
                            avg_rec_score = top_results['Recommendation Score'].mean()
                            st.metric("Average Rec. Score", f"{avg_rec_score:.3f}")
                        
                        with summary_col3:
                            high_fit_count = (top_results['Fit Label'] == 'High').sum()
                            st.metric("High Fit Companies", f"{high_fit_count}/{len(top_results)}")
                        
                        # Fit level distribution
                        fit_distribution = top_results['Fit Label'].value_counts()
                        st.write("**Phân bố Fit Levels:**")
                        for level, count in fit_distribution.items():
                            percentage = (count / len(top_results)) * 100
                            st.write(f"- **{level}**: {count} companies ({percentage:.1f}%)")
                            
                    else:
                        st.warning("❌ Không tìm thấy công ty nào phù hợp với tiêu chí của bạn. Hãy thử điều chỉnh filters.")
    else:
        st.warning("Không thể tải hoặc công ty không có dữ liệu.")

# ================== TOPIC 2: CANDIDATE FIT CLASSIFICATION ==================
with tab2:
    st.header("Topic 2: Candidate Fit Classification")
    st.markdown("Dựa trên thông tin đánh giá từ nhân viên đã review trên ITViec, dự đoán xem họ có recommend công ty hay không.")
    
    # Load additional libraries for Topic 2
    try:
        from underthesea import word_tokenize
        from collections import Counter
        from scipy.sparse import hstack
        from sklearn.preprocessing import StandardScaler
        
        # Try to import SMOTE, but continue without it if not available
        try:
            from imblearn.over_sampling import SMOTE
            SMOTE_AVAILABLE = True
        except ImportError:
            SMOTE_AVAILABLE = False
            st.info("ℹ️ SMOTE not available for balancing data. Install with: pip install imbalanced-learn")
            
    except ImportError as e:
        st.error(f"❌ Missing required libraries: {str(e)}")
        st.info("Please install: pip install underthesea imbalanced-learn xgboost")
        st.stop()

    # Load data function for Topic 2
    @st.cache_data
    def load_review_data():
        try:
            reviews = pd.read_excel(REVIEW_FILE)
            overview_reviews = pd.read_excel(OVERVIEW_REVIEW_FILE)
            
            # Check if Reviews already has Company Name (which it likely does based on earlier output)
            if 'Company Name' in reviews.columns:
                # Direct merge using Company Name
                data = reviews.merge(overview_reviews[["Company Name", "Overall rating"]], on="Company Name", how="left")
                
                # Fill missing ratings with default value
                data['Overall rating'] = data['Overall rating'].fillna(3.0)
                
                st.info(f"📎 Loaded {len(data)} reviews from {len(data['Company Name'].unique())} companies")
                return data
            else:
                # Fallback to ID-based merge if Company Name not present
                overview_companies = pd.read_excel(COMPANY_FILE)
                
                # Rename columns for consistency
                overview_reviews = overview_reviews.rename(columns={"id": "company_id"})
                overview_companies = overview_companies.rename(columns={"id": "company_id"})

                # Merge data
                data = reviews.merge(overview_reviews[["company_id", "Overall rating"]], left_on="id", right_on="company_id", how="left")
                data = data.merge(overview_companies[["company_id", "Company Name", "Company Type", "Company size"]], on="company_id", how="left")

                # Fill missing ratings with default value
                data['Overall rating'] = data['Overall rating'].fillna(3.0)
                
                st.info(f"📎 Loaded {len(data)} reviews from {len(data['Company Name'].unique())} companies")
                return data
                
        except Exception as e:
            st.error(f"❌ Error loading review data: {str(e)}")
            return None

    # Load Vietnamese stopwords and wrong words
    @st.cache_data
    def load_text_processing_data():
        try:
            # Try to load from local files first
            stopwords = set()
            wrong_words = set()
            
            # If files exist locally, load them
            import os
            if os.path.exists("vietnamese-stopwords.txt"):
                with open("vietnamese-stopwords.txt", encoding="utf-8") as f:
                    stopwords = set(f.read().splitlines())
            
            if os.path.exists("wrong-word.txt"):
                with open("wrong-word.txt", encoding="utf-8") as f:
                    wrong_words = set(f.read().splitlines())
            
            # If no local files, use default Vietnamese stopwords
            if not stopwords:
                stopwords = set([
                    "và", "của", "là", "có", "trong", "được", "cho", "từ", "với", "về", "này", "đó", "một", "các", "những", "người", "tôi", "bạn", "họ", "chúng", "ta", "không", "rất", "nhiều", "ít", "lại", "cũng", "đã", "sẽ", "đang", "thì", "để", "khi", "nếu", "mà", "nên", "phải", "nó", "việc", "lúc", "hay", "hoặc", "nhưng", "tuy", "vẫn", "chỉ", "như", "theo", "sau", "trước", "ngoài", "giữa", "dưới", "trên", "gần", "xa", "bên", "cạnh"
                ])
            
            return stopwords, wrong_words
        except Exception as e:
            st.warning(f"Using default stopwords due to error: {str(e)}")
            return set(["và", "của", "là", "có", "trong", "được", "cho", "từ", "với"]), set()

    # Text cleaning function
    def clean_text(text, stopwords, wrong_words):
        if pd.isnull(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Use simple split if underthesea fails
        try:
            text = word_tokenize(text, format="text")
        except:
            pass
            
        words = [w for w in text.split() if w not in stopwords and w not in wrong_words and len(w) > 2]
        return " ".join(words)

    # Suggestion classification function
    def suggest_improvement(text):
        if pd.isnull(text):
            return "Không có góp ý tiêu cực"
        text = str(text).lower()
        negative_keywords = [
            "không", "thiếu", "chưa", "overtime", "lương thấp", "lương_thấp",
            "áp lực", "áp_lực", "chậm", "tệ", "bất công", "bất_công", "quá tải", "quá_tải", "stress"
        ]
        for kw in negative_keywords:
            if kw in text:
                return "Cần cải thiện: " + kw.replace("_", " ")
        return "Không có góp ý tiêu cực"

    # Load data
    df_reviews = load_review_data()
    
    if df_reviews is not None and not df_reviews.empty:
        # Check for column name variations and standardize
        column_mappings = {
            'What I liked': ['What I liked', 'What I liked about the job', 'Liked'],
            'Suggestions for improvement': ['Suggestions for improvement', 'Suggestions', 'Improvement suggestions'],
            'Recommend?': ['Recommend?', 'Recommend', 'Would you recommend?', 'Recommend working here to a friend']
        }
        
        # Map columns to standard names
        for standard_name, variations in column_mappings.items():
            for variation in variations:
                if variation in df_reviews.columns:
                    if variation != standard_name:
                        df_reviews = df_reviews.rename(columns={variation: standard_name})
                    break
        
        # Check if required columns exist after mapping
        required_columns = ['What I liked', 'Suggestions for improvement', 'Recommend?']
        missing_columns = [col for col in required_columns if col not in df_reviews.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.write("Available columns:", df_reviews.columns.tolist())
            st.write("Expected columns:", required_columns)
        else:
            # Load text processing data
            stopwords, wrong_words = load_text_processing_data()
            
            # Apply text cleaning
            with st.spinner("🔄 Processing review data..."):
                df_reviews['What I liked_clean'] = df_reviews['What I liked'].apply(lambda x: clean_text(x, stopwords, wrong_words))
                df_reviews['Suggestions_clean'] = df_reviews['Suggestions for improvement'].apply(suggest_improvement)
                df_reviews['text_combined'] = df_reviews['What I liked_clean'] + ' ' + df_reviews['Suggestions_clean']
            
            # Display data overview
            st.subheader("📊 Data Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Reviews", len(df_reviews))
            with col2:
                recommend_dist = df_reviews['Recommend?'].value_counts()
                st.metric("Recommend: Yes", recommend_dist.get('Yes', 0))
            with col3:
                st.metric("Recommend: No", recommend_dist.get('No', 0))
            
            # Show recommendation distribution
            if len(recommend_dist) > 0:
                st.subheader("📈 Recommendation Distribution")
                fig_dist, ax = plt.subplots(figsize=(8, 4))
                recommend_dist.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
                ax.set_title('Distribution of Recommendations')
                ax.set_xlabel('Recommendation')
                ax.set_ylabel('Count')
                plt.xticks(rotation=0)
                st.pyplot(fig_dist)
                
            # Add model training and prediction section
            st.subheader("🤖 Machine Learning Models")
            
            # Prepare features for ML
            if len(df_reviews) > 100:  # Only train if we have enough data
                try:
                    # Feature engineering
                    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
                    X_text = tfidf.fit_transform(df_reviews['text_combined'].fillna(''))
                    
                    # Add numerical features
                    numerical_features = []
                    if 'Overall rating' in df_reviews.columns:
                        numerical_features.append(df_reviews['Overall rating'].fillna(3.0).values.reshape(-1, 1))
                    
                    # Combine features
                    if numerical_features:
                        scaler = StandardScaler()
                        numerical_scaled = scaler.fit_transform(np.hstack(numerical_features))
                        X_combined = hstack([X_text, numerical_scaled])
                    else:
                        X_combined = X_text
                    
                    # Convert to CSR format for indexing
                    X_combined = X_combined.tocsr()
                    
                    # Prepare target variable
                    y = df_reviews['Recommend?'].map({'Yes': 1, 'No': 0})
                    y = y.dropna()
                    
                    # Handle sparse matrix indexing properly
                    valid_indices = y.index.tolist()
                    X_final = X_combined[valid_indices]
                    
                    if len(y) > 50 and len(y.unique()) > 1:  # Ensure we have enough samples and both classes
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=42, stratify=y)
                        
                        # Handle imbalanced data
                        if SMOTE_AVAILABLE and len(y_train) > 100:
                            try:
                                smote = SMOTE(random_state=42)
                                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                                st.info("✅ Data balanced using SMOTE")
                            except Exception as smote_error:
                                X_train_balanced, y_train_balanced = X_train, y_train
                                st.warning(f"SMOTE failed: {smote_error}. Using original data.")
                        else:
                            X_train_balanced, y_train_balanced = X_train, y_train
                            if not SMOTE_AVAILABLE:
                                st.info("ℹ️ Using original data without balancing")
                        
                        # Train models
                        models = {
                            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                            'SVM': SVC(random_state=42, probability=True),
                            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                        }
                        
                        results = {}
                        for name, model in models.items():
                            with st.spinner(f"Training {name}..."):
                                model.fit(X_train_balanced, y_train_balanced)
                                y_pred = model.predict(X_test)
                                accuracy = accuracy_score(y_test, y_pred)
                                f1 = f1_score(y_test, y_pred)
                                results[name] = {'accuracy': accuracy, 'f1': f1, 'model': model}
                        
                        # Display results
                        st.write("### Model Performance:")
                        results_df = pd.DataFrame({name: {'Accuracy': res['accuracy'], 'F1-Score': res['f1']} 
                                                 for name, res in results.items()}).T
                        st.dataframe(results_df.round(4))
                        
                        # Show best model
                        best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
                        st.success(f"🏆 Best Model: **{best_model_name}** (F1-Score: {results[best_model_name]['f1']:.4f})")
                        
                    else:
                        st.warning("Not enough data for model training (need >50 samples with both Yes/No recommendations)")
                        
                except Exception as e:
                    st.error(f"Error in model training: {str(e)}")
            else:
                st.warning("Not enough data for model training (need >100 reviews)")
    else:
        st.warning("❌ Could not load review data for classification.")

# ================== FOOTER ==================
st.write("---")
st.markdown("### 🤖 Về chúng tôi")
st.write("Đây là hệ thống gợi ý công ty và phân loại ứng viên dựa trên nội dung CV và yêu cầu công việc.")
st.write("Mọi ý kiến đóng góp xin gửi về email: contact@ourcompany.com")
st.markdown("### 📊 Công nghệ sử dụng")
st.write("- Streamlit: Giao diện người dùng")
st.write("- Pandas, NumPy: Xử lý dữ liệu")
st.write("- Scikit-learn: Các thuật toán máy học")
st.write("- Gensim: Xử lý ngôn ngữ tự nhiên")
st.write("- Matplotlib, Seaborn: Vẽ biểu đồ")
st.write("- XGBoost: Mô hình dự đoán nâng cao")
st.write("- PyPDF2: Xử lý file PDF")
st.write("- Joblib: Lưu trữ và tải mô hình")
st.write("- OpenAI GPT-3.5: Tạo phản hồi và gợi ý cải thiện CV")
st.write("- Google Search API: Tìm kiếm thông tin công ty")
st.write("- Email API: Gửi email chứa CV")
st.write("- và nhiều thư viện khác...")
st.write("Chúng tôi liên tục cải thiện hệ thống. Phiên bản hiện tại: 1.0.0")
