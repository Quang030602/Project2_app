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
from sklearn.preprocessing import LabelEncoder
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
    try:
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        onehot_encoder = joblib.load(ONEHOT_ENCODER_PATH)
        tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        xgb_model = joblib.load(XGB_MODEL_PATH)
        return label_encoder, onehot_encoder, tfidf_vectorizer, xgb_model
    except:
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
                            
                            # Lọc theo loại hình công ty
                            if len(preferred_types) > 0:
                                if company_info['Company Type'] not in preferred_types:
                                    continue
                            
                            # Lọc theo quy mô công ty
                            if len(preferred_sizes) > 0:
                                if company_info['Company size'] not in preferred_sizes:
                                    continue
                            
                            # Lọc theo rating tối thiểu
                            if company_info['Overall rating'] < min_overall_rating:
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
                                    (company_info['Overall rating'] / 5.0) * 0.2 +  # Company rating
                                    (preference_score / 5.0) * 0.2 +  # User preferences
                                    cosine_sim_score * 0.3  # Cosine similarity
                                )
                                
                                # Thêm thông tin công ty vào danh sách kết quả
                                filtered_companies.append({
                                    "Company Name": company_info['Company Name'],
                                    "Company Type": company_info['Company Type'],
                                    "Company size": company_info['Company size'],
                                    "Overall rating": company_info['Overall rating'],
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
                                    "Overall rating": company_info['Overall rating'],
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

# ================== TOPIC 2: CANDIDATE CLASSIFICATION ==================
with tab2:
    st.header("Topic 2: Candidate Classification System")
    
    @st.cache_data
    def load_candidate_classification_data():
        df = pd.read_excel(COMPANY_FILE)
        df = df[['Company Name', 'Company overview']].dropna().copy()
        df['tokens'] = df['Company overview'].apply(lambda x: gensim.utils.simple_preprocess(x))
        df['tokens_cleaned'] = df['tokens'].apply(clean_tokens)
        df['tokens_final'] = df['tokens_cleaned'].apply(remove_stopwords)
        df = df[df['tokens_final'].str.len() > 0].copy()
        df['joined_tokens'] = df['tokens_final'].apply(lambda tokens: ' '.join(tokens))
        return df

    df_candidates = load_candidate_classification_data()

    if df_candidates is not None and not df_candidates.empty and tfidf_vectorizer is not None and xgb_model is not None:
        try:
            # Vectorize the text data
            X_candidates = tfidf_vectorizer.transform(df_candidates['joined_tokens'])

            # Predict with the XGBoost model (sửa lại cách predict)
            df_candidates['predicted_label'] = xgb_model.predict(X_candidates)

            # Map the predicted labels to actual labels (sửa lại cách sử dụng label encoder)
            if label_encoder is not None:
                df_candidates['predicted_label'] = label_encoder.inverse_transform(df_candidates['predicted_label'].astype(int))
            else:
                # Nếu không có label encoder, map trực tiếp
                label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
                df_candidates['predicted_label'] = df_candidates['predicted_label'].map(label_map)

            st.write("Dưới đây là một số dự đoán về phân loại ứng viên:")
            st.dataframe(df_candidates[['Company Name', 'predicted_label']].head(10))
            
        except Exception as e:
            st.error(f"❌ Lỗi khi predict: {str(e)}")
            st.write("Có thể model hoặc vectorizer không tương thích với dữ liệu hiện tại.")
    else:
        st.warning("❌ Không thể tải model, vectorizer hoặc dữ liệu ứng viên.")

    st.write("---")
    
    # ================ UPLOAD CV SECTION ================
    st.subheader("📤 Tải lên CV của bạn để phân loại")
    uploaded_file = st.file_uploader("Chọn file CV của bạn (định dạng .txt hoặc .pdf)", type=["txt", "pdf"])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.txt'):
                cv_text = uploaded_file.read().decode("utf-8")
            else:
                # Convert PDF to text
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                cv_text = ""
                for page in pdf_reader.pages:
                    cv_text += page.extract_text()
            
            # Kiểm tra nếu có text
            if not cv_text.strip():
                st.warning("⚠️ Không thể đọc text từ file. Vui lòng kiểm tra lại file.")
            else:
                # Preprocess the CV text
                cv_tokens = gensim.utils.simple_preprocess(cv_text)
                cv_tokens_clean = remove_stopwords(clean_tokens(cv_tokens))
                cv_joined = ' '.join(cv_tokens_clean)
                
                # Kiểm tra nếu có text sau preprocessing
                if not cv_joined.strip():
                    st.warning("⚠️ Không có text hợp lệ sau khi xử lý. Vui lòng kiểm tra nội dung file.")
                else:
                    # Vectorize the CV
                    cv_vectorized = tfidf_vectorizer.transform([cv_joined])
                    
                    # Predict with the XGBoost model (sửa lại)
                    cv_prediction = xgb_model.predict(cv_vectorized)
                    
                    # Xử lý kết quả prediction
                    if label_encoder is not None:
                        cv_predicted_label = label_encoder.inverse_transform(cv_prediction.astype(int))[0]
                    else:
                        # Fallback mapping
                        label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
                        cv_predicted_label = label_map.get(int(cv_prediction[0]), 'Unknown')
                    
                    # Hiển thị kết quả với style đẹp hơn
                    st.success(f"✅ **CV của bạn đã được phân loại là: {cv_predicted_label}**")
                    
                    # Hiển thị confidence score nếu có
                    try:
                        cv_proba = xgb_model.predict_proba(cv_vectorized)
                        max_proba = cv_proba.max()
                        st.info(f"🎯 **Độ tin cậy:** {max_proba:.2%}")
                    except:
                        pass
                    
                    # Provide feedback and improvement suggestions
                    st.write("---")
                    st.subheader("🛠️ Gợi ý cải thiện CV")
                    
                    # Enhanced feedback
                    feedback = {
                        "High": "🔥 **Tuyệt vời!** CV của bạn có chất lượng cao. Hãy tiếp tục duy trì và cập nhật thường xuyên.",
                        "Medium": "👍 **Khá tốt!** CV có thể được cải thiện thêm:\n- Thêm các dự án cụ thể\n- Nêu rõ thành tích bằng số liệu\n- Cập nhật kỹ năng mới",
                        "Low": "⚠️ **Cần cải thiện:** CV cần được nâng cấp:\n- Bổ sung kinh nghiệm cụ thể\n- Thêm kỹ năng chuyên môn\n- Cải thiện cách trình bày\n- Thêm chứng chỉ/khóa học liên quan",
                        "IT/Software": "💻 **Kỹ thuật:** Nhấn mạnh kỹ năng lập trình, framework, và dự án đã thực hiện.",
                        "Marketing": "📢 **Marketing:** Thêm số liệu cụ thể về campaign và ROI đã đạt được.",
                        "Sales": "💼 **Kinh doanh:** Làm nổi bật kỹ năng đàm phán và target đã hoàn thành.",
                        "HR": "👥 **Nhân sự:** Nhấn mạnh kinh nghiệm quản lý người và giải quyết xung đột.",
                        "Finance": "💰 **Tài chính:** Cần các chứng chỉ CPA, CFA và kinh nghiệm phân tích tài chính."
                    }
                    
                    if cv_predicted_label in feedback:
                        st.markdown(feedback[cv_predicted_label])
                    else:
                        st.info("📋 Hãy chắc chắn rằng CV nêu bật được kỹ năng và kinh nghiệm liên quan đến vị trí ứng tuyển.")
                    
                    # Hiển thị preview một phần CV đã xử lý
                    with st.expander("👀 Xem preview text đã xử lý"):
                        st.text(cv_joined[:500] + "..." if len(cv_joined) > 500 else cv_joined)
        
        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý CV: {str(e)}")
            st.write("Vui lòng thử lại với file khác hoặc kiểm tra định dạng file.")
    
    # ================== CV STATISTICS ==================
    if df_candidates is not None and not df_candidates.empty:
        with st.expander("📊 Thống kê phân loại CV"):
            if 'predicted_label' in df_candidates.columns:
                label_counts = df_candidates['predicted_label'].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Phân bố labels:**")
                    for label, count in label_counts.items():
                        percentage = (count / len(df_candidates)) * 100
                        st.write(f"- **{label}**: {count} ({percentage:.1f}%)")
                
                with col2:
                    # Tạo simple bar chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    label_counts.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                    ax.set_title('Phân bố Classification Labels')
                    ax.set_xlabel('Labels')
                    ax.set_ylabel('Số lượng')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
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
