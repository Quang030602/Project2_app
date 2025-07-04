import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import re

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from scipy.sparse import hstack

# 1. Load dữ liệu review
df = pd.read_excel('Reviews.xlsx')

# 2. Tiền xử lý dữ liệu (tuỳ dự án, bạn có thể tuỳ chỉnh)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['review_text_clean'] = df['What I liked'].fillna('') + " " + df['Suggestions for improvement'].fillna('')
df['review_text_clean'] = df['review_text_clean'].apply(clean_text)
X_text = df['review_text_clean'].tolist()

# 3. Tạo TFIDF vectorizer và transform
tfidf_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2), min_df=2, max_df=0.8)
X_tfidf = tfidf_vectorizer.fit_transform(X_text)

# 4. Encode nhãn (Recommend? Yes/No)
le = LabelEncoder()
y = le.fit_transform(df['Recommend?'].fillna('No'))  # 1: Yes, 0: No

# 5. Encode company_name (tuỳ pipeline có cần không)
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
X_company = ohe.fit_transform(df[['Company Name']].fillna(''))

# 6. Combine features (TFIDF + OneHot company, nếu pipeline cần)
X_all = hstack([X_tfidf, X_company])

# 7. Train-test split (dùng cho các model dưới)
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y, test_size=0.2, stratify=y, random_state=42
)

# 8. Khởi tạo & train các model
models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight='balanced', C=0.1, random_state=42
    ),
    "knn": KNeighborsClassifier(n_neighbors=7, weights='distance'),
    "decision_tree": DecisionTreeClassifier(
        class_weight='balanced', max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100, class_weight='balanced', max_depth=15, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', random_state=42
    ),
    "svm": SVC(probability=True, class_weight='balanced', C=0.1, kernel='rbf', random_state=42),
    "xgboost": xgb.XGBClassifier(
        n_estimators=100, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
}

print("🔄 Training models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"{name}_model.pkl")
    print(f"✅ Saved: {name}_model.pkl")

# 9. Lưu các encoder/vectorizer (dùng chung cho tất cả model)
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(ohe, 'onehot_encoder.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("✅ Đã train và lưu thành công toàn bộ models và encoder/vectorizer!")
