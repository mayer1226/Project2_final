"""
HỆ THỐNG TÌM KIẾM VÀ GỢI Ý XE MÁY CŨ
=====================================

Ứng dụng web tìm kiếm và gợi ý xe máy cũ thông minh sử dụng Machine Learning.

Author: Hoàng Phúc & Bích Thủy
Version: 2.0.0
Date: 2025-11-29
Python: 3.9+
Framework: Streamlit 1.31.0

Features:
- Hybrid Search (TF-IDF + Content-based)
- K-Means Clustering (K=5)
- Similar Bike Recommendations
- Advanced Analytics Dashboard
- Admin Panel with Export

Performance:
- Cache optimization with @st.cache_resource and @st.cache_data
- Lazy loading and pagination
- Memory-efficient plotting with plt.close()
- Fast Parquet data format
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from datetime import datetime
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.sparse import csr_matrix, hstack

# ==============================
# 📱 PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Hệ Thống Xe Máy Cũ",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 🤖 HYBRID RECOMMENDER CLASS
# ==============================

class HybridBikeRecommender:
    """
    PHIÊN BẢN TÁCH MODEL / DATAFRAME
    - Model không chứa DataFrame trong file .joblib
    - DataFrame sẽ được load sau và nạp vào model bằng set_dataframe()
    """

    def __init__(self, 
                 tfidf_max_features=5000,
                 brand_model_boost=5,
                 weights=None,
                 verbose=False):

        self.df = None  
        self.tfidf_max_features = tfidf_max_features
        self.brand_model_boost = brand_model_boost
        self.verbose = verbose

        self.weights = weights or {
            "text": 0.35,
            "numeric": 0.45,
            "binary": 0.20
        }

        self.tfidf = None
        self.numeric_scaler = None
        self.text_features = None
        self.numeric_features = None
        self.binary_features = None
        self.combined_features = None

    def set_dataframe(self, df: pd.DataFrame):
        """Gán DataFrame sau khi load model."""
        self.df = df.reset_index(drop=True)

    def build_features(self):
        """Build tất cả features sau khi có DataFrame."""
        if self.df is None:
            raise ValueError("Bạn phải gọi set_dataframe(df) trước khi build features.")

        self.text_features, self.tfidf = self._build_text_features()
        self.numeric_features, self.numeric_scaler = self._build_numeric_features()
        self.binary_features = self._build_binary_features()
        self.combined_features = self._build_combined_matrix()

    def _build_text_features(self):
        df = self.df.copy()
        df["brand_model"] = df["brand"].fillna("") + " " + df["model"].fillna("")
        brand_model_boosted = (df["brand_model"] + " ") * self.brand_model_boost

        # Determine column names (use existing columns)
        col_list = df.columns.tolist()
        vtype_col = "vehicle_type_display" if "vehicle_type_display" in col_list else "vehicle_type"
        engine_col = "engine_capacity_num" if "engine_capacity_num" in col_list else "engine_capacity"
        origin_col = "origin_num" if "origin_num" in col_list else "origin"
        
        # Build clean text
        text_parts = [brand_model_boosted, df["description"].fillna("")]
        
        if vtype_col in col_list:
            text_parts.append(df[vtype_col].fillna("").astype(str))
        if engine_col in col_list:
            text_parts.append(df[engine_col].fillna("").astype(str))
        if origin_col in col_list:
            text_parts.append(df[origin_col].fillna("").astype(str))
        if "location" in col_list:
            text_parts.append(df["location"].fillna(""))
        
        df["clean_text"] = " ".join([""]*len(text_parts))
        for i, part in enumerate(text_parts):
            if i == 0:
                df["clean_text"] = part
            else:
                df["clean_text"] = df["clean_text"] + " " + part

        tfidf = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            dtype=np.float32
        )

        X = tfidf.fit_transform(df["clean_text"])
        return X, tfidf

    def _build_numeric_features(self):
        numeric_cols = ["price", "km_driven", "age"]
        numeric_data = self.df[numeric_cols].fillna(0).astype(np.float32)
        scaler = RobustScaler()
        scaled = scaler.fit_transform(numeric_data).astype(np.float32)
        return scaled, scaler

    def _build_binary_features(self):
        df = self.df
        bool_cols = [
            c for c in df.columns
            if df[c].dropna().isin([0,1,True,False]).all()
        ]

        if not bool_cols:
            return np.zeros((len(df), 0), dtype=np.float32)

        critical = ["xe_chinh_chu"] if "xe_chinh_chu" in bool_cols else []
        normal = [c for c in bool_cols if c not in critical]

        parts = []
        if critical:
            parts.append(df[critical].astype(float).values * 3.0)
        if normal:
            parts.append(df[normal].astype(float).values)

        return np.hstack(parts).astype(np.float32)

    def _build_combined_matrix(self):
        X_text = self.text_features.multiply(self.weights["text"])
        X_num = csr_matrix(self.numeric_features * self.weights["numeric"])
        X_bin = csr_matrix(self.binary_features * self.weights["binary"])
        return hstack([X_text, X_num, X_bin], format="csr")

    def recommend(self, item_id, top_k=5, filter_by_segment=True):
        if self.df is None:
            raise ValueError("Model chưa có DataFrame. Gọi set_dataframe(df) trước.")

        input_vec = self.combined_features[item_id]
        sim = cosine_similarity(input_vec, self.combined_features).flatten()

        # Use vehicle_type_display if vehicle_type doesn't exist
        vtype_col = "vehicle_type_display" if "vehicle_type_display" in self.df.columns else "vehicle_type"
        if filter_by_segment and vtype_col in self.df.columns:
            seg = self.df.iloc[item_id][vtype_col]
            mask = (self.df[vtype_col] == seg).values
            sim[~mask] = -10

        sim[item_id] = -10
        top_idx = np.argsort(sim)[::-1][:top_k]

        out = self.df.iloc[top_idx].copy()
        out["similarity_score"] = sim[top_idx]
        out["position"] = top_idx
        return out.reset_index(drop=True)
    
    def search(self, query, top_k=10):
        """Search using hybrid features"""
        if self.df is None or self.combined_features is None:
            raise ValueError("Model chưa sẵn sàng. Gọi set_dataframe() và build_features() trước.")
        
        query_text = query.lower()
        query_tfidf = self.tfidf.transform([query_text])
        
        query_numeric = np.zeros((1, self.numeric_features.shape[1]), dtype=np.float32)
        query_binary = np.zeros((1, self.binary_features.shape[1]), dtype=np.float32)
        
        query_vec = hstack([
            query_tfidf.multiply(self.weights["text"]),
            csr_matrix(query_numeric * self.weights["numeric"]),
            csr_matrix(query_binary * self.weights["binary"])
        ], format="csr")
        
        similarities = cosine_similarity(query_vec, self.combined_features).flatten()
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = self.df.iloc[top_indices].copy()
        results['search_score'] = similarities[top_indices]
        results['position'] = top_indices
        
        return results.reset_index(drop=True)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

# ==============================
# 🎨 CUSTOM CSS
# ==============================
st.markdown("""
<style>
    /* Main title */
    .main-title {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    /* Cluster badge */
    .cluster-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        margin: 5px;
    }
    
    /* Style for card buttons - make them visible and attractive */
    div[data-testid="column"] button[kind="primary"] {
        margin-top: 0px !important;
    }
    
    /* Bike card */
    .bike-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        background: white;
        transition: all 0.3s ease;
    }
    
    .bike-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, #667eea22 0%, #764ba211 100%);
        border: 1px solid #667eea44;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# 📥 LOAD MODELS & DATA
# ==============================
@st.cache_resource(show_spinner=False, ttl=3600)  # Cache for 1 hour, then reload
def load_clustering_model():
    """Load clustering model (K-Means K=5)"""
    try:
        model = joblib.load('clustering_model.joblib')
        scaler = joblib.load('clustering_scaler.joblib')
        info = joblib.load('clustering_info.joblib')
        return model, scaler, info
    except Exception as e:
        st.error(f"❌ Không thể load clustering model: {e}")
        return None, None, None

@st.cache_resource(show_spinner=False)
def load_hybrid_model():
    """Load hybrid recommender model"""
    try:
        hybrid = HybridBikeRecommender.load('hybrid_model.joblib')
        return hybrid
    except Exception as e:
        st.warning(f"⚠️ Không thể load hybrid model: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_data():
    """Load main dataset"""
    try:
        df = pd.read_parquet('df_clustering.parquet')
        return df
    except Exception as e:
        st.error(f"❌ Không thể load dữ liệu: {e}")
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def initialize_hybrid_model(_hybrid_model, _df):
    """Initialize and build features for hybrid model (cached)"""
    if _hybrid_model is not None and len(_df) > 0:
        _hybrid_model.set_dataframe(_df)
        _hybrid_model.build_features()
    return _hybrid_model

# Load models
cluster_model, cluster_scaler, cluster_info = load_clustering_model()
hybrid_model_raw = load_hybrid_model()
df = load_data()

# Initialize hybrid model with caching
hybrid_model = initialize_hybrid_model(hybrid_model_raw, df)

# Cluster labels and colors - Load from cluster_info if available
if cluster_info and 'cluster_labels' in cluster_info:
    cluster_labels = cluster_info['cluster_labels']
else:
    # Fallback to simplified Vietnamese names
    cluster_labels = {
        0: "Xe Cũ Giá Rẻ - Km Cao",
        1: "Hạng Sang Cao Cấp",
        2: "Phổ Thông Đại Trà",
        3: "Trung Cao Cấp",
        4: "Xe Mới - Ít Sử Dụng"
    }

cluster_colors = {
    0: "#3498db",
    1: "#e74c3c",
    2: "#2ecc71",
    3: "#f39c12",
    4: "#9b59b6"
}

# ==============================
# 🔧 HELPER FUNCTIONS
# ==============================

def search_items(query, df_search, top_k=10):
    """Hybrid search using HybridBikeRecommender model"""
    if len(df_search) == 0:
        return pd.DataFrame()
    
    try:
        # Use hybrid model if available
        if hybrid_model is not None and hybrid_model.combined_features is not None:
            # Get search results with reasonable top_k (optimized)
            search_top_k = min(top_k * 3, 100)  # Get 3x results for filtering
            all_results = hybrid_model.search(query, top_k=search_top_k)
            
            # Filter to only indices in df_search
            search_indices = df_search.index.tolist()
            filtered_results = all_results[all_results.index.isin(search_indices)].head(top_k)
            
            return filtered_results.reset_index(drop=True)
        
        else:
            # Fallback to simple TF-IDF
            search_parts = []
            
            # Add brand
            if 'brand' in df_search.columns:
                search_parts.append(df_search['brand'].fillna(''))
            
            # Add model
            if 'model' in df_search.columns:
                search_parts.append(df_search['model'].fillna(''))
            
            # Add vehicle_type (not vehicle_type_display)
            if 'vehicle_type' in df_search.columns:
                search_parts.append(df_search['vehicle_type'].fillna(''))
            
            # Add description
            if 'description_norm' in df_search.columns:
                search_parts.append(df_search['description_norm'].fillna(''))
            elif 'description' in df_search.columns:
                search_parts.append(df_search['description'].fillna(''))
            
            if not search_parts:
                return df_search.head(top_k).copy()
            
            # Combine all parts
            search_text = search_parts[0]
            for part in search_parts[1:]:
                search_text = search_text + ' ' + part
            
            # TF-IDF
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(search_text)
            query_vec = vectorizer.transform([query])
            
            # Cosine similarity
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[::-1][:top_k]
            results = df_search.iloc[top_indices].copy()
            results['search_score'] = similarities[top_indices]
            results['position'] = top_indices
            
            return results[results['search_score'] > 0]
    except Exception as e:
        return df_search.head(top_k).copy()

def apply_filters(df_filter, brands, models, price_range, vehicle_types, locations):
    """Apply multi-criteria filters"""
    filtered = df_filter.copy()
    
    # Brand filter
    if brands and 'Tất cả' not in brands:
        filtered = filtered[filtered['brand'].isin(brands)]
    
    # Model filter
    if models and 'Tất cả' not in models:
        filtered = filtered[filtered['model'].isin(models)]
    
    # Vehicle type filter
    if vehicle_types and 'Tất cả' not in vehicle_types:
        if 'vehicle_type_display' in filtered.columns:
            filtered = filtered[filtered['vehicle_type_display'].isin(vehicle_types)]
    
    # Location filter
    if locations and 'Tất cả' not in locations:
        filtered = filtered[filtered['location'].isin(locations)]
    
    # Price range filter
    if price_range:
        min_price, max_price = price_range
        filtered = filtered[(filtered['price'] >= min_price) & (filtered['price'] <= max_price)]
    
    return filtered

def get_similar_bikes(bike_idx, df, top_k=5):
    """Get similar bikes using hybrid model or fallback to numerical similarity"""
    try:
        # Use hybrid model if available
        if hybrid_model is not None and hybrid_model.combined_features is not None:
            similar = hybrid_model.recommend(bike_idx, top_k=top_k, filter_by_segment=True)
            return similar
        
        else:
            # Fallback to simple numerical similarity
            # Features: price, km_driven, age
            features = df[['price', 'km_driven', 'age']].copy()
            
            # Standardize
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Cosine similarity
            similarities = cosine_similarity([features_scaled[bike_idx]], features_scaled)[0]
            
            # Exclude itself
            similarities[bike_idx] = -1
            
            # Get top_k
            top_indices = similarities.argsort()[::-1][:top_k]
            similar_bikes = df.iloc[top_indices].copy()
            similar_bikes['similarity'] = similarities[top_indices]
            
            return similar_bikes
    except:
        return pd.DataFrame()

def get_cluster_badge(cluster_id, cluster_name, cluster_color):
    """Generate cluster badge HTML"""
    return f"""
<div style="
    background-color:{cluster_color};
    display:inline-block;
    color:white;
    padding:8px 15px;
    border-radius:6px;
    font-weight:bold;
    margin:10px 0;">
    🚀 {cluster_name}
</div>
"""

def format_price(price):
    """Format giá tiền"""
    return f"{price:.1f} triệu VNĐ"

def format_km(km):
    """Format số km"""
    return f"{int(km):,} km"

def show_banner():
    """Display banner image from GitHub"""
    try:
        # URL raw của file banner trên GitHub
        # Format: https://raw.githubusercontent.com/USERNAME/REPO_NAME/BRANCH/path/to/banner.jpg
        banner_url = "https://github.com/mayer1226/Project2_final/blob/bc7887280349efeeae5ecb0585fe4485a19ea324/banner.jpg"
        
        st.image(banner_url, use_container_width=True)
    except Exception as e:
        # Fallback: Hiển thị banner HTML nếu không load được ảnh
        st.markdown("""
        <div class="main-title">
            <h1 style='color: white; margin: 0;'>🏍️ HỆ THỐNG MUA BÁN XE MÁY CŨ</h1>
            <p style='color: white; margin: 10px 0 0 0;'>Tìm kiếm thông minh với Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)

# ==============================
# 📄 PAGE FUNCTIONS
# ==============================

def show_home_page():
    """Trang chủ"""
    st.header("🏠 Trang Chủ")
    
    # Stats overview
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin:0; color:#667eea;">🏍️</h2>
            <h3 style="margin:5px 0;">{len(df):,}</h3>
            <p style="margin:0; color:#666;">Tổng số xe</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_price = df['price'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin:0; color:#667eea;">💰</h2>
            <h3 style="margin:5px 0;">{avg_price:.1f}M</h3>
            <p style="margin:0; color:#666;">Giá trung bình</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        n_clusters = df['cluster'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin:0; color:#667eea;">🚀</h2>
            <h3 style="margin:5px 0;">{n_clusters}</h3>
            <p style="margin:0; color:#666;">Phân khúc</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        n_brands = df['brand'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin:0; color:#667eea;">🏢</h2>
            <h3 style="margin:5px 0;">{n_brands}</h3>
            <p style="margin:0; color:#666;">Thương hiệu</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        avg_age = df['age'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin:0; color:#667eea;">📅</h2>
            <h3 style="margin:5px 0;">{avg_age:.1f}</h3>
            <p style="margin:0; color:#666;">Tuổi TB (năm)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 5 Clusters overview
    st.subheader("🚀 5 Phân Khúc Xe Máy")
    
    for cluster_id in sorted(cluster_labels.keys()):
        cluster_name = cluster_labels[cluster_id]
        cluster_color = cluster_colors[cluster_id]
        cluster_data = df[df['cluster'] == cluster_id]
        
        if len(cluster_data) == 0:
            continue
        
        st.markdown(get_cluster_badge(cluster_id, cluster_name, cluster_color), unsafe_allow_html=True)
        with st.expander(f"Chi tiết: {cluster_name}", expanded=False):
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("Số lượng", f"{len(cluster_data):,} xe")
            
            with col_b:
                avg_price_cluster = cluster_data['price'].mean()
                st.metric("Giá TB", format_price(avg_price_cluster))
            
            with col_c:
                avg_km_cluster = cluster_data['km_driven'].mean()
                st.metric("Km TB", format_km(avg_km_cluster))
            
            with col_d:
                avg_age_cluster = cluster_data['age'].mean()
                st.metric("Tuổi TB", f"{avg_age_cluster:.1f} năm")
            
            # Sample bikes
            st.write("**Ví dụ xe trong nhóm:**")
            sample_bikes = cluster_data.head(3)
            for _, bike in sample_bikes.iterrows():
                st.markdown(f"- {bike['brand']} {bike['model']} - {format_price(bike['price'])} - {format_km(bike['km_driven'])}")
    
    st.markdown("---")
    
    # Recent bikes
    st.subheader("🆕 Xe Mới Nhất")
    recent_bikes = df.sort_values('age').head(9)
    
    cols = st.columns(3)
    for idx, (_, bike) in enumerate(recent_bikes.iterrows()):
        col = cols[idx % 3]
        with col:
            cluster_id = bike['cluster']
            cluster_name = cluster_labels.get(cluster_id, 'N/A')
            cluster_color = cluster_colors.get(cluster_id, '#667eea')
            
            st.markdown(f"""
<div class="bike-card">
    <span style="background-color:{cluster_color}; color:white; padding:3px 8px; border-radius:4px; font-size:11px;">
        {cluster_name}
    </span>
    <h4 style="margin:5px 0;">{bike['brand']} {bike['model']}</h4>
    <p style="margin:3px 0; font-size:14px;">
        💰 <strong>{format_price(bike['price'])}</strong><br>
        📏 {format_km(bike['km_driven'])} | 📅 {int(bike['age'])} năm
    </p>
</div>
""", unsafe_allow_html=True)

def show_search_page():
    """Trang tìm kiếm"""
    st.header("🔍 Tìm Kiếm Xe Máy")
    
    # Search bar - compact
    col_search1, col_search2 = st.columns([5, 1])
    with col_search1:
        query = st.text_input(
            "🔍 Tìm kiếm xe", 
            value="", 
            placeholder="Nhập tên xe, hãng, model, loại xe hoặc mô tả...", 
            key="search_query",
            label_visibility="collapsed"
        )
    with col_search2:
        search_clicked = st.button("🔍 Tìm", use_container_width=True, type="primary")
    
    # Filters - Ultra compact layout
    with st.expander("⚙️ Bộ Lọc Nâng Cao", expanded=False):
        # Row 1: Basic filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            all_brands = ['Tất cả'] + sorted(df['brand'].unique().tolist())
            selected_brands = st.multiselect("🏢 Hãng", options=all_brands, default=['Tất cả'])
        
        with col2:
            if selected_brands and 'Tất cả' not in selected_brands:
                available_models = df[df['brand'].isin(selected_brands)]['model'].unique().tolist()
            else:
                available_models = df['model'].unique().tolist()
            
            all_models = ['Tất cả'] + sorted(available_models)
            selected_models = st.multiselect("📦 Model", options=all_models, default=['Tất cả'])
        
        with col3:
            if 'vehicle_type_display' in df.columns:
                all_vehicle_types = ['Tất cả'] + sorted(df['vehicle_type_display'].dropna().unique().tolist())
                selected_vehicle_types = st.multiselect("🏷️ Loại xe", options=all_vehicle_types, default=['Tất cả'])
            else:
                selected_vehicle_types = ['Tất cả']
        
        with col4:
            if 'engine_capacity_num' in df.columns:
                all_engine_capacities = ['Tất cả'] + sorted([str(x) for x in df['engine_capacity_num'].dropna().unique().tolist()])
                selected_engine_capacities = st.multiselect("⚙️ Phân khối", options=all_engine_capacities, default=['Tất cả'])
            else:
                selected_engine_capacities = ['Tất cả']
        
        # Row 2: Ranges + Location
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            col5a, col5b = st.columns(2)
            with col5a:
                min_price = st.number_input("💰 Giá từ", min_value=0.0, max_value=float(df['price'].max()), 
                                           value=float(df['price'].min()), step=1.0, key="min_price")
            with col5b:
                max_price = st.number_input("đến", min_value=0.0, max_value=float(df['price'].max()), 
                                           value=float(df['price'].max()), step=1.0, key="max_price")
        
        with col6:
            col6a, col6b = st.columns(2)
            with col6a:
                min_km = st.number_input("🛣️ Km từ", min_value=0.0, max_value=float(df['km_driven'].max()), 
                                        value=0.0, step=5000.0, key="min_km")
            with col6b:
                max_km = st.number_input("đến", min_value=0.0, max_value=float(df['km_driven'].max()), 
                                        value=float(df['km_driven'].max()), step=5000.0, key="max_km")
        
        with col7:
            col7a, col7b = st.columns(2)
            with col7a:
                min_age = st.number_input("📅 Tuổi từ", min_value=0.0, max_value=float(df['age'].max()), 
                                         value=0.0, step=1.0, key="min_age")
            with col7b:
                max_age = st.number_input("đến", min_value=0.0, max_value=float(df['age'].max()), 
                                         value=float(df['age'].max()), step=1.0, key="max_age")
        
        with col8:
            all_locations = ['Tất cả'] + sorted(df['location'].unique().tolist())
            selected_locations = st.multiselect("📍 Khu vực", options=all_locations, default=['Tất cả'])
    
    # Search query persistence
    if search_clicked and query:
        st.session_state.last_query = query
    
    # Determine current query
    current_query = query if query else st.session_state.get('last_query', '')
    
    # Apply filters
    price_range = (min_price, max_price)
    temp_df = df.copy()
    
    # Filter by engine capacity
    if 'engine_capacity_num' in df.columns and selected_engine_capacities and 'Tất cả' not in selected_engine_capacities:
        temp_df = temp_df[temp_df['engine_capacity_num'].astype(str).isin(selected_engine_capacities)]
    
    # Filter by km_driven
    temp_df = temp_df[(temp_df['km_driven'] >= min_km) & (temp_df['km_driven'] <= max_km)]
    
    # Filter by age
    temp_df = temp_df[(temp_df['age'] >= min_age) & (temp_df['age'] <= max_age)]
    
    filtered_df = apply_filters(temp_df, selected_brands, selected_models, price_range, selected_vehicle_types, selected_locations)
    
    # Search
    if current_query:
        results = search_items(current_query, filtered_df, top_k=50)
        if len(results) > 0:
            st.info(f"🔍 Đang tìm kiếm: **{current_query}**")
    else:
        results = filtered_df.head(50).copy()
        if len(results) > 0:
            results['position'] = results.index.tolist()
    
    if query:
        st.session_state.last_query = query
    
    if len(results) == 0:
        st.warning("⚠️ Không tìm thấy xe phù hợp. Vui lòng thử điều chỉnh bộ lọc hoặc từ khóa.")
        filtered_df = df.head(0)
    else:
        if 'position' not in results.columns:
            results['position'] = results.index.tolist()
        filtered_df = results
    
    # Display results
    if len(filtered_df) > 0:
        st.subheader(f"📋 Kết quả ({len(filtered_df)} xe)")
        
        # Pagination
        if 'search_page_num' not in st.session_state:
            st.session_state.search_page_num = 0
        
        items_per_page = 9
        total_pages = (len(filtered_df) + items_per_page - 1) // items_per_page
        
        col_prev, col_page, col_next = st.columns([1, 2, 1])
        
        with col_prev:
            if st.button("⬅️ Trang trước", disabled=st.session_state.search_page_num == 0, key="prev_page"):
                st.session_state.search_page_num -= 1
                st.rerun()
        
        with col_page:
            st.markdown(f"<p style='text-align:center;'>Trang {st.session_state.search_page_num + 1} / {total_pages}</p>", 
                       unsafe_allow_html=True)
        
        with col_next:
            if st.button("➡️ Trang sau", disabled=st.session_state.search_page_num >= total_pages - 1, key="next_page"):
                st.session_state.search_page_num += 1
                st.rerun()
        
        # Display bikes
        start_idx = st.session_state.search_page_num * items_per_page
        end_idx = start_idx + items_per_page
        page_bikes = filtered_df.iloc[start_idx:end_idx]
        
        cols = st.columns(3)
        for idx, (_, bike) in enumerate(page_bikes.iterrows()):
            col = cols[idx % 3]
            with col:
                cluster_id = bike['cluster']
                cluster_name = cluster_labels.get(cluster_id, 'N/A')
                cluster_color = cluster_colors.get(cluster_id, '#667eea')
                
                bike_position = bike.get('position', bike.name)
                
                # Get description preview (first 80 chars)
                description = ""
                desc_col = None
                
                if 'description_norm' in bike.index:
                    desc_col = 'description_norm'
                elif 'description' in bike.index:
                    desc_col = 'description'
                
                if desc_col and pd.notna(bike[desc_col]):
                    desc_text = str(bike[desc_col]).strip()
                    if desc_text and desc_text.lower() != 'nan':
                        description = desc_text[:80] + "..." if len(desc_text) > 80 else desc_text
                
                # Create clickable card using container + button
                card_container = st.container()
                with card_container:
                    # Build description HTML if exists
                    desc_html = ""
                    if description:
                        desc_html = f"<p style='margin:8px 0 0 0; font-size:12px; color:#888; font-style:italic; line-height:1.4; height:34px; overflow:hidden;'>📝 {description}</p>"
                    else:
                        # Empty placeholder to maintain consistent height
                        desc_html = "<div style='height:34px;'></div>"
                    
                    st.markdown(f"""
<div style="
    background: white;
    border: 2px solid {cluster_color};
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    min-height: 220px;
    display: flex;
    flex-direction: column;
">
    <span style="
        background-color:{cluster_color}; 
        color:white; 
        padding:5px 12px; 
        border-radius:6px; 
        font-size:11px; 
        font-weight:600;
        display: inline-block;
        margin-bottom: 10px;
    ">
        {cluster_name}
    </span>
    <h4 style="margin:8px 0 5px 0; color:#2c3e50; font-size:16px; height:40px; overflow:hidden;">{bike['brand']} {bike['model']}</h4>
    <p style="margin:5px 0; font-size:14px; color:#555; line-height:1.6;">
        💰 <strong style="color:#667eea; font-size:15px;">{format_price(bike['price'])}</strong><br>
        📏 {format_km(bike['km_driven'])} | 📅 {int(bike['age'])} năm
    </p>
    {desc_html}
</div>
""", unsafe_allow_html=True)
                    
                    # Click button
                    if st.button("🔍 Xem chi tiết", key=f"card_{bike_position}", 
                               use_container_width=True):
                        st.session_state.selected_bike_idx = int(bike_position)
                        st.session_state.page = "detail"
                        st.rerun()

def show_detail_page():
    """Trang chi tiết xe"""
    
    if st.session_state.get('selected_bike_idx') is None:
        st.error("❌ Không tìm thấy thông tin xe. Vui lòng chọn xe từ trang tìm kiếm.")
        if st.button("← Quay lại tìm kiếm", key="back_error1"):
            st.session_state.page = "search"
            st.rerun()
        return
    
    bike_idx = st.session_state.selected_bike_idx
    
    if bike_idx >= len(df):
        st.error("❌ Index xe không hợp lệ.")
        if st.button("← Quay lại tìm kiếm", key="back_error2"):
            st.session_state.page = "search"
            st.rerun()
        return
    
    bike = df.iloc[bike_idx]
    
    # Scroll to top component
    st.components.v1.html("""
        <script>
            window.parent.document.querySelector('.main').scrollTo({top: 0, behavior: 'smooth'});
        </script>
    """, height=0)
    
    # Back button
    if st.button("← Quay lại tìm kiếm"):
        st.session_state.page = "search"
        st.rerun()
    
    st.markdown("---")
    
    # Title
    st.title(f"{bike['brand']} {bike['model']}")
    
    # Cluster badge
    cluster_id = bike['cluster']
    cluster_name = cluster_labels.get(cluster_id, 'Chưa phân loại')
    cluster_color = cluster_colors.get(cluster_id, '#667eea')
    
    st.markdown(f"""
<div style="
    background-color:{cluster_color};
    display:inline-block;
    color:white;
    padding:8px 15px;
    border-radius:6px;
    font-weight:bold;
    margin-top:5px;
    margin-bottom:15px;">
    🚀 {cluster_name}
</div>
""", unsafe_allow_html=True)
    
    # Main info
    st.markdown("### 💳 Thông Tin Chính")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Giá bán", format_price(bike['price']))
    col2.metric("📏 Số km đã đi", format_km(bike['km_driven']))
    col3.metric("📅 Tuổi xe", f"{int(bike['age'])} năm")
    
    # Use vehicle_type_display if available
    vtype_col = "vehicle_type_display" if "vehicle_type_display" in bike.index else "vehicle_type"
    if vtype_col in bike.index and pd.notna(bike[vtype_col]):
        col4.metric("🏷️ Loại xe", bike[vtype_col])
    
    st.markdown("---")
    
    # Detailed info
    st.markdown("### 📋 Thông Tin Chi Tiết")
    
    col_x, col_y = st.columns(2)
    
    with col_x:
        info_parts = [f"- **🏢 Thương hiệu:** {bike['brand']}", f"- **🏍️ Model:** {bike['model']}"]
        
        # Use engine_capacity_num if engine_capacity doesn't exist
        engine_col = "engine_capacity" if "engine_capacity" in bike.index else "engine_capacity_num"
        if engine_col in bike.index and pd.notna(bike[engine_col]):
            info_parts.append(f"- **⚙️ Dung tích động cơ:** {bike[engine_col]}")
        st.markdown('\n'.join(info_parts))
    
    with col_y:
        info_parts2 = []
        
        # Use origin_num if origin doesn't exist
        origin_col = "origin" if "origin" in bike.index else "origin_num"
        if origin_col in bike.index and pd.notna(bike[origin_col]):
            info_parts2.append(f"- **🌍 Xuất xứ:** {bike[origin_col]}")
        
        info_parts2.append(f"- **📍 Địa điểm:** {bike['location']}")
        st.markdown('\n'.join(info_parts2))
    
    st.markdown("---")
    
    # Description - Always show
    st.markdown("### 📝 Mô Tả Chi Tiết")
    
    # Try different description columns
    desc_text = ""
    if 'description_norm' in bike.index and pd.notna(bike['description_norm']) and str(bike['description_norm']).strip():
        desc_text = str(bike['description_norm'])
    elif 'description' in bike.index and pd.notna(bike['description']) and str(bike['description']).strip():
        desc_text = str(bike['description'])
    
    if desc_text:
        st.write(desc_text)
    else:
        st.info("ℹ️ Chưa có mô tả chi tiết cho xe này.")
    
    st.markdown("---")
    
    # Similar bikes
    st.markdown("## 🎯 Xe Tương Tự Bạn Có Thể Quan Tâm")
    
    similar_bikes = get_similar_bikes(bike_idx, df, top_k=5)
    
    if len(similar_bikes) > 0:
        for idx, sim_bike in similar_bikes.iterrows():
            sim_cluster_id = sim_bike.get('cluster', 0)
            sim_cluster_name = cluster_labels.get(sim_cluster_id, 'Chưa phân loại')
            sim_cluster_color = cluster_colors.get(sim_cluster_id, '#667eea')
            similarity_score = sim_bike.get('similarity_score', sim_bike.get('similarity', 0))
            
            # Get actual index from similar_bikes
            similar_idx = sim_bike.name if hasattr(sim_bike, 'name') else idx
            
            st.markdown(f"""
<div style="
    background: white;
    border-left: 5px solid {sim_cluster_color};
    padding: 20px;
    margin: 15px 0;
    border-radius: 12px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.12);
">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 200px;">
            <div style="margin-bottom: 10px;">
                <strong style="font-size: 18px; color:#2c3e50;">{sim_bike['brand']} {sim_bike['model']}</strong>
            </div>
            <span style="
                background: linear-gradient(135deg, {sim_cluster_color} 0%, {sim_cluster_color}dd 100%);
                color:white;
                padding:5px 12px;
                border-radius:6px;
                font-size:11px;
                font-weight:600;
                display: inline-block;
            ">
                {sim_cluster_name}
            </span>
        </div>
        <div style="text-align: right; margin-top: 10px;">
            <div style="
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color:white;
                padding:10px 18px;
                border-radius:10px;
                font-weight:700;
                font-size:16px;
            ">
                {similarity_score*100:.1f}% tương tự
            </div>
        </div>
    </div>
    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee; font-size:15px; color:#555;">
        <span style="margin-right: 20px;">
            💰 <strong style="color:#667eea;">{format_price(sim_bike['price'])}</strong>
        </span>
        <span style="margin-right: 20px;">
            📏 {format_km(sim_bike['km_driven'])}
        </span>
        <span>
            📅 {int(sim_bike['age'])} năm
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
            
            # Button underneath card
            if st.button("🔍 Xem chi tiết xe này", key=f"similar_{similar_idx}", use_container_width=True, type="primary"):
                st.session_state.selected_bike_idx = similar_idx
                st.session_state.scroll_to_top = True
                st.rerun()
    else:
        st.info("ℹ️ Không tìm thấy xe tương tự.")

@st.cache_data(show_spinner=False)
def get_top_brands(df_input, n=10):
    """Cache top brands"""
    return df_input['brand'].value_counts().head(n)

@st.cache_data(show_spinner=False)
def get_location_stats(df_input, n=15):
    """Cache location statistics"""
    counts = df_input['location'].value_counts().head(n)
    prices = df_input.groupby('location')['price'].mean().sort_values(ascending=False).head(n)
    return counts, prices

@st.cache_data(show_spinner=False)
def compute_analysis_metrics(df_input):
    """Cache các metrics cơ bản để tránh tính lại"""
    return {
        'total_bikes': len(df_input),
        'avg_price': df_input['price'].mean(),
        'median_price': df_input['price'].median(),
        'total_value': df_input['price'].sum(),
        'avg_km': df_input['km_driven'].mean(),
        'avg_age': df_input['age'].mean()
    }

def show_analysis_page():
    """Trang phân tích chuyên sâu cho quản lý - Optimized"""
    st.header("📊 Phân Tích Thị Trường Chuyên Sâu")
    
    # Get cached metrics
    metrics = compute_analysis_metrics(df)
    
    # KPIs Dashboard
    st.markdown("### 🎯 Chỉ Số Kinh Doanh Chính")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        st.metric("🏍️ Tổng số xe", f"{metrics['total_bikes']:,}")
    
    with kpi2:
        st.metric("💰 Giá TB", f"{metrics['avg_price']:.1f}M", delta=f"Median: {metrics['median_price']:.1f}M")
    
    with kpi3:
        st.metric("💵 Tổng giá trị", f"{metrics['total_value']:,.0f}M")
    
    with kpi4:
        st.metric("🛣️ Km TB", f"{metrics['avg_km']:,.0f}")
    
    with kpi5:
        st.metric("📅 Tuổi TB", f"{metrics['avg_age']:.1f} năm")
    
    st.markdown("---")
    
    # Tabs for different analysis - Only load active tab
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Tổng Quan", "💰 Phân Tích Giá", "🏢 Thương Hiệu", 
        "📍 Khu Vực", "🚀 Phân Khúc", "📊 Ma Trận"
    ])
    
    with tab1:
        st.subheader("📈 Tổng Quan Thị Trường")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution
            with st.spinner('Đang vẽ biểu đồ giá...'):
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(df['price'], bins=30, color='#667eea', alpha=0.7, edgecolor='black')
                ax.axvline(metrics['avg_price'], color='red', linestyle='--', linewidth=2, label=f"Trung bình: {metrics['avg_price']:.1f}M")
                ax.axvline(metrics['median_price'], color='green', linestyle='--', linewidth=2, label=f"Trung vị: {metrics['median_price']:.1f}M")
                ax.set_xlabel('Giá (triệu VNĐ)')
                ax.set_ylabel('Số lượng xe')
            ax.set_title('Phân Bố Giá Xe', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Age distribution
            with st.spinner('Đang vẽ biểu đồ tuổi xe...'):
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(df['age'], bins=20, color='#f39c12', alpha=0.7, edgecolor='black')
                ax.axvline(metrics['avg_age'], color='red', linestyle='--', linewidth=2, label=f"Trung bình: {metrics['avg_age']:.1f} năm")
                ax.set_xlabel('Tuổi xe (năm)')
                ax.set_ylabel('Số lượng xe')
                ax.set_title('Phân Bố Tuổi Xe', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        # Correlation heatmap
        st.markdown("#### 🔥 Ma Trận Tương Quan")
        numeric_cols = ['price', 'km_driven', 'age']
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
        
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(['Giá', 'Km đã đi', 'Tuổi xe'])
        ax.set_yticklabels(['Giá', 'Km đã đi', 'Tuổi xe'])
        
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=12, fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        ax.set_title('Ma Trận Tương Quan Giữa Các Biến', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with tab2:
        st.subheader("💰 Phân Tích Giá Chi Tiết")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price by cluster - boxplot
            st.markdown("#### 📦 Phân Bố Giá Theo Phân Khúc")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            cluster_prices = [df[df['cluster'] == i]['price'].values for i in sorted(cluster_labels.keys())]
            positions = range(len(cluster_labels))
            
            bp = ax.boxplot(cluster_prices, positions=positions, patch_artist=True, widths=0.6)
            
            for patch, cluster_id in zip(bp['boxes'], sorted(cluster_labels.keys())):
                patch.set_facecolor(cluster_colors.get(cluster_id, '#667eea'))
                patch.set_alpha(0.7)
            
            for whisker in bp['whiskers']:
                whisker.set(linewidth=1.5)
            for cap in bp['caps']:
                cap.set(linewidth=1.5)
            for median in bp['medians']:
                median.set(color='red', linewidth=2)
            
            ax.set_xticks(positions)
            ax.set_xticklabels([cluster_labels.get(i, f'Nhóm {i}')[:20] for i in sorted(cluster_labels.keys())], 
                              rotation=45, ha='right', fontsize=10)
            ax.set_ylabel('Giá (triệu VNĐ)', fontsize=11)
            ax.set_title('Phân Bố Giá Theo Phân Khúc', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Price statistics by cluster
            st.markdown("#### 📊 Thống Kê Giá Theo Phân Khúc")
            price_stats = []
            for cluster_id in sorted(cluster_labels.keys()):
                cluster_data = df[df['cluster'] == cluster_id]
                price_stats.append({
                    'Phân khúc': cluster_labels.get(cluster_id, f'Nhóm {cluster_id}')[:30],
                    'Số xe': len(cluster_data),
                    'Giá TB': f"{cluster_data['price'].mean():.1f}M",
                    'Giá Min': f"{cluster_data['price'].min():.1f}M",
                    'Giá Max': f"{cluster_data['price'].max():.1f}M",
                    'Trung vị': f"{cluster_data['price'].median():.1f}M"
                })
            
            stats_df = pd.DataFrame(price_stats)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Price vs Km scatter with trend
        st.markdown("#### 📉 Giá Theo Km Đã Đi")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for cluster_id in sorted(cluster_labels.keys()):
            cluster_data = df[df['cluster'] == cluster_id]
            ax.scatter(cluster_data['km_driven'], cluster_data['price'],
                      color=cluster_colors.get(cluster_id, '#667eea'),
                      label=cluster_labels.get(cluster_id, f'Nhóm {cluster_id}')[:25],
                      alpha=0.5, s=30)
        
        # Add trend line
        z = np.polyfit(df['km_driven'], df['price'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(df['km_driven'].min(), df['km_driven'].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Xu hướng', alpha=0.8)
        
        ax.set_xlabel('Km đã đi', fontsize=11)
        ax.set_ylabel('Giá (triệu VNĐ)', fontsize=11)
        ax.set_title('Mối Quan Hệ Giữa Giá và Km Đã Đi', fontsize=13, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        st.subheader("🏢 Phân Tích Thương Hiệu")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Top brands pie chart
            st.markdown("#### 🥧 Top 10 Thương Hiệu")
            top_brands = df['brand'].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.cm.Set3(range(len(top_brands)))
            wedges, texts, autotexts = ax.pie(top_brands.values, labels=top_brands.index,
                                               autopct='%1.1f%%', colors=colors, startangle=90,
                                               textprops={'fontsize': 10})
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('Thị Phần Top 10 Thương Hiệu', fontsize=13, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Brand statistics
            st.markdown("#### 📊 Thống Kê Thương Hiệu")
            brand_stats = []
            for brand in top_brands.head(10).index:
                brand_data = df[df['brand'] == brand]
                brand_stats.append({
                    'Thương hiệu': brand,
                    'Số xe': len(brand_data),
                    'Giá TB': f"{brand_data['price'].mean():.1f}M",
                    'Km TB': f"{brand_data['km_driven'].mean():,.0f}",
                    'Tuổi TB': f"{brand_data['age'].mean():.1f}"
                })
            
            brand_df = pd.DataFrame(brand_stats)
            st.dataframe(brand_df, use_container_width=True, hide_index=True)
        
        # Average price by top brands
        st.markdown("#### 💰 Giá Trung Bình Theo Thương Hiệu")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        avg_prices = df.groupby('brand')['price'].mean().sort_values(ascending=False).head(15)
        bars = ax.barh(range(len(avg_prices)), avg_prices.values, color='#667eea', alpha=0.7)
        ax.set_yticks(range(len(avg_prices)))
        ax.set_yticklabels(avg_prices.index, fontsize=10)
        ax.set_xlabel('Giá trung bình (triệu VNĐ)', fontsize=11)
        ax.set_title('Top 15 Thương Hiệu Theo Giá Trung Bình', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, val) in enumerate(zip(bars, avg_prices.values)):
            ax.text(val, i, f' {val:.1f}M', va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.pyplot(fig)
        plt.close()
    
    with tab4:
        st.subheader("📍 Phân Tích Theo Khu Vực")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Location distribution
            st.markdown("#### 🗺️ Phân Bố Xe Theo Khu Vực")
            location_counts = df['location'].value_counts().head(15)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(range(len(location_counts)), location_counts.values, 
                          color='#2ecc71', alpha=0.7)
            ax.set_yticks(range(len(location_counts)))
            ax.set_yticklabels(location_counts.index, fontsize=10)
            ax.set_xlabel('Số lượng xe', fontsize=11)
            ax.set_title('Top 15 Khu Vực Có Nhiều Xe Nhất', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            for i, (bar, val) in enumerate(zip(bars, location_counts.values)):
                ax.text(val, i, f' {val:,}', va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Average price by location
            st.markdown("#### 💰 Giá Trung Bình Theo Khu Vực")
            location_prices = df.groupby('location')['price'].mean().sort_values(ascending=False).head(15)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(range(len(location_prices)), location_prices.values,
                          color='#e74c3c', alpha=0.7)
            ax.set_yticks(range(len(location_prices)))
            ax.set_yticklabels(location_prices.index, fontsize=10)
            ax.set_xlabel('Giá trung bình (triệu VNĐ)', fontsize=11)
            ax.set_title('Top 15 Khu Vực Giá Cao Nhất', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            for i, (bar, val) in enumerate(zip(bars, location_prices.values)):
                ax.text(val, i, f' {val:.1f}M', va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.pyplot(fig)
            plt.close()
        
        # Location statistics table
        st.markdown("#### 📊 Bảng Thống Kê Khu Vực")
        location_stats = df.groupby('location').agg({
            'price': ['count', 'mean', 'median'],
            'km_driven': 'mean',
            'age': 'mean'
        }).round(1)
        
        location_stats.columns = ['Số xe', 'Giá TB (M)', 'Giá median (M)', 'Km TB', 'Tuổi TB']
        location_stats = location_stats.sort_values('Số xe', ascending=False).head(20)
        st.dataframe(location_stats, use_container_width=True)
    
    with tab5:
        st.subheader("🚀 Phân Tích Phân Khúc")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster distribution
            st.markdown("#### 📊 Số Lượng Xe Theo Phân Khúc")
            cluster_dist = df['cluster'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors_list = [cluster_colors.get(i, '#667eea') for i in cluster_dist.index]
            
            bars = ax.bar(range(len(cluster_dist)), cluster_dist.values, color=colors_list, alpha=0.8, edgecolor='black')
            ax.set_xticks(range(len(cluster_dist)))
            ax.set_xticklabels([f'Nhóm {i}' for i in cluster_dist.index], fontsize=10)
            ax.set_ylabel('Số lượng xe', fontsize=11)
            ax.set_title('Phân Bố Xe Theo Phân Khúc', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Cluster characteristics
            st.markdown("#### 🎯 Đặc Điểm Phân Khúc")
            cluster_char = []
            for cluster_id in sorted(cluster_labels.keys()):
                cluster_data = df[df['cluster'] == cluster_id]
                cluster_char.append({
                    'Cụm': f'{cluster_id}',
                    'Tên': cluster_labels.get(cluster_id, f'Nhóm {cluster_id}')[:30],
                    'Số xe': f"{len(cluster_data):,}",
                    'Giá TB': f"{cluster_data['price'].mean():.1f}M",
                    'Km TB': f"{cluster_data['km_driven'].mean():,.0f}",
                    'Tuổi TB': f"{cluster_data['age'].mean():.1f}"
                })
            
            cluster_df = pd.DataFrame(cluster_char)
            st.dataframe(cluster_df, use_container_width=True, hide_index=True)
        
        # 3D scatter plot (Age vs Km vs Price) by cluster
        st.markdown("#### 🌐 Mối Quan Hệ 3D: Tuổi - Km - Giá")
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for cluster_id in sorted(cluster_labels.keys()):
            cluster_data = df[df['cluster'] == cluster_id].sample(min(300, len(df[df['cluster'] == cluster_id])))
            ax.scatter(cluster_data['age'], cluster_data['km_driven'], cluster_data['price'],
                      c=cluster_colors.get(cluster_id, '#667eea'),
                      label=cluster_labels.get(cluster_id, f'Nhóm {cluster_id}')[:20],
                      alpha=0.6, s=20)
        
        ax.set_xlabel('Tuổi xe (năm)', fontsize=10)
        ax.set_ylabel('Km đã đi', fontsize=10)
        ax.set_zlabel('Giá (triệu)', fontsize=10)
        ax.set_title('Phân Bố 3D Theo Tuổi - Km - Giá', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.pyplot(fig)
        plt.close()
    
    with tab6:
        st.subheader("📊 Ma Trận Phân Tích")
        
        # Price range distribution
        st.markdown("#### 💵 Phân Bố Theo Khoảng Giá")
        price_ranges = pd.cut(df['price'], bins=[0, 10, 20, 30, 50, 100, 500], 
                             labels=['<10M', '10-20M', '20-30M', '30-50M', '50-100M', '>100M'])
        price_range_dist = price_ranges.value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(range(len(price_range_dist)), price_range_dist.values, 
                         color='#9b59b6', alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(price_range_dist)))
            ax.set_xticklabels(price_range_dist.index, fontsize=10)
            ax.set_ylabel('Số lượng xe', fontsize=11)
            ax.set_title('Phân Bố Theo Khoảng Giá', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Km range distribution
            st.markdown("##### 🛣️ Phân Bố Theo Khoảng Km")
            km_ranges = pd.cut(df['km_driven'], bins=[0, 5000, 10000, 20000, 50000, 100000, 1000000],
                              labels=['<5K', '5-10K', '10-20K', '20-50K', '50-100K', '>100K'])
            km_range_dist = km_ranges.value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(range(len(km_range_dist)), km_range_dist.values,
                         color='#3498db', alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(km_range_dist)))
            ax.set_xticklabels(km_range_dist.index, fontsize=10)
            ax.set_ylabel('Số lượng xe', fontsize=11)
            ax.set_title('Phân Bố Theo Khoảng Km', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.pyplot(fig)
            plt.close()
        
        # Cross-tabulation: Brand vs Cluster
        st.markdown("#### 🔀 Ma Trận: Thương Hiệu × Phân Khúc (Top 10)")
        top_brands_list = df['brand'].value_counts().head(10).index
        cross_tab = pd.crosstab(df[df['brand'].isin(top_brands_list)]['brand'], 
                                df[df['brand'].isin(top_brands_list)]['cluster'])
        
        # Rename columns
        cross_tab.columns = [f'Nhóm {i}' for i in cross_tab.columns]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(cross_tab.values, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(range(len(cross_tab.columns)))
        ax.set_yticks(range(len(cross_tab.index)))
        ax.set_xticklabels(cross_tab.columns, fontsize=10)
        ax.set_yticklabels(cross_tab.index, fontsize=10)
        
        for i in range(len(cross_tab.index)):
            for j in range(len(cross_tab.columns)):
                text = ax.text(j, i, f'{cross_tab.iloc[i, j]}',
                             ha="center", va="center", color="black", fontsize=9, fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        ax.set_title('Phân Bố Thương Hiệu Theo Phân Khúc', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.pyplot(fig)
        plt.close()
    
    with tab1:
        st.subheader("📈 Phân Bố Theo Phân Khúc")
        
        cluster_dist = df['cluster'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_list = [cluster_colors.get(i, '#667eea') for i in cluster_dist.index]
        
        bars = ax.bar(range(len(cluster_dist)), cluster_dist.values, color=colors_list)
        ax.set_xticks(range(len(cluster_dist)))
        ax.set_xticklabels([cluster_labels.get(i, f'Nhóm {i}') for i in cluster_dist.index], rotation=45, ha='right')
        ax.set_ylabel('Số lượng xe')
        ax.set_title('Phân bố xe theo phân khúc')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.pyplot(fig)
        plt.close()
    
    with tab2:
        st.subheader("💰 Phân Tích Giá Theo Phân Khúc")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cluster_prices = [df[df['cluster'] == i]['price'].values for i in sorted(cluster_labels.keys())]
        positions = range(len(cluster_labels))
        
        bp = ax.boxplot(cluster_prices, positions=positions, patch_artist=True)
        
        for patch, cluster_id in zip(bp['boxes'], sorted(cluster_labels.keys())):
            patch.set_facecolor(cluster_colors.get(cluster_id, '#667eea'))
        
        ax.set_xticklabels([cluster_labels.get(i, f'Nhóm {i}') for i in sorted(cluster_labels.keys())], rotation=45, ha='right')
        ax.set_ylabel('Giá (triệu VNĐ)')
        ax.set_title('Phân bố giá theo phân khúc')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.pyplot(fig)
        plt.close()
    
    with tab3:
        st.subheader("📏 Mối Quan Hệ Km & Tuổi Xe")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for cluster_id in sorted(cluster_labels.keys()):
            cluster_data = df[df['cluster'] == cluster_id]
            ax.scatter(cluster_data['age'], cluster_data['km_driven'], 
                      color=cluster_colors.get(cluster_id, '#667eea'),
                      label=cluster_labels.get(cluster_id, f'Nhóm {cluster_id}'),
                      alpha=0.6, s=50)
        
        ax.set_xlabel('Tuổi xe (năm)')
        ax.set_ylabel('Km đã đi')
        ax.set_title('Mối quan hệ giữa tuổi xe và km đã đi')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.pyplot(fig)
        plt.close()
    
    with tab4:
        st.subheader("🏢 Phân Tích Thương Hiệu")
        
        top_brands = df['brand'].value_counts().head(10)
        
        st.markdown("#### 🏆 Top 10 Thương Hiệu")
        
        for idx, (brand, count) in enumerate(top_brands.items(), 1):
            pct = (count / len(df)) * 100
            avg_price = df[df['brand'] == brand]['price'].mean()
            
            st.markdown(f"""
<div style="
    background-color: #f0f0f0;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 10px;
    border-left: 5px solid #667eea;
">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <strong style="font-size: 18px;">#{idx}. {brand}</strong>
            <div style="color: #666; margin-top: 5px;">
                {count:,} xe ({pct:.1f}%) | Giá TB: {avg_price:.1f}M VNĐ
            </div>
        </div>
        <div style="
            background-color: #667eea;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
        ">
            {count:,}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
        
        brand_price = df.groupby('brand')['price'].mean().sort_values(ascending=False)
        st.info(f"""
💡 **Insights:**
- Thương hiệu hàng đầu: **{top_brands.index[0]}** ({top_brands.values[0]:,} xe)
- Giá trung bình cao nhất: **{brand_price.index[0]}** ({brand_price.values[0]:.1f}M VNĐ)
- Tổng số thương hiệu: **{df['brand'].nunique()}**
""")
    
    with tab5:
        st.subheader("📍 Phân Tích Khu Vực")
        
        location_dist = df['location'].value_counts().head(15)
        
        st.markdown("#### 📊 Top 15 Khu Vực")
        
        for location, count in location_dist.items():
            pct = (count / len(df)) * 100
            avg_price = df[df['location'] == location]['price'].mean()
            
            bar_color = "#667eea" if pct >= 10 else "#764ba2" if pct >= 5 else "#a78bfa"
            
            st.markdown(f"""
<div style="margin-bottom: 15px;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
        <span><strong>{location}</strong></span>
        <span><strong>{count:,} xe ({pct:.1f}%) | Giá TB: {avg_price:.1f}M</strong></span>
    </div>
    <div style="
        width: 100%;
        background-color: #e0e0e0;
        border-radius: 5px;
        overflow: hidden;
    ">
        <div style="
            width: {pct}%;
            background-color: {bar_color};
            padding: 10px;
            color: white;
            text-align: center;
            font-weight: bold;
        ">
            {pct:.1f}%
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
        
        st.info(f"""
💡 **Insights:**
- Khu vực có nhiều xe nhất: **{location_dist.index[0]}** ({location_dist.values[0]:,} xe)
- Khu vực có ít xe nhất (top 15): **{location_dist.index[-1]}** ({location_dist.values[-1]:,} xe)
- Tổng số khu vực: **{df['location'].nunique()}**
""")

def show_help_page():
    """Trang hướng dẫn"""
    st.title("📘 Hướng Dẫn Sử Dụng")
    st.markdown("---")
    
    tab_quick, tab_search, tab_detail, tab_cluster, tab_tips = st.tabs([
        "🚀 Bắt Đầu Nhanh",
        "🔍 Tìm Kiếm & Lọc",
        "👁️ Chi Tiết & Gợi Ý",
        "🧠 Phân Nhóm",
        "💡 Mẹo Nhanh"
    ])
    
    with tab_quick:
        st.markdown("### 3 bước để bắt đầu")
        st.markdown("""
1. Vào tab **🔍 Tìm Kiếm** để tìm hoặc lọc xe theo nhu cầu.
2. Dùng **🔧 Bộ Lọc Tìm Kiếm** để thu hẹp theo hãng, model, loại, khu vực, dung tích và khoảng giá.
3. Nhấn **Xem chi tiết** trên một xe để xem mô tả và các gợi ý tương tự.
""")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Đi đến Trang Tìm Kiếm", use_container_width=True, type="primary"):
                st.session_state.page = "search"
                st.rerun()
    
    with tab_search:
        st.markdown("### Cách sử dụng Tìm Kiếm & Bộ Lọc")
        st.markdown("""
- **Tìm kiếm**: Nhập từ khóa (tên, model, loại) vào ô tìm kiếm.
- **Bộ Lọc**: Mở phần "🔧 Bộ Lọc Tìm Kiếm" → chọn Hãng/Model/Loại/Khu vực/Dung tích.
- **Khoảng giá**: Điều chỉnh giá từ/đến (đơn vị: triệu VNĐ).
- **Xuất dữ liệu**: Nhấn "📥 Tải xuống CSV" để tải kết quả tìm kiếm.
- **Thống kê**: Nhấn "📊 Xem thống kê tóm tắt" để xem phân tích nhanh.
- Kết quả sẽ tự động cập nhật hoặc nhấn nút **Tìm kiếm**.
""")
    
    with tab_detail:
        st.markdown("### Xem chi tiết & Xe tương tự")
        st.markdown("""
- Nhấn **Xem chi tiết** để mở trang chi tiết xe.
- Xem **badge Phân Nhóm** màu sắc để biết phân khúc.
- Kéo xuống phần **Xe Tương Tự** để xem gợi ý; độ tương đồng được hiển thị (%)
- Click vào xe gợi ý để xem chi tiết xe đó.
""")
    
    with tab_cluster:
        st.markdown("### Hiểu về phân Nhóm (Cluster)")
        st.markdown("""
- Hệ thống dùng K-Means để gom xe theo đặc trưng (giá, km, tuổi xe...).
- Có **5 cụm**: mỗi Nhóm có tên và màu riêng giúp nhận diện nhanh.
- Lợi ích: tìm xe cùng nhóm, so sánh giá, phân tích thị trường.
""")
        
        st.markdown("#### 🎨 Các Phân Khúc")
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            with col:
                cluster_name = cluster_labels.get(idx, f'Nhóm {idx}')
                cluster_color = cluster_colors.get(idx, '#667eea')
                cluster_count = len(df[df['cluster'] == idx])
                
                st.markdown(f"""
<div style="
    background-color: {cluster_color};
    color: white;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
">
    <div style="font-size: 24px; font-weight: bold;">{cluster_count:,}</div>
    <div style="font-size: 12px; margin-top: 5px;">{cluster_name}</div>
</div>
""", unsafe_allow_html=True)
    
    with tab_tips:
        st.markdown("### 💡 Mẹo & Thủ Thuật")
        st.markdown("""
- **Tìm kiếm thông minh**: Nhập từ khóa ngắn gọn (ví dụ: 'SH 2020' thay vì 'Honda SH Mode 2020').
- **Kết hợp bộ lọc**: Dùng nhiều bộ lọc cùng lúc để tìm chính xác hơn (ví dụ: Hãng + Khoảng giá).
- **Xuất dữ liệu**: Tải CSV để phân tích thêm trên Excel/Google Sheets.
- **Xem thống kê**: Dùng nút thống kê để biết giá trung bình, min/max nhanh chóng.
- **So sánh xe**: Mở nhiều tab chi tiết để so sánh các xe khác nhau.
- **Phân tích thị trường**: Vào tab **📊 Phân Tích** để xem tổng quan thị trường.
- **Lọc theo dung tích**: Sử dụng bộ lọc dung tích động cơ để tìm xe phù hợp với nhu cầu.
""")
        
        st.success("💡 **Mẹo Pro**: Bookmark những xe yêu thích bằng cách lưu link trang chi tiết!")

def show_admin_page():
    """Trang quản trị viên"""
    st.header("🔑 Trang Quản Trị Viên")
    
    # Password protection
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.warning("🔒 Vui lòng đăng nhập để truy cập trang quản trị")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            password = st.text_input("🔑 Mật khẩu", type="password", key="admin_password")
            
            if st.button("✅ Đăng nhập", use_container_width=True):
                # Simple password check (in production, use proper authentication)
                if password == "admin123":  # Change this password!
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Mật khẩu không chính xác!")
        
        st.info("💡 **Gợi ý:** Mật khẩu mặc định là 'admin123'")
        return
    
    # Logout button
    if st.button("🚪 Đăng xuất", key="logout_btn"):
        st.session_state.admin_authenticated = False
        st.rerun()
    
    st.markdown("---")
    
    # Admin tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Thống kê tổng quan",
        "💾 Xuất dữ liệu",
        "🛠️ Quản lý hệ thống",
        "📈 Báo cáo chi tiết"
    ])
    
    with tab1:
        st.subheader("📊 Thống Kê Tổng Quan")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Tổng số xe", f"{len(df):,}")
        
        with col2:
            avg_price = df['price'].mean()
            st.metric("💰 Giá trung bình", f"{avg_price:.1f}M")
        
        with col3:
            avg_km = df['km_driven'].mean()
            st.metric("📍 Km trung bình", f"{avg_km:,.0f} km")
        
        with col4:
            avg_age = df['age'].mean()
            st.metric("📅 Tuổi trung bình", f"{avg_age:.1f} năm")
        
        st.markdown("---")
        
        # Cluster distribution
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("### 📈 Phân bố theo cụm")
            cluster_counts = df['cluster'].value_counts().sort_index()
            cluster_data = pd.DataFrame({
                'Cụm': [cluster_labels.get(i, f'Nhóm {i}') for i in cluster_counts.index],
                'Số lượng': cluster_counts.values,
                'Tỉ lệ (%)': (cluster_counts.values / len(df) * 100).round(2)
            })
            st.dataframe(cluster_data, use_container_width=True, hide_index=True)
        
        with col_b:
            st.markdown("### 🏭 Top 5 thương hiệu")
            brand_counts = df['brand'].value_counts().head(5)
            brand_data = pd.DataFrame({
                'Thương hiệu': brand_counts.index,
                'Số lượng': brand_counts.values,
                'Tỉ lệ (%)': (brand_counts.values / len(df) * 100).round(2)
            })
            st.dataframe(brand_data, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("💾 Xuất Dữ Liệu")
        
        st.markdown("### 🎯 Chọn bộ lọc để xuất")
        
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            export_brands = st.multiselect(
                "Thương hiệu",
                options=['Tất cả'] + sorted(df['brand'].unique().tolist()),
                default=['Tất cả'],
                key="admin_export_brands"
            )
        
        with col_filter2:
            export_clusters = st.multiselect(
                "Cụm",
                options=['Tất cả'] + [cluster_labels.get(i, f'Nhóm {i}') for i in sorted(df['cluster'].unique())],
                default=['Tất cả'],
                key="admin_export_clusters"
            )
        
        # Apply filters for export
        export_df = df.copy()
        
        if export_brands and 'Tất cả' not in export_brands:
            export_df = export_df[export_df['brand'].isin(export_brands)]
        
        if export_clusters and 'Tất cả' not in export_clusters:
            cluster_ids = [k for k, v in cluster_labels.items() if v in export_clusters]
            export_df = export_df[export_df['cluster'].isin(cluster_ids)]
        
        st.info(f"📊 Số lượng xe sau khi lọc: **{len(export_df):,}**")
        
        st.markdown("---")
        st.markdown("### 📄 Chọn cột để xuất")
        
        all_export_cols = ['brand', 'model', 'price', 'km_driven', 'age', 'location', 'cluster']
        
        # Add optional columns if they exist
        if 'vehicle_type_display' in export_df.columns:
            all_export_cols.append('vehicle_type_display')
        if 'engine_capacity_num' in export_df.columns:
            all_export_cols.append('engine_capacity_num')
        if 'origin_num' in export_df.columns:
            all_export_cols.append('origin_num')
        if 'description' in export_df.columns:
            all_export_cols.append('description')
        
        selected_cols = st.multiselect(
            "Cột dữ liệu",
            options=all_export_cols,
            default=['brand', 'model', 'price', 'km_driven', 'age', 'location', 'cluster'],
            key="admin_selected_cols"
        )
        
        if selected_cols:
            # Prepare export dataframe
            final_export_df = export_df[selected_cols].copy()
            
            # Add cluster name
            if 'cluster' in selected_cols:
                final_export_df['cluster_name'] = final_export_df['cluster'].map(cluster_labels)
            
            # Rename columns to Vietnamese
            col_rename = {
                'brand': 'Hãng',
                'model': 'Model',
                'price': 'Giá (triệu)',
                'km_driven': 'Km đã đi',
                'age': 'Tuổi xe',
                'location': 'Khu vực',
                'cluster': 'Mã nhóm',
                'cluster_name': 'Tên nhóm',
                'vehicle_type_display': 'Loại xe',
                'engine_capacity_num': 'Dung tích',
                'origin_num': 'Xuất xứ',
                'description': 'Mô tả'
            }
            
            final_export_df = final_export_df.rename(
                columns={k: v for k, v in col_rename.items() if k in final_export_df.columns}
            )
            
            # Preview
            st.markdown("### 👀 Xem trước")
            st.dataframe(final_export_df.head(10), use_container_width=True)
            
            st.markdown("---")
            
            # Export buttons
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            
            with col_exp1:
                csv = final_export_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "📅 Tải xuống CSV",
                    data=csv,
                    file_name=f"admin_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_exp2:
                excel_buffer = pd.io.common.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    final_export_df.to_excel(writer, index=False, sheet_name='Data')
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    "📊 Tải xuống Excel",
                    data=excel_data,
                    file_name=f"admin_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col_exp3:
                json_data = final_export_df.to_json(orient='records', force_ascii=False).encode('utf-8')
                st.download_button(
                    "📝 Tải xuống JSON",
                    data=json_data,
                    file_name=f"admin_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    with tab3:
        st.subheader("🛠️ Quản Lý Hệ Thống")
        
        st.markdown("### 💾 Thông tin hệ thống")
        
        col_sys1, col_sys2 = st.columns(2)
        
        with col_sys1:
            st.markdown(f"""
            - **Tổng số dòng:** {len(df):,}
            - **Tổng số cột:** {len(df.columns)}
            - **Kích thước bộ nhớ:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
            - **Các cột:** {', '.join(df.columns[:10].tolist())}...
            """)
        
        with col_sys2:
            hybrid_status = '✅ Đã load' if hybrid_model else '❌ Chưa load'
            cluster_status = '✅ Đã load' if cluster_model else '❌ Chưa load'
            features_status = '✅ Sẵn sàng' if hybrid_model and hybrid_model.combined_features is not None else '❌ Chưa build'
            
            st.markdown(f"""
            - **Hybrid Model:** {hybrid_status}
            - **Clustering Model:** {cluster_status}
            - **Features built:** {features_status}
            - **Số nhóm:** {len(cluster_labels)}
            """)
        
        st.markdown("---")
        st.markdown("### 🗑️ Cache Management")
        
        col_cache1, col_cache2 = st.columns(2)
        
        with col_cache1:
            if st.button("🔄 Xóa cache dữ liệu", use_container_width=True):
                st.cache_data.clear()
                st.success("✅ Đã xóa cache dữ liệu!")
        
        with col_cache2:
            if st.button("🔄 Xóa cache model", use_container_width=True):
                st.cache_resource.clear()
                st.success("✅ Đã xóa cache model! Vui lòng tải lại trang.")
    
    with tab4:
        st.subheader("📈 Báo Cáo Chi Tiết")
        
        # Detailed statistics
        st.markdown("### 📊 Thống kê chi tiết theo cột")
        
        numeric_cols = ['price', 'km_driven', 'age']
        stats_df = df[numeric_cols].describe().T
        stats_df.columns = ['Số lượng', 'Trung bình', 'Độ lệch chuẩn', 'Min', '25%', '50%', '75%', 'Max']
        st.dataframe(stats_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📏 Phân tích giá trị thiếu (Missing Values)")
        
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Cột': missing_data.index,
            'Số lượng thiếu': missing_data.values,
            'Tỉ lệ (%)': missing_pct.values
        })
        missing_df = missing_df[missing_df['Số lượng thiếu'] > 0].sort_values('Số lượng thiếu', ascending=False)
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
        else:
            st.success("✅ Không có giá trị thiếu trong dữ liệu!")
        
        st.markdown("---")
        st.markdown("### 📍 Phân bố theo khu vực")
        
        location_stats = df.groupby('location').agg({
            'price': ['count', 'mean'],
            'km_driven': 'mean',
            'age': 'mean'
        }).round(2)
        location_stats.columns = ['Số lượng', 'Giá TB', 'Km TB', 'Tuổi TB']
        location_stats = location_stats.sort_values('Số lượng', ascending=False).head(10)
        st.dataframe(location_stats, use_container_width=True)

def show_about_page():
    """Trang giới thiệu"""
    st.header("ℹ️ Giới Thiệu Hệ Thống")
    
    st.markdown("""
    ## 🏍️ Hệ Thống Xe Máy Cũ
    
    ### 🎯 Mục đích
    Hệ thống này giúp bạn:
    - 🔍 Tìm kiếm xe máy cũ phù hợp với nhu cầu
    - 🤖 Phân loại xe theo phân khúc tự động bằng Machine Learning
    - 🎯 Nhận gợi ý xe tương tự dựa trên AI
    - 📊 Phân tích thị trường xe máy cũ
    - 💾 Xuất dữ liệu để phân tích thêm
    
    ### 🤖 Công nghệ
    - **Clustering**: K-Means (5 phân khúc) với StandardScaler
    - **Recommendation**: Cosine Similarity trên đặc trưng giá/km/tuổi
    - **Search**: TF-IDF Vectorization với n-gram
    - **Framework**: Streamlit + Scikit-learn + Pandas
    
    ### ✨ Tính năng nổi bật
    - ⚡ Tìm kiếm thông minh với TF-IDF
    - 🔧 Bộ lọc đa tiêu chí (Hãng, Model, Loại, Khu vực, Dung tích, Giá)
    - 📥 Xuất CSV với timestamp tự động
    - 📊 Thống kê tóm tắt real-time
    - 🎨 HTML progress bars cho phân tích
    - 💡 Insights tự động
    
    ### 👥 Nhóm phát triển
    - 👨‍💻 Hoàng Phúc
    - 👩‍💻 Bích Thủy
    
    ### 📧 Liên hệ
    📧 Email: phucthuy@buonbanxemay.vn
    
    ---
    
    ### 📝 Phiên bản
    **Version 2.0** - Enhanced Edition
    - ✅ Advanced filters với engine capacity
    - ✅ Export CSV functionality
    - ✅ Statistics summary
    - ✅ Search query persistence
    - ✅ Brand & Location analysis
    - ✅ Dedicated Help page
    - ✅ HTML progress bars
    """)
    
    st.markdown("---")
    
    # Cluster info
    st.markdown("### 🎨 Các Phân Khúc")
    
    for cluster_id in sorted(cluster_labels.keys()):
        cluster_name = cluster_labels[cluster_id]
        cluster_color = cluster_colors[cluster_id]
        cluster_data = df[df['cluster'] == cluster_id]
        
        st.markdown(f"""
        <div style="background-color:{cluster_color}22; border-left: 4px solid {cluster_color}; 
                    padding: 15px; margin: 10px 0; border-radius: 5px;">
            <strong style="color:{cluster_color};">🚀 {cluster_name}</strong><br>
            Số lượng: {len(cluster_data):,} xe ({len(cluster_data)/len(df)*100:.1f}%)<br>
            Giá TB: {cluster_data['price'].mean():.1f}M VNĐ
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>💡 Được phát triển bởi <strong>Hoàng Phúc & Bích Thủy</strong></p>
    <p>🚀 Tích hợp Machine Learning Clustering cho phân loại thông minh</p>
    <p>📧 Liên hệ hỗ trợ: phucthuy@buonbanxemay.vn</p>
    <p style='margin-top: 15px; font-size: 12px; color: #999;'>© 2025 Hệ Thống Xe Máy Cũ v2.0</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# 🚀 MAIN APPLICATION
# ==============================

# Banner
show_banner()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "about"
if 'selected_bike_idx' not in st.session_state:
    st.session_state.selected_bike_idx = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'search_page_num' not in st.session_state:
    st.session_state.search_page_num = 0

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🧭 Điều Hướng")
    
    # Navigation mapping (excluding detail page from sidebar)
    nav_map = {
        "🏠 Trang Chủ": "home",
        "🔍 Tìm Kiếm": "search",
        "📊 Phân Tích": "analysis",
        "🔑 Quản Trị": "admin",
        "📘 Hướng Dẫn": "help",
        "📖 Giới Thiệu": "about"
    }
    
    # Determine current selection (map detail to search)
    labels = list(nav_map.keys())
    current_page = st.session_state.page
    
    # If on detail page, show search as selected
    if current_page == "detail":
        current_page = "search"
    
    current_label = None
    for lab, p in nav_map.items():
        if current_page == p:
            current_label = lab
            break
    
    default_index = labels.index(current_label) if current_label in labels else 0
    
    # Radio navigation
    nav_choice = st.radio(
        label="Chọn trang",
        options=labels,
        index=default_index,
        key="nav_radio",
        label_visibility="collapsed"
    )
    
    # Sync page (only if not on detail page or user explicitly changed)
    if 'last_nav_choice' not in st.session_state:
        st.session_state.last_nav_choice = nav_choice
    
    # Only navigate if user explicitly changed selection
    if nav_choice != st.session_state.last_nav_choice:
        new_page = nav_map.get(nav_choice)
        # Only change page if not already on detail page, or user is navigating away
        if st.session_state.page != "detail" or new_page != "search":
            st.session_state.page = new_page
            st.session_state.last_nav_choice = nav_choice
            st.rerun()
    
    # Author info
    st.markdown("---")
    st.markdown("""
        <div style='
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 10px;
            color: white;
            text-align: center;
        '>
            <h4 style='margin: 0 0 10px 0; color: white;'>👥 Tác Giả</h4>
            <p style='margin: 5px 0; font-size: 14px;'>
                <strong>Hoàng Phúc & Bích Thủy</strong>
            </p>
            <hr style='border: 1px solid rgba(255,255,255,0.3); margin: 10px 0;'>
            <p style='margin: 5px 0; font-size: 13px;'>
                📅 <strong>Ngày phát hành:</strong><br>28/11/2025
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==============================
# 🔀 PAGE ROUTING
# ==============================

if st.session_state.page == "about":
    show_about_page()
elif st.session_state.page == "help":
    show_help_page()
elif st.session_state.page == "search":
    show_search_page()
elif st.session_state.page == "admin":
    show_admin_page()
elif st.session_state.page == "detail":
    show_detail_page()
elif st.session_state.page == "analysis":
    show_analysis_page()
else:
    # Default to home
    show_home_page()

# Footer
st.markdown("---")
st.markdown(f"*Hệ thống gợi ý xe máy - Tổng số xe: {len(df):,}*")



