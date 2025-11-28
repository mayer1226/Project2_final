# 🏍️ HỆ THỐNG TÌM KIẾM VÀ GỢI Ý XE MÁY CŨ

Hệ thống tìm kiếm và gợi ý xe máy cũ thông minh sử dụng Machine Learning, được xây dựng với Streamlit.

## 📋 Mục Lục

- [Tính Năng](#-tính-năng)
- [Công Nghệ](#-công-nghệ)
- [Cài Đặt](#-cài-đặt)
- [Sử Dụng](#-sử-dụng)
- [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
- [Tối Ưu Hiệu Suất](#-tối-ưu-hiệu-suất)
- [Tác Giả](#-tác-giả)

## ✨ Tính Năng

### 1. 🔍 Tìm Kiếm Thông Minh
- **Hybrid Search**: Kết hợp TF-IDF và phân tích nội dung
- **Bộ lọc nâng cao**: 8 tiêu chí lọc (Hãng, Model, Loại xe, Phân khối, Giá, Km, Tuổi, Khu vực)
- **Kết quả phân trang**: 9 xe/trang với điều hướng dễ dàng
- **Tìm kiếm ngữ nghĩa**: Hiểu ngữ cảnh từ mô tả

### 2. 🎯 Gợi Ý Xe Tương Tự
- **Cosine Similarity**: Tính toán độ tương đồng chính xác
- **Top 5 xe tương tự**: Hiển thị với % tương đồng
- **Đa chiều**: Dựa trên giá, km, tuổi, loại xe, thương hiệu

### 3. 🤖 Phân Nhóm Xe (Clustering)
- **K-Means Clustering**: 5 nhóm xe được phân loại tự động
- **Nhãn thông minh**:
  - Nhóm 0: Xe Cũ Giá Rẻ - Km Cao
  - Nhóm 1: Hạng Sang Cao Cấp
  - Nhóm 2: Phổ Thông Đại Trà
  - Nhóm 3: Trung Cao Cấp
  - Nhóm 4: Xe Mới - Ít Sử Dụng
- **Badge màu sắc**: Nhận diện nhanh phân khúc

### 4. 📊 Phân Tích Chuyên Sâu
- **Dashboard KPI**: 5 chỉ số kinh doanh chính
- **6 Tab phân tích**:
  - 📈 Tổng Quan: Histogram giá/tuổi, Ma trận tương quan
  - 💰 Phân Tích Giá: Boxplot, Scatter plot với trendline
  - 🏢 Thương Hiệu: Pie chart, Bar chart, Bảng thống kê
  - 📍 Khu Vực: Top 15 khu vực theo số lượng/giá
  - 🚀 Phân Khúc: Phân bố nhóm, 3D scatter plot
  - 📊 Ma Trận: Heatmap Brand×Cluster, Location×Cluster

### 5. 🔑 Quản Trị
- **Thống kê tổng quan**: Phân bố theo nhóm, hãng, khu vực
- **Xuất dữ liệu**: Export Excel/CSV với filter
- **Quản lý dữ liệu**: Xem và phân tích dataset

## 🛠️ Công Nghệ

### Machine Learning
- **Clustering**: K-Means (K=5) với StandardScaler
- **Text Processing**: TF-IDF Vectorizer (max 5000 features)
- **Similarity**: Cosine Similarity
- **Feature Engineering**: Text + Numeric + Binary features

### Framework & Libraries
- **Streamlit**: Web framework chính (v1.31.0)
- **Pandas**: Xử lý dữ liệu (v2.1.4)
- **Scikit-learn**: ML algorithms (v1.3.2)
- **Matplotlib/Seaborn**: Visualization (v3.8.2/v0.13.1)
- **NumPy**: Tính toán số học (v1.26.3)

### Data Format
- **Parquet**: Lưu trữ dữ liệu chính (nén, nhanh)
- **Joblib**: Cache model và metadata
- **Excel/CSV**: Import/Export

## 📦 Cài Đặt

### 1. Yêu Cầu Hệ Thống
- Python 3.9 - 3.11
- RAM: Tối thiểu 2GB (Khuyến nghị 4GB)
- Disk: 500MB trống

### 2. Clone/Download Project
```bash
# Giải nén hoặc copy folder vào máy
cd C:\temp\Streamlit\Final
```

### 3. Tạo Virtual Environment (Khuyến nghị)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 4. Cài Đặt Dependencies
```bash
pip install -r requirements.txt
```

### 5. Kiểm Tra Files Cần Thiết
Đảm bảo các file sau tồn tại:
```
Final/
├── final_app.py              # App chính
├── df_clustering.parquet      # Dữ liệu chính
├── clustering_model.joblib    # K-Means model
├── clustering_scaler.joblib   # Scaler
├── clustering_info.joblib     # Metadata (labels, stats)
├── hybrid_model.joblib        # Hybrid recommender
├── banner.jpg                 # Banner (optional)
└── requirements.txt
```

## 🚀 Sử Dụng

### Chạy App

```bash
# Chạy với port mặc định (8501)
streamlit run final_app.py

# Chạy với port tùy chỉnh
streamlit run final_app.py --server.port 8503

# Chạy với host cụ thể
streamlit run final_app.py --server.address 0.0.0.0
```

### Truy Cập App
- **Local**: http://localhost:8501
- **Network**: http://<your-ip>:8501

### Sử Dụng Các Trang

#### 🏠 Trang Chủ
- Xem tổng quan thống kê (Tổng xe, Giá TB, Phân khúc, Brands)
- Khám phá 5 nhóm xe với biểu đồ phân bố
- Click vào expander để xem ví dụ xe trong từng nhóm

#### 🔍 Tìm Kiếm
1. Nhập từ khóa vào thanh search (vd: "Honda SH", "tay ga", "dưới 30 triệu")
2. Bấm "🔍 Tìm" hoặc Enter
3. Mở "⚙️ Bộ Lọc Nâng Cao" để tinh chỉnh:
   - Row 1: Hãng, Model, Loại xe, Phân khối
   - Row 2: Giá, Km, Tuổi, Khu vực
4. Xem kết quả phân trang (9 xe/trang)
5. Click "🔍 Xem chi tiết" để xem thông tin đầy đủ

#### 📄 Chi Tiết Xe
- Xem đầy đủ thông tin: Giá, Km, Tuổi, Loại xe, Động cơ, Xuất xứ
- Đọc mô tả chi tiết
- Xem 5 xe tương tự (với % tương đồng)
- Click "🔍 Xem chi tiết xe này" để chuyển xe

#### 📊 Phân Tích
- Xem KPI dashboard (5 chỉ số)
- Chuyển đổi 6 tabs để phân tích đa chiều
- Tất cả biểu đồ đã được tối ưu với cache

#### 🔑 Quản Trị
- Xem thống kê chi tiết
- Filter và export dữ liệu (Excel/CSV)
- Phân tích phân bố theo nhiều tiêu chí

## 📁 Cấu Trúc Dự Án

```
Final/
│
├── final_app.py                    # Main application (2280 lines)
│   ├── HybridBikeRecommender      # Class gợi ý xe hybrid
│   ├── Page Functions             # 7 trang: home, search, detail, analysis, admin, help, about
│   ├── Helper Functions           # search, filter, similarity
│   └── Cache Functions            # Performance optimization
│
├── Data Files
│   ├── df_clustering.parquet      # Main dataset (~6700 records)
│   ├── motorcycles_clustered_v2.parquet  # Backup data
│   └── data_motobikes.xlsx        # Original Excel
│
├── Model Files
│   ├── clustering_model.joblib    # K-Means model (K=5)
│   ├── clustering_scaler.joblib   # RobustScaler
│   ├── clustering_info.joblib     # Labels, stats, metadata
│   └── hybrid_model.joblib        # Hybrid recommender (TF-IDF + Features)
│
├── Config Files
│   ├── requirements.txt           # Python dependencies
│   └── README.md                  # Tài liệu này
│
└── Assets
    └── banner.jpg                 # Banner image (optional)
```

## ⚡ Tối Ưu Hiệu Suất

### 1. Caching Strategy
```python
# Resource cache cho models (load 1 lần)
@st.cache_resource
- load_clustering_model() 
- load_hybrid_model()
- initialize_hybrid_model()

# Data cache cho computations
@st.cache_data
- load_data()
- compute_analysis_metrics()
- get_top_brands()
- get_location_stats()
```

### 2. Data Optimization
- **Parquet format**: Nén tốt, load nhanh hơn CSV 5-10x
- **Lazy loading**: Chỉ load data khi cần
- **Pagination**: 9 items/page thay vì load tất cả
- **Sampling**: 3D plots dùng 300 points/cluster thay vì toàn bộ

### 3. Visualization Optimization
- **Reduced bins**: Histogram 20-30 bins thay vì 50
- **plt.close()**: Giải phóng memory sau mỗi plot (17 vị trí)
- **Conditional rendering**: Chỉ render tab đang active
- **TTL cache**: 1 hour cho cluster info

### 4. Search Optimization
- **Top-K limiting**: Chỉ lấy 50 kết quả tốt nhất
- **Index filtering**: Filter trước khi search
- **Feature caching**: Combined features được cache

### 5. Memory Management
- **Reset index**: Tránh index fragmentation
- **Sparse matrices**: Dùng csr_matrix cho TF-IDF
- **Garbage collection**: plt.close() sau plots

### Metrics Hiệu Suất
- **First load**: ~3-4s (từ ~7-8s)
- **Subsequent loads**: ~1-2s (cache hit)
- **Search**: <500ms cho 50 results
- **Page switch**: <200ms
- **Memory usage**: ~300-400MB stable

## 🔧 Troubleshooting

### App chạy chậm
```bash
# Clear Streamlit cache
rm -rf .streamlit/cache  # Linux/Mac
Remove-Item -Recurse .streamlit/cache  # Windows

# Restart app
streamlit run final_app.py
```

### Import Error
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### File Not Found
```bash
# Kiểm tra files
ls *.parquet *.joblib  # Linux/Mac
dir *.parquet *.joblib  # Windows
```

### Port đã được sử dụng
```bash
# Dùng port khác
streamlit run final_app.py --server.port 8502
```

## 📈 Phát Triển Tương Lai

- [ ] Thêm filter theo ngân sách
- [ ] Tích hợp API giá thị trường
- [ ] Chatbot tư vấn AI
- [ ] Mobile responsive tốt hơn
- [ ] Export PDF báo cáo
- [ ] User authentication
- [ ] Favorites/Wishlist
- [ ] Price prediction model

## 🤝 Đóng Góp

Mọi đóng góp đều được chào đón! Vui lòng:
1. Fork project
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📝 License

Project này được phát triển cho mục đích học tập và nghiên cứu.

## 👥 Tác Giả

**Hoàng Phúc & Bích Thủy**

- 📧 Email: [Your Email]
- 🌐 GitHub: [Your GitHub]

---

## 🎯 Quick Start

```bash
# 1. Clone/Download project
cd C:\temp\Streamlit\Final

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run app
streamlit run final_app.py

# 4. Open browser
# http://localhost:8501
```

**🎉 Chúc bạn sử dụng app thành công!**
