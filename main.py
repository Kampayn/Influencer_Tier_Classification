from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
import json
from werkzeug.exceptions import NotFound

app = Flask(__name__)

MODEL_DIR = 'models'
REQUIRED_FILES = [
    'kmeans_model.joblib',
    'preprocessor.joblib',
    'tf_engagement_model.h5',
    'hasil_clustering_dengan_label.csv',
    'model_metadata.json',
    'pca_model.joblib'
]

missing_files = []
for file in REQUIRED_FILES:
    file_path = os.path.join(MODEL_DIR, file)
    if not os.path.exists(file_path):
        missing_files.append(file)

if missing_files:
    print(f"PERINGATAN: File berikut tidak ditemukan: {', '.join(missing_files)}")
    print("Beberapa fitur API mungkin tidak akan berfungsi.")
else:
    print("Semua file model ditemukan. API siap digunakan.")
try:
    kmeans_model = joblib.load(os.path.join(MODEL_DIR, 'kmeans_model.joblib'))
    print("Model KMeans berhasil dimuat.")
    
    preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.joblib'))
    print("Preprocessor berhasil dimuat.")
    
    tf_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'tf_engagement_model.h5'))
    print("Model TensorFlow berhasil dimuat.")
    
    pca_model = joblib.load(os.path.join(MODEL_DIR, 'pca_model.joblib'))
    print("Model PCA berhasil dimuat.")
    
    df_influencer = pd.read_csv(os.path.join(MODEL_DIR, 'hasil_clustering_dengan_label.csv'))
    print(f"Data influencer berhasil dimuat. Total: {len(df_influencer)} influencer.")
    
    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)
    print("Metadata model berhasil dimuat.")
    
    if 'clustering' in metadata and 'cluster_tiers' in metadata['clustering']:
        cluster_tiers = metadata['clustering']['cluster_tiers']
    else:

        cluster_tiers = {}
        for cluster in df_influencer['Cluster'].unique():
            tier = df_influencer[df_influencer['Cluster'] == cluster]['Tier'].iloc[0]
            cluster_tiers[str(cluster)] = tier
    
    print(f"Informasi tier cluster: {cluster_tiers}")
    
    if 'Mapped_Label' in df_influencer.columns:
        available_categories = df_influencer['Mapped_Label'].unique().tolist()
    else:
        available_categories = df_influencer['Label'].unique().tolist()
    
    print(f"Kategori yang tersedia: {available_categories}")
    
    available_tiers = df_influencer['Tier'].unique().tolist()
    print(f"Tier yang tersedia: {available_tiers}")
    
    MODELS_LOADED = True
    
except Exception as e:
    print(f"Error saat memuat model: {str(e)}")
    print("API akan berjalan dalam mode terbatas.")
    MODELS_LOADED = False

def preprocess_input(data):
    """Fungsi untuk memproses input user menjadi format yang sesuai untuk model"""
    required_features = metadata.get('feature_columns', 
                                   ['Followers', 'Engagement Rate', 'Average Likes', 
                                    'Average Comments', 'Is Verified', 'Is Professional Account', 'Mapped_Label'])
    
    input_df = pd.DataFrame([data])
    
    for col in required_features:
        if col not in input_df.columns:
            if col in ['Followers', 'Engagement Rate', 'Average Likes', 'Average Comments']:
                input_df[col] = 0 
            else:
                input_df[col] = ''  
    
    input_df = input_df[required_features]
    
    return input_df

def predict_cluster(data):
    """Memprediksi cluster berdasarkan data input"""
    input_df = preprocess_input(data)

    input_processed = preprocessor.transform(input_df)
    
    cluster = int(kmeans_model.predict(input_processed)[0])
    
    tier = cluster_tiers.get(str(cluster), "Undefined")
    
    return {
        "cluster": cluster,
        "tier": tier
    }

def predict_engagement(data):
    """Memprediksi engagement rate berdasarkan data input"""
    input_df = preprocess_input(data)
    
    input_processed = preprocessor.transform(input_df)
    
    if hasattr(input_processed, 'toarray'):
        input_processed = input_processed.toarray()
    
    engagement_prediction = float(tf_model.predict(input_processed)[0][0])
    
    return {
        "predicted_engagement_rate": engagement_prediction
    }

def visualize_with_pca(data):
    """Visualisasi posisi data input dalam ruang PCA dengan data lain"""
    input_df = preprocess_input(data)
    
    input_processed = preprocessor.transform(input_df)
    
    if hasattr(input_processed, 'toarray'):
        pca_result = pca_model.transform(input_processed.toarray())
    else:
        pca_result = pca_model.transform(input_processed)
    
    return {
        "pca_coordinates": {
            "x": float(pca_result[0][0]),
            "y": float(pca_result[0][1])
        }
    }

def recommend_influencers(tier=None, category=None, top_n=5, sort_by='Engagement Rate'):
    """Fungsi untuk merekomendasikan influencer berdasarkan tier dan kategori"""
    filtered = df_influencer.copy()
    if tier:
        filtered = filtered[filtered['Tier'] == tier]
        if filtered.empty:
            return []
    
    if category:
        if 'Mapped_Label' in filtered.columns:
            filtered = filtered[filtered['Mapped_Label'] == category]
        else:
            filtered = filtered[filtered['Label'] == category]
        
        if filtered.empty:
            return []
    
    if sort_by in filtered.columns:
        result = filtered.sort_values(by=sort_by, ascending=False).head(top_n)
    else:
        result = filtered.sort_values(by='Engagement Rate', ascending=False).head(top_n)
    
    display_columns = [
        'Username', 'Followers', 'Engagement Rate', 'Average Likes', 
        'Average Comments', 'Mapped_Label' if 'Mapped_Label' in result.columns else 'Label', 
        'Tier', 'Url'
    ]

    display_columns = [col for col in display_columns if col in result.columns]
    
    return result[display_columns].to_dict(orient='records')

@app.route('/')
def index():
    """Root endpoint untuk API"""
    return jsonify({
        "status": "online",
        "message": "Influencer Recommendation API",
        "models_loaded": MODELS_LOADED,
        "endpoints": [
            {"method": "GET", "path": "/health", "description": "Check API health"},
            {"method": "GET", "path": "/metadata", "description": "Get model metadata"},
            {"method": "GET", "path": "/categories", "description": "Get available categories"},
            {"method": "GET", "path": "/tiers", "description": "Get available tiers"},
            {"method": "GET", "path": "/recommend", "description": "Get influencer recommendations"},
            {"method": "POST", "path": "/predict", "description": "Predict cluster, tier and engagement rate"}
        ]
    })

@app.route('/health')
def health_check():
    """Endpoint untuk health check"""
    return jsonify({
        "status": "healthy",
        "models_loaded": MODELS_LOADED,
        "missing_files": missing_files if missing_files else None
    })

@app.route('/metadata')
def get_metadata():
    """Endpoint untuk mendapatkan metadata model"""
    if MODELS_LOADED:
        return jsonify(metadata)
    else:
        return jsonify({"error": "Models not loaded"}), 503

@app.route('/categories')
def get_categories():
    """Endpoint untuk mendapatkan kategori yang tersedia"""
    if MODELS_LOADED:
        return jsonify({"categories": available_categories})
    else:
        return jsonify({"error": "Models not loaded"}), 503

@app.route('/tiers')
def get_tiers():
    """Endpoint untuk mendapatkan tier yang tersedia"""
    if MODELS_LOADED:
        return jsonify({"tiers": available_tiers})
    else:
        return jsonify({"error": "Models not loaded"}), 503

@app.route('/recommend')
def recommend():
    """Endpoint untuk mendapatkan rekomendasi influencer"""
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded"}), 503
    
    try:
        tier = request.args.get('tier')
        category = request.args.get('category')
        top_n = int(request.args.get('top_n', 5))
        sort_by = request.args.get('sort_by', 'Engagement Rate')
        
        if tier and tier not in available_tiers:
            return jsonify({
                "error": f"Invalid tier: {tier}. Available tiers: {available_tiers}"
            }), 400
        
        if category and category not in available_categories:
            return jsonify({
                "error": f"Invalid category: {category}. Available categories: {available_categories}"
            }), 400
        
        recommendations = recommend_influencers(tier, category, top_n, sort_by)
        
        return jsonify({
            "count": len(recommendations),
            "tier": tier,
            "category": category,
            "sort_by": sort_by,
            "recommendations": recommendations
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk memprediksi cluster, tier, dan engagement rate"""
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded"}), 503
    
    try:
        # Ambil data dari request
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validasi data
        required_fields = ['Followers', 'Engagement Rate']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Prediksi cluster dan tier
        cluster_result = predict_cluster(data)
        
        # Prediksi engagement rate
        engagement_result = predict_engagement(data)
        
        # PCA visualization (optional)
        try:
            pca_result = visualize_with_pca(data)
        except:
            pca_result = {"pca_coordinates": None}
        
        # Gabungkan hasil
        result = {
            **data,
            **cluster_result,
            **engagement_result,
            **pca_result
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/visualizations/<path:filename>')
def serve_visualizations(filename):
    """Endpoint untuk mengakses file visualisasi"""
    try:
        return send_from_directory('visualizations', filename)
    except NotFound:
        return jsonify({"error": "Visualization not found"}), 404

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port, debug=True)