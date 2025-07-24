from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import io
import mysql.connector
from werkzeug.utils import secure_filename
from datetime import datetime
import cloudinary
import cloudinary.uploader

app = Flask(__name__)
CORS(app)
@app.route('/')
def index():
    return jsonify({"status": "ok", "message": "API is running"}), 200

# Konfigurasi upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'public')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model saat aplikasi dijalankan
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dnn_cabai_kedua.h5')
model = load_model(MODEL_PATH)

# Fungsi untuk koneksi ke database MySQL
def get_db_connection():
    return mysql.connector.connect(
        host=os.environ.get('MYSQLHOST'),
        user=os.environ.get('MYSQLUSER'),
        password=os.environ.get('MYSQLPASSWORD'),
        database=os.environ.get('MYSQLDATABASE'), # Ini akan mengambil 'cabai_klasifikasi'
        port=int(os.environ.get('MYSQLPORT'))
    )

def prepare_image(img, target_size=(256, 256)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisasi jika diperlukan
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        # Baca file sebagai BytesIO
        img_bytes = file.read()
        img = image.load_img(io.BytesIO(img_bytes), target_size=(256, 256))
        img_array = prepare_image(img, target_size=(256, 256))
        preds = model.predict(img_array)
        pred_class = int(np.argmax(preds, axis=1)[0])
        pred_score = float(np.max(preds))
        if pred_score < 0.7:
            return jsonify({'class': -1, 'score': pred_score, 'message': 'Bukan cabai'}), 200
        return jsonify({'class': pred_class, 'score': pred_score}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint POST: Upload gambar & simpan history
# @app.route('/history', methods=['POST'])
# def add_history():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400
#     image_file = request.files['image']
#     if image_file and allowed_file(image_file.filename):
#         filename = secure_filename(image_file.filename)
#         save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         image_file.save(save_path)
#         # Data lain dari form
#         name = request.form.get('name')
#         accuracy = request.form.get('accuracy')
#         date_str = request.form.get('date')  # format: YYYY-MM-DD HH:MM:SS
#         if not (name and accuracy and date_str):
#             return jsonify({'error': 'Missing data'}), 400
#         # Simpan ke database
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         cursor.execute(
#             "INSERT INTO history (image, name, accuracy, date) VALUES (%s, %s, %s, %s)",
#             (f"/{filename}", name, int(accuracy), date_str)
#         )
#         conn.commit()
#         cursor.close()
#         conn.close()
#         return jsonify({'message': 'History added successfully'}), 201
#     else:
#         return jsonify({'error': 'Invalid file type'}), 400

# Ganti endpoint /history POST Anda dengan ini
@app.route('/history', methods=['POST'])
def add_history():
    # 1. Cek data form
    print("Menerima permintaan ke /history")
    name = request.form.get('name')
    accuracy = request.form.get('accuracy')
    date_str = request.form.get('date')

    if not all([name, accuracy, date_str]):
        return jsonify({'error': 'Missing form data'}), 400

    # 2. Cek file
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    conn = None
    try:
        # 3. Upload file ke Cloudinary ☁️
        print("Mengupload file ke Cloudinary...")
        upload_result = cloudinary.uploader.upload(image_file)
        # Dapatkan URL gambar yang aman dari hasil upload
        image_url = upload_result['secure_url']
        print(f"File berhasil diupload. URL: {image_url}")

        # 4. Simpan URL ke database
        print("Mencoba terhubung ke database...")
        conn = get_db_connection()
        cursor = conn.cursor()
        print("Koneksi database berhasil.")

        sql_query = "INSERT INTO history (image, name, accuracy, date) VALUES (%s, %s, %s, %s)"
        data_to_insert = (image_url, name, int(accuracy), date_str) # Simpan image_url

        print(f"Menjalankan query dengan data: {data_to_insert}")
        cursor.execute(sql_query, data_to_insert)

        conn.commit()
        print("Commit berhasil. Data dan URL gambar sudah tersimpan.")

        cursor.close()
        return jsonify({'message': 'History added successfully', 'image_url': image_url}), 201

    except Exception as e:
        print(f"!!!!!! TERJADI ERROR !!!!!: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Terjadi kesalahan di server', 'details': str(e)}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()
            print("Koneksi database ditutup.")

# Endpoint GET: Ambil semua history
@app.route('/history', methods=['GET'])
def get_history():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, image, name, accuracy, date FROM history ORDER BY date DESC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(rows), 200

# Endpoint DELETE: Hapus history berdasarkan id
@app.route('/history/<int:id>', methods=['DELETE'])
def delete_history(id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM history WHERE id = %s", (id,))
    conn.commit()
    affected = cursor.rowcount
    cursor.close()
    conn.close()
    if affected == 0:
        return jsonify({'error': 'Data tidak ditemukan'}), 404
    return jsonify({'message': 'History berhasil dihapus'}), 200

if __name__ == '__main__':
    app.run() 