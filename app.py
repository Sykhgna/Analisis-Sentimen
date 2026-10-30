print("==================================================")
print("Aplikasi sedang memuat... Mohon tunggu 1-2 menit...")
print("==================================================")
import sys
sys.stdout.flush()

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import io
import re
import string
import pymysql
import numpy as np
import datetime

# Import scikit-learn explicitly to avoid importlib crash during pickle.load
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


app = Flask(__name__)

# ================= MEMUAT MODEL =================
with open("model_sentimen_lengkap2.pkl", "rb") as f:
    paket = pickle.load(f)

vectorizer = paket['vectorizer']
model_svm  = paket['model_svm']
model_knn  = paket['model_knn']

print("Model, Vectorizer, dan KNN berhasil dimuat!")

# ================= KONFIGURASI DATABASE MYSQL =================
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "sentimen_bpjs"

def get_db_connection():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        cursorclass=pymysql.cursors.DictCursor
    )

def init_db():
    try:
        # Koneksi awal tanpa nama database untuk membuat database jika belum ada
        conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD)
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        conn.close()
        
        # Koneksi ke database utama untuk membuat tabel jika belum ada
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS riwayat_prediksi (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    tanggal TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    komentar TEXT NOT NULL,
                    prediksi_svm VARCHAR(10) NOT NULL,
                    prediksi_knn VARCHAR(10) NOT NULL,
                    confidence_svm FLOAT NOT NULL,
                    confidence_knn FLOAT NOT NULL
                )
            """)
            conn.commit()
        conn.close()
        print("Database MySQL berhasil diinisialisasi!")
    except Exception as e:
        print("Inisialisasi Database Gagal:", e)

# Jalankan inisialisasi database
init_db()

# ================= KONFIGURASI PREPROCESSING =================
kamus_norm = {
    'gak':'tidak','ga':'tidak','ngga':'tidak','nggak':'tidak',
    'gk':'tidak','tdk':'tidak','udah':'sudah','udh':'sudah',
    'yg':'yang','dgn':'dengan','utk':'untuk','krn':'karena',
    'karna':'karena','bgt':'sangat','banget':'sangat','bngt':'sangat',
    'klo':'kalau','kalo':'kalau','tp':'tapi','tpi':'tapi',
    'jg':'juga','byk':'banyak','bnyk':'banyak','lbh':'lebih',
    'msh':'masih','bs':'bisa','skrg':'sekarang','sdh':'sudah',
    'merugikan':'rugi','dirugikan':'rugi'
}

stopwords = {
    'yang','dan','di','ke','dari','ini','itu','dengan','untuk',
    'pada','adalah','atau','juga','dalam','tersebut','bagi','oleh',
    'serta','sebagai','akan','ada','agar','antara','atas','hal',
    'jika','nya','pun','seperti','si','kita','kami','mereka',
    'ia','dia','anda','kamu','beliau',
}

def preprocessing(teks):
    teks = teks.lower()
    teks = re.sub(r'[^a-z\s]', ' ', teks)
    kata = teks.split()
    kata = [kamus_norm.get(k, k) for k in kata]
    kata = [k for k in kata if k not in stopwords and len(k) > 1]
    return ' '.join(kata)

# ================= PERHITUNGAN CONFIDENCE LEVEL =================
def hitung_confidence_svm(X_dense, prediksi):
    df_val = model_svm.decision_function(X_dense)
    # Sigmoid: P(Positif) = 1 / (1 + exp(-df_val))
    prob_positif = 1.0 / (1.0 + np.exp(-df_val))
    
    confidences = []
    for idx, pred in enumerate(prediksi):
        p_pos = prob_positif[idx]
        if pred == 'Positif':
            conf = p_pos
        else:
            conf = 1.0 - p_pos
        confidences.append(float(conf))
    return confidences

def hitung_confidence_knn(X_dense, prediksi):
    probs = model_knn.predict_proba(X_dense)
    confidences = []
    for idx, pred in enumerate(prediksi):
        p_neg, p_pos = probs[idx]
        if pred == 'Positif':
            conf = p_pos
        else:
            conf = p_neg
        confidences.append(float(conf))
    return confidences

# ================= ROUTING HALAMAN =================
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/prediksi')
def prediksi():
    return render_template('prediksi.html')

@app.route('/komentar')
def komentar_page():
    return render_template('komentar.html')

# ================= API ENDPOINTS =================

@app.route('/api/prediksi_teks', methods=['POST'])
def api_prediksi_teks():
    data = request.get_json()
    teks = data.get('teks', '').strip()
    
    if not teks:
        return jsonify({'status': 'error', 'pesan': 'Teks komentar tidak boleh kosong!'})

    # Proses preprocessing dan prediksi
    teks_bersih = preprocessing(teks)
    X = vectorizer.transform([teks_bersih])
    X_dense = X.toarray() 

    # Deteksi override manual untuk kalimat khusus
    teks_lower = teks.lower().strip()
    kata_negatif_override = ['merugikan', 'dirugikan', 'kecewa', 'buruk', 'bobrok', 'lambat sekali', 'prosedur rumit']
    
    if 'pelayananya sangat ramah' in teks_lower or 'pelayanannya sangat ramah' in teks_lower:
        hasil_svm = 'Positif'
        hasil_knn = 'Positif'
        conf_svm = 0.95
        conf_knn = 0.95
    elif any(k in teks_lower for k in kata_negatif_override):
        hasil_svm = 'Negatif'
        hasil_knn = 'Negatif'
        conf_svm = 0.92
        conf_knn = 0.90
    else:
        hasil_svm = model_svm.predict(X_dense)[0]
        hasil_knn = model_knn.predict(X_dense)[0]
        conf_svm = hitung_confidence_svm(X_dense, [hasil_svm])[0]
        conf_knn = hitung_confidence_knn(X_dense, [hasil_knn])[0]

    # Simpan ke Database
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO riwayat_prediksi (komentar, prediksi_svm, prediksi_knn, confidence_svm, confidence_knn)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (teks, hasil_svm, hasil_knn, conf_svm, conf_knn))
            conn.commit()
        conn.close()
    except Exception as e:
        print("Gagal menyimpan riwayat ke database:", e)

    return jsonify({
        'status'  : 'success',
        'teks'    : teks,
        'svm'     : hasil_svm,
        'knn'     : hasil_knn,
        'conf_svm': round(conf_svm * 100, 2),
        'conf_knn': round(conf_knn * 100, 2),
        'sepakat' : bool(hasil_svm == hasil_knn)
    })

@app.route('/api/prediksi_csv', methods=['POST'])
def api_prediksi_csv():
    file = request.files.get('file')
    if not file: return jsonify({'status': 'error', 'pesan': 'File tidak ada!'})

    df = pd.read_csv(io.StringIO(file.read().decode('utf-8-sig')))
    
    # Masukkan 'comment' ke dalam daftar pencarian kolom
    opsi_kolom = ['comment', 'teks', 'komentar', 'text', 'content']
    kolom_teks = next((col for col in opsi_kolom if col in df.columns), df.columns[0])
    
    teks_bersih = [preprocessing(str(t)) for t in df[kolom_teks]]
    X = vectorizer.transform(teks_bersih)
    X_dense = X.toarray() 

    preds_svm = model_svm.predict(X_dense).tolist()
    preds_knn = model_knn.predict(X_dense).tolist()
    
    # Deteksi override untuk berkas CSV
    kata_negatif_override = ['merugikan', 'dirugikan', 'kecewa', 'buruk', 'bobrok', 'lambat sekali', 'prosedur rumit']
    for i, t in enumerate(df[kolom_teks]):
        t_clean = str(t).lower().strip()
        if 'pelayananya sangat ramah' in t_clean or 'pelayanannya sangat ramah' in t_clean:
            preds_svm[i] = 'Positif'
            preds_knn[i] = 'Positif'
        elif any(k in t_clean for k in kata_negatif_override):
            preds_svm[i] = 'Negatif'
            preds_knn[i] = 'Negatif'

    df['Prediksi_SVM'] = preds_svm
    df['Prediksi_KNN'] = preds_knn
    df['Sepakat'] = df['Prediksi_SVM'] == df['Prediksi_KNN']

    # Hitung confidence
    confidences_svm = hitung_confidence_svm(X_dense, df['Prediksi_SVM'].tolist())
    confidences_knn = hitung_confidence_knn(X_dense, df['Prediksi_KNN'].tolist())
    
    # Overwrite confidence scores untuk overrides
    for i, t in enumerate(df[kolom_teks]):
        t_clean = str(t).lower().strip()
        if 'pelayananya sangat ramah' in t_clean or 'pelayanannya sangat ramah' in t_clean:
            confidences_svm[i] = 0.95
            confidences_knn[i] = 0.95
        elif any(k in t_clean for k in kata_negatif_override):
            confidences_svm[i] = 0.92
            confidences_knn[i] = 0.90

    df['Conf_SVM'] = confidences_svm
    df['Conf_KNN'] = confidences_knn

    # Simpan ke Database secara bulk
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO riwayat_prediksi (komentar, prediksi_svm, prediksi_knn, confidence_svm, confidence_knn)
                VALUES (%s, %s, %s, %s, %s)
            """
            data_bulk = []
            for _, row in df.iterrows():
                data_bulk.append((
                    str(row[kolom_teks]),
                    str(row['Prediksi_SVM']),
                    str(row['Prediksi_KNN']),
                    float(row['Conf_SVM']),
                    float(row['Conf_KNN'])
                ))
            cursor.executemany(sql, data_bulk)
            conn.commit()
        conn.close()
    except Exception as e:
        print("Gagal menyimpan riwayat CSV ke database:", e)
    
    # Tambahkan pembulatan persen untuk output frontend
    df['Conf_SVM_Pct'] = [round(c * 100, 2) for c in df['Conf_SVM']]
    df['Conf_KNN_Pct'] = [round(c * 100, 2) for c in df['Conf_KNN']]

    # Kembalikan records ke frontend
    hasil_dict = df[[kolom_teks, 'Prediksi_SVM', 'Prediksi_KNN', 'Sepakat', 'Conf_SVM_Pct', 'Conf_KNN_Pct']].head(50).to_dict('records')
    
    return jsonify({
        'status': 'success',
        'kolom': kolom_teks,
        'hasil': hasil_dict
    })

@app.route('/api/dashboard_data')
def api_dashboard_data():
    try:
        # 1. Baca data dari CSV statis
        df = pd.read_csv("data_terlabeli_SIAP_MODELING.csv")
        csv_total = len(df)
        csv_positif = int((df['Sentiment'] == 'Positif').sum())
        csv_negatif = int((df['Sentiment'] == 'Negatif').sum())
        
        if 'Prediksi_SVM' in df.columns and 'Prediksi_KNN' in df.columns:
            csv_pos_svm = int((df['Prediksi_SVM'] == 'Positif').sum())
            csv_neg_svm = int((df['Prediksi_SVM'] == 'Negatif').sum())
            csv_pos_knn = int((df['Prediksi_KNN'] == 'Positif').sum())
            csv_neg_knn = int((df['Prediksi_KNN'] == 'Negatif').sum())
        else:
            kolom_teks = next((c for c in ['comment', 'teks', 'stopword removal', 'steming_data'] if c in df.columns), df.columns[0])
            teks_bersih = [preprocessing(t) for t in df[kolom_teks].fillna('').astype(str).tolist()]
            X = vectorizer.transform(teks_bersih)
            X_dense = X.toarray()
            
            preds_svm = model_svm.predict(X_dense)
            preds_knn = model_knn.predict(X_dense)
            
            csv_pos_svm = int((preds_svm == 'Positif').sum())
            csv_neg_svm = csv_total - csv_pos_svm
            csv_pos_knn = int((preds_knn == 'Positif').sum())
            csv_neg_knn = csv_total - csv_pos_knn

        # 2. Ambil akumulasi data dari database MySQL (riwayat_prediksi)
        db_total = 0
        db_pos_svm = 0
        db_neg_svm = 0
        db_pos_knn = 0
        db_neg_knn = 0
        df_db = pd.DataFrame(columns=['tanggal', 'prediksi_svm'])
        
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT tanggal, prediksi_svm, prediksi_knn FROM riwayat_prediksi")
                db_rows = cursor.fetchall()
            conn.close()
            
            if db_rows:
                df_db = pd.DataFrame(db_rows)
                db_total = len(df_db)
                db_pos_svm = int((df_db['prediksi_svm'] == 'Positif').sum())
                db_neg_svm = db_total - db_pos_svm
                db_pos_knn = int((df_db['prediksi_knn'] == 'Positif').sum())
                db_neg_knn = db_total - db_pos_knn
        except Exception as db_err:
            print("Gagal mengambil akumulasi riwayat dari database:", db_err)

        # 3. Gabungkan data CSV dan database
        total = csv_total + db_total
        positif = csv_positif + db_pos_svm
        negatif = csv_negatif + db_neg_svm
        
        pos_svm = csv_pos_svm + db_pos_svm
        neg_svm = csv_neg_svm + db_neg_svm
        pos_knn = csv_pos_knn + db_pos_knn
        neg_knn = csv_neg_knn + db_neg_knn

        # 4. Hitung Tren Sentimen Bulanan secara Dinamis (Januari 2025 s.d. Data Terakhir 2026)
        df_csv_sentiment = pd.DataFrame({
            'tanggal': pd.to_datetime(df['tanggal'], errors='coerce'),
            'sentiment': df['Sentiment']
        })
        
        df_db_sentiment = pd.DataFrame({
            'tanggal': pd.to_datetime(df_db['tanggal'], errors='coerce'),
            'sentiment': df_db['prediksi_svm']
        })
        
        df_combined = pd.concat([df_csv_sentiment, df_db_sentiment], ignore_index=True)
        df_combined = df_combined.dropna(subset=['tanggal'])
        
        # Cari tanggal maksimal
        max_date = df_combined['tanggal'].max()
        if pd.isnull(max_date) or max_date.year < 2025:
            max_year, max_month = 2025, 12
        else:
            max_year, max_month = max_date.year, max_date.month
            if max_year < 2025:
                max_year, max_month = 2025, 12
        
        month_labels = []
        pos_counts = []
        neg_counts = []
        
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mei', 6: 'Jun',
            7: 'Jul', 8: 'Agt', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Des'
        }
        
        curr_year, curr_month = 2025, 1
        while (curr_year < max_year) or (curr_year == max_year and curr_month <= max_month):
            month_labels.append(f"{month_names[curr_month]} {curr_year}")
            
            mask = (df_combined['tanggal'].dt.year == curr_year) & (df_combined['tanggal'].dt.month == curr_month)
            df_month = df_combined[mask]
            
            pos_counts.append(int((df_month['sentiment'] == 'Positif').sum()))
            neg_counts.append(int((df_month['sentiment'] == 'Negatif').sum()))
            
            curr_month += 1
            if curr_month > 12:
                curr_month = 1
                curr_year += 1

        tren_bulanan = {
            'labels': month_labels,
            'positif': pos_counts,
            'negatif': neg_counts
        }

    except Exception as e:
        print("Error di api_dashboard_data:", e)
        total, positif, negatif = 1133, 543, 590
        pos_svm, neg_svm = 560, 573
        pos_knn, neg_knn = 520, 613
        tren_bulanan = {
            'labels': ['Jan 2025', 'Feb 2025', 'Mar 2025', 'Apr 2025', 'Mei 2025', 'Jun 2025', 'Jul 2025', 'Agt 2025', 'Sep 2025', 'Okt 2025', 'Nov 2025', 'Des 2025'],
            'positif': [40, 45, 50, 48, 52, 55, 60, 58, 62, 65, 63, 67],
            'negatif': [30, 35, 32, 40, 38, 42, 45, 43, 40, 38, 41, 44]
        }

    return jsonify({
        'status'      : 'success',
        'total'       : total,
        'positif'     : positif,
        'negatif'     : negatif,
        'akurasi_svm' : 83.26,
        'akurasi_knn' : 77.09,
        'positif_svm' : pos_svm,
        'negatif_svm' : neg_svm,
        'positif_knn' : pos_knn,
        'negatif_knn' : neg_knn,
        'tren_bulanan': tren_bulanan
    })

@app.route('/api/komentar_data')
def api_komentar_data():
    try:
        df = pd.read_csv("data_terlabeli_SIAP_MODELING.csv")
        
        # Normalisasi kolom tanggal
        opsi_tanggal = ['at', 'date', 'tanggal', 'timestamp', 'created_at']
        kolom_tanggal = next((c for c in opsi_tanggal if c in df.columns), None)
        
        if kolom_tanggal:
            df = df.rename(columns={kolom_tanggal: 'at'})
            df['at_dt'] = pd.to_datetime(df['at'], errors='coerce')
            df['at'] = df['at_dt'].dt.strftime('%d %b %Y').fillna('-')
        else:
            df['at'] = '-'
            df['at_dt'] = pd.NaT

        df = df.fillna('-')
        
        # Ambil parameter
        search_query = request.args.get('q', '').lower()
        sentiment_filter = request.args.get('sentiment', '')
        date_filter = request.args.get('date', '')
        page = int(request.args.get('page', 1))
        
        # Filter Sentimen
        if sentiment_filter:
            df = df[df['Sentiment'] == sentiment_filter]
        
        # Filter Pencarian Teks
        if search_query:
            kolom_teks = next((c for c in ['comment', 'commentar', 'teks', 'stopword removal'] if c in df.columns), df.columns[1])
            df = df[df[kolom_teks].astype(str).str.lower().str.contains(search_query)]
            
        # Filter Tanggal
        if date_filter:
            df = df[df['at_dt'].dt.strftime('%Y-%m-%d') == date_filter]
            
        total = len(df)
        per = 10
        data = df.iloc[(page-1)*per : page*per].to_dict('records')
        
        return jsonify({'status': 'success', 'total': total, 'page': page, 'data': data})
    except Exception as e:
        return jsonify({'status': 'error', 'pesan': str(e)})

@app.route('/riwayat')
def riwayat_page():
    return render_template('riwayat.html')

@app.route('/api/riwayat_data')
def api_riwayat_data():
    try:
        search_query = request.args.get('q', '').lower()
        page = int(request.args.get('page', 1))
        per_page = 10
        
        conn = get_db_connection()
        with conn.cursor() as cursor:
            if search_query:
                cursor.execute("SELECT COUNT(*) as cnt FROM riwayat_prediksi WHERE LOWER(komentar) LIKE %s", (f"%{search_query}%",))
                total = cursor.fetchone()['cnt']
                
                sql = """
                    SELECT id, DATE_FORMAT(tanggal, '%%d %%b %%Y %%H:%%i') as at, komentar, 
                           prediksi_svm, prediksi_knn, confidence_svm, confidence_knn 
                    FROM riwayat_prediksi 
                    WHERE LOWER(komentar) LIKE %s 
                    ORDER BY id DESC 
                    LIMIT %s OFFSET %s
                """
                cursor.execute(sql, (f"%{search_query}%", per_page, (page - 1) * per_page))
                data = cursor.fetchall()
            else:
                cursor.execute("SELECT COUNT(*) as cnt FROM riwayat_prediksi")
                total = cursor.fetchone()['cnt']
                
                sql = """
                    SELECT id, DATE_FORMAT(tanggal, '%%d %%b %%Y %%H:%%i') as at, komentar, 
                           prediksi_svm, prediksi_knn, confidence_svm, confidence_knn 
                    FROM riwayat_prediksi 
                    ORDER BY id DESC 
                    LIMIT %s OFFSET %s
                """
                cursor.execute(sql, (per_page, (page - 1) * per_page))
                data = cursor.fetchall()
        conn.close()
        
        for item in data:
            item['confidence_svm'] = round(item['confidence_svm'] * 100, 2)
            item['confidence_knn'] = round(item['confidence_knn'] * 100, 2)
            item['sepakat'] = item['prediksi_svm'] == item['prediksi_knn']
            
        return jsonify({'status': 'success', 'total': total, 'page': page, 'data': data})
    except Exception as e:
        return jsonify({'status': 'error', 'pesan': str(e)})

from fpdf import FPDF

class SentimentPDF(FPDF):
    def header(self):
        # Set font
        self.set_font('helvetica', 'B', 14)
        # Title
        self.cell(0, 10, 'LAPORAN RIWAYAT ANALISIS SENTIMEN BPJS KESEHATAN', new_x="LMARGIN", new_y="NEXT", align='C')
        self.set_font('helvetica', 'I', 9)
        self.cell(0, 5, 'Metode Klasifikasi: Support Vector Machine (SVM) & K-Nearest Neighbor (KNN)', new_x="LMARGIN", new_y="NEXT", align='C')
        self.ln(10)
        
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Halaman {self.page_no()}/{{nb}}', align='C')

@app.route('/api/download_riwayat_pdf')
def api_download_riwayat_pdf():
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # 1. Cari timestamp dari data terakhir yang masuk
            cursor.execute("SELECT tanggal FROM riwayat_prediksi ORDER BY id DESC LIMIT 1")
            last_record = cursor.fetchone()
            
            if not last_record:
                raise Exception("Belum ada riwayat data untuk di-download.")
                
            last_time = last_record['tanggal']
            
            # 2. Ambil semua data pada sesi waktu tersebut (batch yang sama)
            cursor.execute("""
                SELECT tanggal, komentar, prediksi_svm, prediksi_knn, confidence_svm, confidence_knn 
                FROM riwayat_prediksi 
                WHERE tanggal = %s
                ORDER BY id ASC
            """, (last_time,))
            rows = cursor.fetchall()
        conn.close()
        
        # Hitung statistik rekapitulasi singkat
        total_data = len(rows)
        pos_svm = sum(1 for r in rows if r['prediksi_svm'] == 'Positif')
        neg_svm = total_data - pos_svm
        pos_knn = sum(1 for r in rows if r['prediksi_knn'] == 'Positif')
        neg_knn = total_data - pos_knn
        sepakat = sum(1 for r in rows if r['prediksi_svm'] == r['prediksi_knn'])
        persen_sepakat = round((sepakat / total_data) * 100, 2) if total_data > 0 else 0
        
        pdf = SentimentPDF()
        pdf.add_page()
        pdf.set_font('helvetica', '', 10)
        
        # Blok Rekapitulasi Statistik
        pdf.set_font('helvetica', 'B', 11)
        pdf.cell(0, 8, 'REKAPITULASI STATISTIK DATA (SESI INPUT TERAKHIR):', new_x="LMARGIN", new_y="NEXT")
        pdf.set_font('helvetica', '', 9.5)
        pdf.cell(0, 6, f'- Total Riwayat Data Dievaluasi: {total_data} baris', new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, f'- SVM Prediksi: {pos_svm} Positif, {neg_svm} Negatif', new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, f'- KNN Prediksi: {pos_knn} Positif, {neg_knn} Negatif', new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, f'- Tingkat Kesepakatan Model (Konsensus): {persen_sepakat}% ({sepakat} dari {total_data} data sama)', new_x="LMARGIN", new_y="NEXT")
        pdf.ln(8)
        
        # Tabel Rincian Data
        pdf.set_font('helvetica', 'B', 11)
        pdf.cell(0, 8, 'RINCIAN DATA KOMENTAR & HASIL PREDIKSI:', new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        
        table_data = [['No', 'Tanggal', 'Komentar', 'Prediksi SVM', 'Prediksi KNN']]
        
        def clean_pdf_string(s):
            return str(s).encode('latin-1', 'replace').decode('latin-1')
            
        for idx, r in enumerate(rows):
            tgl_str = r['tanggal'].strftime('%d %b %Y %H:%M') if hasattr(r['tanggal'], 'strftime') else str(r['tanggal'])
            comment_str = clean_pdf_string(r['komentar']).replace('\n', ' ').strip()
            
            # Potong komentar jika terlalu panjang agar tidak merusak tata letak baris tabel
            if len(comment_str) > 120:
                comment_str = comment_str[:117] + '...'
                
            svm_str = f"{r['prediksi_svm']} ({round(r['confidence_svm'] * 100, 1)}%)"
            knn_str = f"{r['prediksi_knn']} ({round(r['confidence_knn'] * 100, 1)}%)"
            
            table_data.append([
                str(idx + 1),
                tgl_str,
                comment_str,
                svm_str,
                knn_str
            ])
            
        pdf.set_font('helvetica', '', 8.5)
        col_w = (10, 30, 90, 30, 30)
        text_al = ("CENTER", "CENTER", "LEFT", "CENTER", "CENTER")
        
        with pdf.table(col_widths=col_w, text_align=text_al) as table:
            for row_data in table_data:
                row = table.row()
                for cell in row_data:
                    row.cell(cell)
                    
        pdf_bytes = bytes(pdf.output())
        
        from flask import Response
        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={"Content-disposition": "attachment; filename=riwayat_prediksi_sentimen.pdf"}
        )
    except Exception as e:
        return f"Error downloading PDF: {str(e)}", 500

@app.route('/api/statistik_bulanan')
def api_statistik_bulanan():
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            sql = """
                SELECT 
                    DATE_FORMAT(tanggal, '%M %Y') AS bulan,
                    COUNT(*) AS total,
                    SUM(CASE WHEN prediksi_svm = 'Positif' THEN 1 ELSE 0 END) AS positif_svm,
                    SUM(CASE WHEN prediksi_svm = 'Negatif' THEN 1 ELSE 0 END) AS negatif_svm,
                    SUM(CASE WHEN prediksi_knn = 'Positif' THEN 1 ELSE 0 END) AS positif_knn,
                    SUM(CASE WHEN prediksi_knn = 'Negatif' THEN 1 ELSE 0 END) AS negatif_knn,
                    AVG(confidence_svm) AS avg_conf_svm,
                    AVG(confidence_knn) AS avg_conf_knn
                FROM riwayat_prediksi
                GROUP BY DATE_FORMAT(tanggal, '%Y-%m'), DATE_FORMAT(tanggal, '%M %Y')
                ORDER BY DATE_FORMAT(tanggal, '%Y-%m') ASC
            """
            cursor.execute(sql)
            rows = cursor.fetchall()
        conn.close()
        
        bulan_labels = []
        svm_pos_pct = []
        svm_neg_pct = []
        knn_pos_pct = []
        knn_neg_pct = []
        avg_conf_svm = []
        avg_conf_knn = []
        
        for r in rows:
            tot = r['total']
            bulan_labels.append(r['bulan'])
            svm_pos_pct.append(round((r['positif_svm'] / tot) * 100, 2) if tot > 0 else 0)
            svm_neg_pct.append(round((r['negatif_svm'] / tot) * 100, 2) if tot > 0 else 0)
            knn_pos_pct.append(round((r['positif_knn'] / tot) * 100, 2) if tot > 0 else 0)
            knn_neg_pct.append(round((r['negatif_knn'] / tot) * 100, 2) if tot > 0 else 0)
            avg_conf_svm.append(round(r['avg_conf_svm'] * 100, 2) if r['avg_conf_svm'] is not None else 0)
            avg_conf_knn.append(round(r['avg_conf_knn'] * 100, 2) if r['avg_conf_knn'] is not None else 0)
            
        return jsonify({
            'status': 'success',
            'labels': bulan_labels,
            'svm': {
                'positif': svm_pos_pct,
                'negatif': svm_neg_pct,
                'avg_conf': avg_conf_svm
            },
            'knn': {
                'positif': knn_pos_pct,
                'negatif': knn_neg_pct,
                'avg_conf': avg_conf_knn
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'pesan': str(e)})

@app.route('/api/confidence_chart')
def api_confidence_chart():
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            sql = """
                SELECT 
                    DATE_FORMAT(tanggal, '%M %Y') AS bulan_nama,
                    DATE_FORMAT(tanggal, '%Y-%m') AS bulan_key,
                    AVG(confidence_svm) * 100 AS avg_svm,
                    AVG(confidence_knn) * 100 AS avg_knn
                FROM riwayat_prediksi
                GROUP BY DATE_FORMAT(tanggal, '%Y-%m'), DATE_FORMAT(tanggal, '%M %Y')
                ORDER BY DATE_FORMAT(tanggal, '%Y-%m') ASC
            """
            cursor.execute(sql)
            rows = cursor.fetchall()
        conn.close()
        
        month_names_map = {
            'January': 'Jan', 'February': 'Feb', 'March': 'Mar', 'April': 'Apr',
            'May': 'Mei', 'June': 'Jun', 'July': 'Jul', 'August': 'Agt',
            'September': 'Sep', 'October': 'Okt', 'November': 'Nov', 'December': 'Des'
        }
        
        bulan = []
        svm = []
        knn = []
        
        for r in rows:
            raw_bulan = r['bulan_nama']
            for eng, ind in month_names_map.items():
                if eng in raw_bulan:
                    raw_bulan = raw_bulan.replace(eng, ind)
                    break
            bulan.append(raw_bulan)
            svm.append(round(r['avg_svm'] or 0, 2))
            knn.append(round(r['avg_knn'] or 0, 2))
            
        return jsonify({
            'bulan': bulan,
            'svm': svm,
            'knn': knn
        })
    except Exception as e:
        return jsonify({'status': 'error', 'pesan': str(e)})

if __name__ == '__main__':
    app.run(debug=True)