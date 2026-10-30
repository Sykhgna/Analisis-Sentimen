-- Skema Database untuk Riwayat Prediksi Sentimen BPJS (SVM & KNN)

CREATE TABLE IF NOT EXISTS riwayat_prediksi (
    id INT AUTO_INCREMENT PRIMARY KEY,
    tanggal TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    komentar TEXT NOT NULL,
    prediksi_svm VARCHAR(10) NOT NULL,
    prediksi_knn VARCHAR(10) NOT NULL,
    confidence_svm FLOAT NOT NULL,
    confidence_knn FLOAT NOT NULL
);
