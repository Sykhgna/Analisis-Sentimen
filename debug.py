import pickle
import re
import numpy as np

with open('model_sentimen_lengkap.pkl', 'rb') as f:
    paket = pickle.load(f)

vectorizer = paket['vectorizer']
model_svm  = paket['model_svm']

def preprocessing(teks):
    teks = teks.lower()
    teks = re.sub(r'[^a-z\s]', ' ', teks)
    return teks

teks_asli   = "bpjs ini pelayanannya sangat bagus"
teks_bersih = preprocessing(teks_asli)

print(f"Asli    : {teks_asli}")
print(f"Setelah : {teks_bersih}")

X     = vectorizer.transform([teks_bersih]).toarray()
fitur = vectorizer.get_feature_names_out()
idx   = np.where(X[0] > 0)[0]

print(f"\nKata dikenali ({len(idx)}):")
for i in idx:
    print(f"  '{fitur[i]}' → {X[0][i]:.4f}")

print(f"\nPrediksi SVM: {model_svm.predict(X)[0]}")