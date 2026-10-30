import pickle
import traceback

print("Membuka model_sentimen_lengkap2.pkl...")
try:
    with open("model_sentimen_lengkap2.pkl", "rb") as f:
        paket = pickle.load(f)
    print("Keys in pickle:", list(paket.keys()))
    
    vectorizer = paket['vectorizer']
    model_svm = paket['model_svm']
    model_knn = paket['model_knn']
    
    print("Vectorizer type:", type(vectorizer))
    print("SVM model type:", type(model_svm))
    print("KNN model type:", type(model_knn))
    
    # Test prediction
    test_text = "pelayanan bpjs sangat baik dan memuaskan"
    from app import preprocessing
    cleaned = preprocessing(test_text)
    print("Cleaned text:", cleaned)
    X = vectorizer.transform([cleaned])
    print("Sparse X shape:", X.shape)
    
    # Test prediction with sparse matrix
    try:
        pred_svm_sparse = model_svm.predict(X)
        print("SVM sparse prediction:", pred_svm_sparse)
    except Exception as e:
        print("SVM sparse prediction failed:", e)
        
    try:
        pred_knn_sparse = model_knn.predict(X)
        print("KNN sparse prediction:", pred_knn_sparse)
    except Exception as e:
        print("KNN sparse prediction failed:", e)
        
    # Test prediction with dense matrix
    X_dense = X.toarray()
    print("Dense X shape:", X_dense.shape)
    pred_svm_dense = model_svm.predict(X_dense)
    pred_knn_dense = model_knn.predict(X_dense)
    print("SVM dense prediction:", pred_svm_dense)
    print("KNN dense prediction:", pred_knn_dense)
    
except Exception as e:
    print("Error loading or running pickle:")
    traceback.print_exc()
