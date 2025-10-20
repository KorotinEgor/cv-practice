import cv2
import numpy as np
import os
import argparse
import joblib
import json
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, recall_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

# Глобальные переменные
CLASS_NAMES = ['Архангельский собор', 'Дворец труда', 'Нижегородский Кремль']
LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Инициализация объектов
sift = cv2.SIFT_create()
kmeans = None
svm = None
scaler = StandardScaler()
model = None
algorithm = 'bow'
image_size = (224, 224)
n_clusters = 100

def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка загрузки изображения: {image_path}")
        return None
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

def build_vocabulary(all_descriptors):
    global kmeans
    all_descriptors = np.vstack(all_descriptors)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_descriptors)
    
    return kmeans

def descr_to_histogram(descriptors):
    labels = kmeans.predict(descriptors)
    
    histogram, _ = np.histogram(labels, bins=n_clusters, range=(0, n_clusters))
    
    histogram = histogram.astype(float)
    if np.sum(histogram) > 0:
        histogram /= np.sum(histogram)
        
    return histogram

def train_bow(train_paths, train_labels):
    global svm, scaler
    
    all_descriptors = []
    
    for image_path in train_paths:
        descriptors = extract_features(image_path)
        if descriptors is not None:
            all_descriptors.append(descriptors)
    
    build_vocabulary(all_descriptors)

    train_features = []
    for descriptors in all_descriptors:
        histogram = descr_to_histogram(descriptors)
        train_features.append(histogram)
    
    train_features = np.array(train_features)     
    train_features_scaled = scaler.fit_transform(train_features)

    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(train_features_scaled, train_labels)

    train_predictions = svm.predict(train_features_scaled)
    train_tpr = recall_score(train_labels, train_predictions, average="weighted")
    
    return train_tpr

def test_bow(test_paths, test_labels):
    test_features = []
    
    for path in test_paths:
        histogram = descr_to_histogram(extract_features(path))
        test_features.append(histogram)
    
    test_features = np.array(test_features)
    test_features_scaled = scaler.transform(test_features)
    
    test_predictions = svm.predict(test_features_scaled)
    test_tpr = recall_score(test_labels, test_predictions, average="weighted")
    
    return test_tpr, test_predictions

def create_cnn_model():
    base_model = VGG16(weights='imagenet', 
                      include_top=False, 
                      input_shape=(image_size[0], image_size[1], 3))
    
    base_model.trainable = False
    
    cnn_model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])
    
    cnn_model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
    
    return cnn_model

def preprocess_image_cnn(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = image.astype('float32') / 255.0
    
    return image

def train_cnn(train_paths, train_labels, label_ids):
    global model
    
    X_train = []
    y_train = []
    
    for (path, label_id) in zip(train_paths, label_ids):
        image = preprocess_image_cnn(path)
        if image is not None:
            X_train.append(image)
            y_train.append(label_id)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    model = create_cnn_model()
    
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=5
    )
    
    train_predictions = np.argmax(model.predict(X_train), axis=1)
    train_tpr = recall_score(y_train, train_predictions, average="weighted")
    
    return train_tpr

def test_cnn(test_paths, test_labels, label_ids):
    X_test = []
    y_test = []
    
    for (path, label_id) in zip(test_paths, label_ids):
        image = preprocess_image_cnn(path)
        if image is not None:
            X_test.append(image)
            y_test.append(label_id)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    test_predictions_proba = model.predict(X_test)
    test_predictions = np.argmax(test_predictions_proba, axis=1)
    test_tpr = recall_score(y_test, test_predictions, average="weighted")
    
    test_predictions_labels = [CLASS_NAMES[pred] for pred in test_predictions]
    
    return test_tpr, test_predictions_labels

def detect_label_from_path(file_path):
    if '01_NizhnyNovgorodKremlin' in file_path:
        return 'Нижегородский Кремль'
    elif '08_PalaceOfLabor' in file_path:
        return 'Дворец труда'
    elif '04_ArkhangelskCathedral' in file_path:
        return 'Архангельский собор'
    else:
        print(f"Не удалось определить метку для файла: {file_path}")
        return None

def load_data(file_list_path, images_dir="."):
    with open(file_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    image_paths = []
    labels = []
    label_ids = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        full_path = os.path.join(images_dir, line)
        
        if os.path.exists(full_path):
            label = detect_label_from_path(full_path)
            if label is not None:
                image_paths.append(full_path)
                labels.append(label)
                label_ids.append(LABEL_TO_ID[label])
            else:
                print(f"Пропущен файл (не определена метка): {full_path}")
        else:
            print(f"Файл не найден: {full_path}")
    
    print(f"Загружено {len(image_paths)} изображений")
    print(f"Распределение по классам:")
    for label_name in CLASS_NAMES:
        count = labels.count(label_name)
        print(f"  {label_name}: {count} изображений")
    
    return image_paths, labels, label_ids

def train_model(train_file, images_dir=".", alg='bow', clusters=100):
    global algorithm, n_clusters
    algorithm = alg
    n_clusters = clusters
    
    train_paths, train_labels, label_ids = load_data(train_file, images_dir)
    
    if len(train_paths) == 0:
        print("Ошибка: нет данных для обучения!")
        return 0
    
    if algorithm == 'bow':
        train_tpr = train_bow(train_paths, train_labels)
    elif algorithm == 'cnn':
        train_tpr = train_cnn(train_paths, train_labels, label_ids)
    else:
        print(f"Неизвестный алгоритм: {algorithm}")
        return 0
    
    print(f"Точность на обучающей выборке: {train_tpr:.4f}")
    return train_tpr

def test_model(test_file, images_dir="."):
    test_paths, test_labels, label_ids = load_data(test_file, images_dir)
    
    if len(test_paths) == 0:
        print("Ошибка: нет данных для тестирования!")
        return 0
    
    if algorithm == 'bow':
        test_tpr, test_predictions = test_bow(test_paths, test_labels)
    elif algorithm == 'cnn':
        test_tpr, test_predictions = test_cnn(test_paths, test_labels, label_ids)
    else:
        print(f"Неизвестный алгоритм: {algorithm}")
        return 0
    
    print(f"Точность на тестовой выборке: {test_tpr:.4f}")
    
    print("\nДетальный отчет по классификации:")
    print(classification_report(test_labels, test_predictions))
    
    return test_tpr

def save_model(model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    
    if algorithm == 'bow':
        joblib.dump(kmeans, os.path.join(model_dir, 'kmeans_model.pkl'))
        joblib.dump(svm, os.path.join(model_dir, 'svm_model.pkl'))
        joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
        
        metadata = {
            'algorithm': 'bow',
            'n_clusters': n_clusters,
            'class_names': CLASS_NAMES
        }
        
    elif algorithm == 'cnn':
        model.save(os.path.join(model_dir, 'cnn_model.h5'))
        
        metadata = {
            'algorithm': 'cnn',
            'image_size': image_size,
            'class_names': CLASS_NAMES
        }
    
    with open(os.path.join(model_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Модель сохранена в директории: {model_dir}")

def load_model(model_dir="models"):
    global algorithm, CLASS_NAMES, LABEL_TO_ID, kmeans, svm, scaler, model, image_size, n_clusters
    
    with open(os.path.join(model_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    algorithm = metadata['algorithm']
    CLASS_NAMES = metadata['class_names']
    LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    if algorithm == 'bow':
        kmeans = joblib.load(os.path.join(model_dir, 'kmeans_model.pkl'))
        svm = joblib.load(os.path.join(model_dir, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        n_clusters = metadata['n_clusters']
        
    elif algorithm == 'cnn':
        model = tf.keras.models.load_model(os.path.join(model_dir, 'cnn_model.h5'))
        image_size = tuple(metadata['image_size'])
    
    print(f"Модель загружена из директории: {model_dir}")
    print(f"Алгоритм: {algorithm}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Классификация достопримечательностей Нижнего Новгорода')
    parser.add_argument('--data_dir', type=str, default='./NNClassification', help='Путь к директории с изображениями')
    parser.add_argument('--train_file', type=str, default='train_test_split/train.txt', help='Путь к файлу train.txt')
    parser.add_argument('--test_file', type=str, default='train_test_split/test.txt', help='Путь к файлу test.txt')
    parser.add_argument('--algorithm', type=str, choices=['bow', 'cnn'], default='bow',
                       help='Алгоритм классификации: bow (мешок слов) или cnn (нейронная сеть)')
    parser.add_argument('--clusters', type=int, default=100, help='Количество кластеров для метода мешок слов')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both',
                       help='Режим работы: train (обучение), test (тестирование), both (обучение и тестирование)')
    parser.add_argument('--model_dir', type=str, default='models', help='Директория для сохранения/загрузки моделей')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both'] and not args.train_file:
        parser.error("Для режима обучения требуется указать --train_file")
    
    if args.mode in ['test', 'both'] and not args.test_file:
        parser.error("Для режима тестирования требуется указать --test_file")
    
    print(f"Режим работы: {args.mode}")
    print(f"Алгоритм: {args.algorithm}")
    print(f"Директория с данными: {args.data_dir}")
    
    train_tpr = 0
    test_tpr = 0
    
    if args.mode in ['train', 'both']:
        train_tpr = train_model(args.train_file, args.data_dir, args.algorithm, args.clusters)
        
        save_model(args.model_dir)
    
    if args.mode in ['test', 'both']:
        if args.mode == 'test':
            if not os.path.exists(args.model_dir):
                print(f"Ошибка: директория с моделью {args.model_dir} не существует!")
                return
            load_model(args.model_dir)
        
        test_tpr = test_model(args.test_file, args.data_dir)
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ:")
    print(f"Алгоритм: {args.algorithm}")
    print(f"Режим работы: {args.mode}")
    
    if args.mode in ['train', 'both']:
        print(f"TPR на обучающей выборке: {train_tpr:.4f}")
    
    if args.mode in ['test', 'both']:
        print(f"TPR на тестовой выборке: {test_tpr:.4f}")
    
    print("="*50)

if __name__ == "__main__":
    main()