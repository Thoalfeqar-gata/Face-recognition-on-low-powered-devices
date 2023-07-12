import mediapipe as mp, cv2, numpy as np, os, dlib, tensorflow as tf, pickle
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

shape_predictor = dlib.shape_predictor('models and data/dlib/shape_predictor_5_face_landmarks.dat')
def preprocess_faces(image, detections, size = 160, normalize = True):
    global shape_predictor
    
    if len(detections) <= 0:
        detections = [(0, 0, image.shape[1], image.shape[0])]

    dets = []
    for x, y, w, h in detections:
        dets.append(dlib.rectangle(left = x, top = y, right = x+w, bottom = y+h))
    
    faces = dlib.full_object_detections()
    for det in dets:
        faces.append(shape_predictor(image, det))
    
    if normalize:
        faces = [face[:, :, ::-1] / 255.0 for face in dlib.get_face_chips(image, faces, size = size, padding = 0.075)]
    else:
        faces = [face[:, :, ::-1] for face in dlib.get_face_chips(image, faces, size = size, padding = 0.075)]
    return faces


face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection = 0,
    min_detection_confidence = 0.5
)
def detect_faces_mp(image):
    h, w = image.shape[:2]
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.detections:
        return []
    dets = []
    for result in results.detections:
        if result.score[0] < 0.75:
            continue
        det = result.location_data.relative_bounding_box
        x, y, width, height = det.xmin, det.ymin, det.width, det.height
        x = int(x * w)
        y = int(y * h)
        width = int(width * w)
        height = int(height * h)
        dets.append([x, y, width, height])
    
    return dets

def train_svm():
    feature_extractor = tf.lite.Interpreter(model_path = './models and data/facenet optimized/model.tflite', num_threads = 8)
    feature_extractor.allocate_tensors()
    feature_extractor_input_details = feature_extractor.get_input_details()
    feature_extractor_output_details = feature_extractor.get_output_details()
    
    training_data = []
    training_labels = []
    subjects = os.listdir('Database')
    for i in tqdm(range(len(subjects))):
        subject = subjects[i]
        files = os.listdir(os.path.join('Database', subject))
        if len(files) < 10:
            return (subject, -1)
        
        for file in files:
            image = cv2.imread(os.path.join('Database', subject, file))
            image = preprocess_faces(image, [])[0]
            image = np.expand_dims(image, axis = 0)
            feature_extractor.set_tensor(feature_extractor_input_details[0]['index'], image.astype(np.float32))
            feature_extractor.invoke()
            feature = feature_extractor.get_tensor(feature_extractor_output_details[0]['index'])[0]
            training_data.append(feature)
            training_labels.append(i)
    
    training_data = np.array(training_data)
    training_labels = np.array(training_labels)
    X_train, X_test, y_train, y_test = train_test_split(training_data, training_labels, test_size = 0.1, train_size = 0.9, random_state = 200)        

    model = OneVsRestClassifier(SVC(kernel = 'rbf', max_iter = -1, probability = True, verbose = True), n_jobs = -1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average = 'weighted')
    return (model, score, subjects)

def extract_embedding_average(subject):
    feature_extractor = tf.lite.Interpreter(model_path = './models and data/facenet optimized/model.tflite', num_threads = 8)
    feature_extractor.allocate_tensors()
    feature_extractor_input_details = feature_extractor.get_input_details()
    feature_extractor_output_details = feature_extractor.get_output_details()
    
    features = np.zeros((128,), dtype = np.float32)
    for i, file in enumerate(os.listdir(os.path.join('Database', subject))):
        image = cv2.imread(os.path.join('Database', subject, file))
        image = preprocess_faces(image, [])[0]
        image = np.expand_dims(image, axis = 0)
        feature_extractor.set_tensor(feature_extractor_input_details[0]['index'], image.astype(np.float32))
        feature_extractor.invoke()
        feature = feature_extractor.get_tensor(feature_extractor_output_details[0]['index'])[0]
        features += feature
    features = features / i
    return features
