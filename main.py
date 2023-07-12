import cv2, numpy as np, utilities, os, tkinter as tk, pickle, tensorflow as tf, time, shutil
from tkinter import messagebox, simpledialog
from scipy.spatial.distance import cosine

class HoverButton(tk.Button):
    def __init__(self, master, **kw):
        tk.Button.__init__(self,master=master,**kw)
        self.defaultBackground = self["background"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self['background'] = self['activebackground']

    def on_leave(self, e):
        self['background'] = self.defaultBackground


class Application(object):
    def __init__(self, database_path = os.path.join('.', 'Database'), apply_clahe = True, threshold = 0.965):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self.main_window_name = 'Output'
        self.add_subject_window_name = 'Adding a new subject'
        self.database_path = database_path
        self.font = ('Consolas', 18)
        self.width, self.height = 800, 350
        self.window = tk.Tk()
        self.window.config(background = 'teal')
        self.window.title('Face recognition application')
        self.window.maxsize(self.width, self.height)
        self.window.minsize(self.width, self.height)
        self.frames_count = 50
        self.threshold = threshold
        self.apply_clahe = apply_clahe
        if self.apply_clahe:
            self.clahe = cv2.createCLAHE(5, (5, 5))
        self.hover_button_background = 'lightgreen'
        

    def with_classifier(self):
        if os.path.isfile('models and data/svm classifier/svm_model.joblib'):
            with open('models and data/svm classifier/svm_model.joblib', 'rb') as file:
                classifier, subjects = pickle.load(file)
        else:
            messagebox.showerror('No model exists!', "A model doesn't exist.\nConsider training one by pressing the 'train' button.")                
            return
        
        feature_extractor = tf.lite.Interpreter(model_path = './models and data/facenet optimized/model.tflite', num_threads = 8)
        feature_extractor.allocate_tensors()
        feature_extractor_input_details = feature_extractor.get_input_details()
        feature_extractor_output_details = feature_extractor.get_output_details()
        
        cv2.namedWindow(self.main_window_name)
        while self.cap.isOpened() and cv2.getWindowProperty(self.main_window_name, cv2.WND_PROP_VISIBLE) == 1:
            t1 = time.perf_counter()
            ret, frame = self.cap.read()
            if self.apply_clahe:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                frame[:, :, 0] = self.clahe.apply(frame[:, :, 0])
                frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
                
            dets = utilities.detect_faces_mp(frame)
            faces = utilities.preprocess_faces(frame, dets)
            
            for det, face in zip(dets, faces):
                face = np.expand_dims(face, axis = 0)
                feature_extractor.set_tensor(feature_extractor_input_details[0]['index'], face.astype(np.float32))
                feature_extractor.invoke()
                feature = feature_extractor.get_tensor(feature_extractor_output_details[0]['index'])
                prediction = classifier.predict_proba(feature)[0]
                p = np.argmax(prediction)
                prediction = prediction[p]
                if prediction <= self.threshold:
                    name = 'Unknown'
                else:
                    name = subjects[p]
                    
                x, y, w, h = det
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.putText(frame, name, (x, y-20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
            t2 = time.perf_counter()
            fps = round(1/(t2 - t1), 1)
            cv2.putText(frame, f'FPS: {fps}', (0, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 4)
            cv2.putText(frame, f'FPS: {fps}', (0, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 165, 255), 1)
            cv2.imshow(self.main_window_name, frame)
            key = cv2.waitKey(1)
        cv2.destroyAllWindows()


    def without_classifier(self):
        if len(os.listdir('Database')) <= 0:
            messagebox.showerror('No subjects are in the database!', 'The database is empty.\nConsider adding subjects first!')
            return
        
        with open('models and data/Embedding averages/averages.joblib', 'rb') as file:
            embeddings, subjects = pickle.load(file)
        
        feature_extractor = tf.lite.Interpreter(model_path = './models and data/facenet optimized/model.tflite', num_threads = 8)
        feature_extractor.allocate_tensors()
        feature_extractor_input_details = feature_extractor.get_input_details()
        feature_extractor_output_details = feature_extractor.get_output_details()

        cv2.namedWindow(self.main_window_name)
        while self.cap.isOpened() and cv2.getWindowProperty(self.main_window_name, cv2.WND_PROP_VISIBLE) == 1:
            t1 = time.perf_counter()
            ret, frame = self.cap.read()
            if self.apply_clahe:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                frame[:, :, 0] = self.clahe.apply(frame[:, :, 0])
                frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
                
            dets = utilities.detect_faces_mp(frame)
            faces = utilities.preprocess_faces(frame, dets)
            
            for det, face in zip(dets, faces):
                face = np.expand_dims(face, axis = 0)
                feature_extractor.set_tensor(feature_extractor_input_details[0]['index'], face.astype(np.float32))
                feature_extractor.invoke()
                feature = feature_extractor.get_tensor(feature_extractor_output_details[0]['index'])[0]
                distances = []
                for embedding in embeddings:
                    distances.append(cosine(embedding, feature))

                distances = np.array(distances)
                minimum_distance = np.argmin(distances)
                if distances[minimum_distance] < 0.2:
                    name = subjects[minimum_distance]
                else:
                    name = 'Unknown'

                x, y, w, h = det
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.putText(frame, name, (x, y-20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
            
            t2 = time.perf_counter()
            fps = round(1/(t2 - t1), 1)
            cv2.putText(frame, f'FPS: {fps}', (0, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 4)
            cv2.putText(frame, f'FPS: {fps}', (0, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 165, 255), 1)
            cv2.imshow(self.main_window_name, frame)
            key = cv2.waitKey(1)



    def add_subject(self):            
        subject_name = simpledialog.askstring('Subject name', 'Please enter the name of the subject you wish to add')
        if subject_name == None:
            messagebox.showerror('Invalid name!', 'Invalid subject name!')
            return
        
        output_dir = os.path.join(self.database_path, subject_name)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
            required_samples = self.frames_count
        else:
            messagebox.showinfo('Already exists', 'The subject already exists in the database. New images will be added!')
            required_samples = simpledialog.askinteger('Number of samples', 'How many new samples do you want?')
            if required_samples is None or required_samples <= 0:
                messagebox.showerror('Invalid number of new samples', 'You should enter a proper number of samples')
            elif required_samples > self.frames_count:
                messagebox.showinfo('Too many samples!', f'The number of samples that will be recorded is {self.frames_count}!')
                required_samples = self.frames_count
                
        cv2.namedWindow(self.add_subject_window_name)
        i = 0
        while self.cap.isOpened() and cv2.getWindowProperty(self.add_subject_window_name, cv2.WND_PROP_VISIBLE) == 1 and i < required_samples:
            ret, frame = self.cap.read()
            if self.apply_clahe:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                frame[:, :, 0] = self.clahe.apply(frame[:, :, 0])
                frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
            dets = utilities.detect_faces_mp(frame)
            if len(dets) > 1:
                messagebox.showinfo('Too many subjects!', 'Only a single subject must be in the frame!')
                continue
            elif len(dets) == 1:
                x, y, w, h = dets[0]
                face = frame[y : y+h, x : x+w]
                cv2.imwrite(os.path.join(self.database_path, subject_name, f'{time.time()}.jpg'), face)
                i += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            
            cv2.imshow(self.add_subject_window_name, frame)
            key = cv2.waitKey(100)
        cv2.destroyAllWindows()

        embedding = utilities.extract_embedding_average(subject_name)
        averages_path = 'models and data/Embedding averages/averages.joblib'
        if not os.path.isfile(averages_path):
            with open('models and data/Embedding averages/averages.joblib', 'x') as file:
                embeddings, subjects = [], []
        else:
            with open(averages_path, 'rb') as file:
                embeddings, subjects = pickle.load(file)
        
        if subject_name in subjects:
            i = subjects.index(subject_name)
            embeddings[i] = embedding
        else:
            embeddings.append(embedding)
            subjects.append(subject_name)

        with open(averages_path, 'wb') as file:
            pickle.dump([embeddings, subjects], file)


    def train(self):
        if len(os.listdir('Database')) <= 0:
            messagebox.showerror('No subjects are in the database!', 'The database is empty.\nConsider adding subjects first!')
            return
        
        messagebox.showinfo('Training in progress...', 'The training has started.\nClose this message and check the terminal as the training is in progress!')
        output = utilities.train_svm()
        
        if output[1] == -1:
            messagebox.showerror('Error', f'Subject {output[0]} has less than 10 images!')
            return
        
        output_path = 'models and data/svm classifier'
        messagebox.showinfo('Done training!', f'The F1 score of the final model is: {output[1]}')
        with open(os.path.join(output_path, 'svm_model.joblib'), 'wb') as file:
            pickle.dump((output[0], output[2]), file)

    
    def delete_database(self):
        delete_db = messagebox.askyesno('Data will be lost!', 'Are your sure you want to delete the database?\nYour data will be lost!')
        if not delete_db:
            return
        
        subjects = os.listdir('Database')
        if len(subjects) <= 0:
            return
        
        backup_directory_name = str(time.time())
        os.mkdir(f'old Database versions/{backup_directory_name}')
        backup_directory_path = os.path.join('old Database versions', backup_directory_name)
        for dir in subjects:
            shutil.move(os.path.join('Database', dir), backup_directory_path)
        
        averages_path = 'models and data/Embedding averages/averages.joblib'
        if os.path.isfile(averages_path):
            shutil.move(averages_path, os.path.join('old Database versions', backup_directory_name, 'averages.joblib'))

        svm_path = 'models and data/svm classifier/svm_model.joblib'
        if os.path.isfile(svm_path):
            shutil.move(svm_path, os.path.join('old Database versions', backup_directory_name, 'svm_model.joblib'))


    def run(self):
        new_subject_btn = HoverButton(master = self.window, text = 'Add new subject', command = self.add_subject, font = self.font, activebackground = self.hover_button_background)
        start_classifier_btn = HoverButton(master = self.window, text = 'Start with classifier', command = self.with_classifier, font = self.font, activebackground = self.hover_button_background)
        start_without_classifier_btn = HoverButton(master = self.window, text = 'Start without classifier', command = self.without_classifier, font = self.font, activebackground = self.hover_button_background)
        train_btn = HoverButton(master = self.window, text = 'Train the SVM', command = self.train, font = self.font, activebackground = self.hover_button_background)
        delete_database_btn = HoverButton(master = self.window, text = 'Delete the database', command = self.delete_database, font = self.font, background = '#ff9999', activebackground = 'red')

        new_subject_btn.place(x = (self.width/4 - new_subject_btn.winfo_reqwidth()/2), y = 50)
        train_btn.place(x = self.width * 3/4 - train_btn.winfo_reqwidth()/2, y = 50)
        start_classifier_btn.place(x = (self.width/4 - start_classifier_btn.winfo_reqwidth()/2), y = 150)
        start_without_classifier_btn.place(x = self.width*3/4 - start_without_classifier_btn.winfo_reqwidth()/2, y = 150)
        delete_database_btn.place(x = self.width/2 - delete_database_btn.winfo_reqwidth()/2, y = 250)
        self.window.mainloop()

app = Application()
app.run()  