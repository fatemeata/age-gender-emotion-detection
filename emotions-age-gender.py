import cv2
import numpy as np
import os
from keras.models import load_model
from keras.utils.data_utils import get_file
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from utils.wide_resnet import WideResNet


class AGEDetection:
    def __init__(self):
        self.emotion_model_path = './models/emotion_model.hdf5'
        self.emotion_labels = get_labels('fer2013')
        self.emotion_labels_list = []
        for g in range(7):
            self.emotion_labels_list.append(self.emotion_labels[g])
        self.face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
        self.WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"
        self.emotion_classifier = load_model(self.emotion_model_path)

    def get_emotion_classifier(self):
        return self.emotion_classifier

    def main(self):

        USE_WEBCAM = True # If false, loads video file source
        # parameters for loading data and images

        # hyper-parameters for bounding boxes shape
        frame_window = 10
        emotion_offsets = (20, 40)

        # Age- gender model
        depth = 16
        width = 8
        face_size = 64
        age_gender_model = WideResNet(face_size, depth=depth, k=width)()
        age_gender_model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5', self.WRN_WEIGHTS_PATH, cache_subdir=age_gender_model_dir)
        age_gender_model.load_weights(fpath)

        # getting input model shapes for inference
        emotion_target_size = self.emotion_classifier.input_shape[1:3]

        # starting lists for calculating modes
        emotion_window = []
        WINDOW_NAME = "Age, Gender and Emotion detection"
        # Select video or webcam feed
        cap = None
        if (USE_WEBCAM == True):
            cap = cv2.VideoCapture(0)  # Webcam source
        else:
            cap = cv2.VideoCapture('./demo/dinner.mp4')  # Video file source

        while cap.isOpened():
            ret, bgr_image = cap.read()

            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            faces = self.face_cascade.detectMultiScale(
                gray_image,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE)

            face_imgs = np.empty((len(faces), face_size, face_size, 3))

            for i, face_coordinates in enumerate(faces):
                face_img, cropped = self.crop_face(bgr_image, face_coordinates, margin=40, size=face_size)
                (x, y, w, h) = cropped
                cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (255, 200, 0), 2)
                face_imgs[i, :, :, :] = face_img

                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]
                rgb_face = rgb_image[y1:y2, x1:x2]
                try:
                    gray_face = cv2.resize(gray_face, (emotion_target_size))
                    rgb_face = cv2.resize(rgb_face, (emotion_target_size))
                except:
                    continue

                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_prediction = self.emotion_classifier.predict(gray_face)
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = self.emotion_labels[emotion_label_arg]
                emotion_window.append(emotion_text)

                emotion_list = []
                for index in range(7):
                    # print(self.emotion_labels[index], ":", emotion_prediction[0][index])
                    emotion_list.append(emotion_prediction[0][index])

                print("Emotion List: ", self.emotion_labels_list)
                print("Emotion List Values: ", emotion_list)
                results = age_gender_model.predict(face_imgs)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()

                if emotion_text == 'angry':
                    color = emotion_probability * np.asarray((255, 0, 0))
                elif emotion_text == 'sad':
                    color = emotion_probability * np.asarray((0, 0, 255))
                elif emotion_text == 'happy':
                    color = emotion_probability * np.asarray((255, 255, 0))
                elif emotion_text == 'surprise':
                    color = emotion_probability * np.asarray((0, 255, 255))
                elif emotion_text == 'fear':
                    color = emotion_probability * np.asarray((63, 6, 49))
                elif emotion_text == 'disgust':
                    color = emotion_probability * np.asarray((3, 46, 4))
                else:  # Neutral
                    color = emotion_probability * np.asarray((0, 255, 0))

                color = color.astype(int)
                color = color.tolist()

                gender = "Female" if predicted_genders[i][0] > 0.5 else "Male"
                age = int(predicted_ages[i])
                emotion = emotion_text

                label = " {}: {}\n {}: {}\n {}: {}".format("Age", age, "Gender", gender, "Emotion", emotion)
                # print(label)

                draw_bounding_box(face_coordinates, rgb_image, color)
                draw_text(face_coordinates, rgb_image, label, color, 0, -45, 1, 1)

            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Emotion, Age and Gender Detection', bgr_image)
            cv2.setWindowProperty("Emotion, Age and Gender Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def crop_face(self, imgarray, section, margin=40, size=64):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

if __name__ == "__main__":
    detector = AGEDetection()
    detector.main()
