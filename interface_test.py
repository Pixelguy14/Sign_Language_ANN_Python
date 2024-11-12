import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as rt
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture

# Inicializar MediaPipe Hands y el modelo ONNX
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
sess = rt.InferenceSession('trained_model.onnx')

# Mapeo de las etiquetas a las letras
labels = {
    0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g",
    7: "h", 8: "i", 9: "k", 10: "l", 11: "m", 12: "n", 13: "o",
    14: "p", 15: "q", 16: "r", 17: "s", 18: "t", 19: "u", 20: "v",
    21: "w", 22: "x", 23: "y"
}

# Preprocesar los landmarks para hacerlos compatibles con el modelo
def preprocess_landmarks(hand_landmarks):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    processed_landmarks = np.array(landmarks).reshape(1, -1).astype(np.float32)
    return processed_landmarks

# Función para predecir el gesto con el modelo
def predict_gesture(landmarks):
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred = sess.run([output_name], {input_name: landmarks})[0]

    # Si el modelo devuelve una probabilidad, usamos np.argmax para obtener el índice de la clase con mayor probabilidad
    predicted_label_index = np.argmax(pred)

    # Asegurarse de que la etiqueta esté dentro del rango de clases posibles
    predicted_label = labels.get(predicted_label_index, "Unknown")
    return predicted_label

class SignLanguageApp(App):
    def build(self):
        # Layout principal de la interfaz
        main_layout = BoxLayout(orientation="vertical", padding=10, spacing=10)
        
        # Barra de botones de modo
        mode_layout = BoxLayout(size_hint_y=0.1)
        mode_image_button = Button(text="Modo imagen")
        mode_real_time_button = Button(text="Modo tiempo real")
        mode_layout.add_widget(mode_image_button)
        mode_layout.add_widget(mode_real_time_button)
        main_layout.add_widget(mode_layout)

        # Layout de imágenes con la separación
        image_layout = BoxLayout(size_hint_y=0.7, spacing=20)
        self.input_image = Image(source="path/to/input_image.png")  # Imagen estática como ejemplo
        image_layout.add_widget(self.input_image)

        # Widget para mostrar la cámara en tiempo real
        self.camera_image = Image()
        image_layout.add_widget(self.camera_image)
        main_layout.add_widget(image_layout)

        # Layout de navegación (para control de interfaz)
        navigation_layout = BoxLayout(size_hint_y=0.1)
        prev_button = Button(text="←")
        next_button = Button(text="→")
        navigation_layout.add_widget(prev_button)
        navigation_layout.add_widget(next_button)
        main_layout.add_widget(navigation_layout)

        # Label para mostrar la letra reconocida
        self.letter_label = Label(text="Letra reconocida: ")
        main_layout.add_widget(self.letter_label)

        # Inicializa la cámara y MediaPipe Hands para detección de manos
        self.capture = cv2.VideoCapture(0)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        Clock.schedule_interval(self.update_camera, 1.0 / 30.0)  # Actualiza 30 veces por segundo

        return main_layout

    def update_camera(self, dt):
        # Captura el frame de la cámara y lo procesa
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, -1)  # Modo espejo para facilitar el gesto
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesa el frame con MediaPipe Hands
            results = self.hands.process(rgb_frame)
            gesture = "Desconocido"

            # Si se detectan landmarks en la mano, se predice el gesto
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Preprocesa los landmarks y predice el gesto
                    landmarks = preprocess_landmarks(hand_landmarks)
                    gesture = predict_gesture(landmarks)

            # Muestra la letra reconocida en el Label de Kivy
            self.letter_label.text = f"Letra reconocida: {gesture}"

            # Convierte el frame a una textura para mostrarlo en Kivy
            buf = frame.tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.camera_image.texture = texture

    def on_stop(self):
        # Libera la cámara y los recursos de MediaPipe cuando la aplicación se cierre
        if self.capture:
            self.capture.release()
        if self.hands:
            self.hands.close()

# Ejecuta la aplicación
if __name__ == '__main__':
    SignLanguageApp().run()
