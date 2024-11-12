import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget

class SignLanguageApp(App):
    def build(self):
        # Layout principal
        main_layout = BoxLayout(orientation="vertical", padding=10, spacing=10)
        
        # Barra de botones de modo
        mode_layout = BoxLayout(size_hint_y=0.1)
        mode_image_button = Button(text="Modo imagen")
        mode_real_time_button = Button(text="Modo tiempo real")
        mode_layout.add_widget(mode_image_button)
        mode_layout.add_widget(mode_real_time_button)
        main_layout.add_widget(mode_layout)

        # Layout de imágenes con separación
        image_layout = BoxLayout(size_hint_y=0.7, spacing=20)
        
        # Imagen de entrada (imagen estática)
        self.input_image = Image(source="path/to/input_image.png")
        image_layout.add_widget(self.input_image)

        # Widget de la cámara en tiempo real
        self.camera_image = Image()
        image_layout.add_widget(self.camera_image)
        
        main_layout.add_widget(image_layout)

        # Layout de botones de navegación
        navigation_layout = BoxLayout(size_hint_y=0.1)
        prev_button = Button(text="<-")
        next_button = Button(text="->")
        navigation_layout.add_widget(prev_button)
        navigation_layout.add_widget(next_button)
        main_layout.add_widget(navigation_layout)

        # Label para la letra reconocida
        letter_label = Label(text="Letra reconocida")
        main_layout.add_widget(letter_label)

        # Inicializa la cámara
        self.capture = cv2.VideoCapture(0)  # 0 para la cámara por defecto
        Clock.schedule_interval(self.update_camera, 1.0 / 30.0)  # Actualiza 30 veces por segundo

        return main_layout

    def update_camera(self, dt):
        # Actualiza la imagen de la cámara en tiempo real
        ret, frame = self.capture.read()
        if ret:

            # Voltea la imagen horizontalmente para que no esté al revés
            frame = cv2.flip(frame, -1)
            # Convierte la imagen de BGR a RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convierte el frame a una textura de Kivy
            buf = frame.tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.camera_image.texture = texture

    def on_stop(self):
        # Libera la cámara al cerrar la aplicación
        if self.capture:
            self.capture.release()

# Ejecuta la aplicación
if __name__ == '__main__':
    SignLanguageApp().run()
