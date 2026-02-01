import os
import cv2
import numpy as np
import absl.logging
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Mencoba untuk supress (menyembunyikan) log warning dari TensorFlow mediapipe
# Harapan: untuk membuat output lebih bersih tanpa log yang tidak berpengaruh
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
absl.logging.set_verbosity(absl.logging.ERROR)


class FaceLandmark:

    BASE_ROOT_DIR: str = Path(os.getcwd()).parent

    BASE_MODEL_PATH: str = "src/face/tasks/face_landmarker.task"

    MODEL_PATH: Path = Path(BASE_ROOT_DIR, BASE_MODEL_PATH)

    base_options: python.BaseOptions #type: ignore

    options: vision.FaceLandmarkerOptions #type: ignore

    landmarker: vision.FaceLandmarker #type: ignore

    landmark: vision.FaceLandmarkerResult | None #type: ignore

    def __init__(self):

        # Jika file model tidak ditemukan, raise dengan sebuah error
        # untuk menghindari kesalahan saat inisialisasi model FaceLandmark
        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {self.MODEL_PATH}")

        # Inisialisasi base options dengan menyertakan path model yang benar
        self.base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH.as_posix())

        # Inisialisasi options untuk FaceLandmarker
        # Beberapa parameter ditambahkan untuk konfigurasi yang lebih baik, seperti:
        # num_faces -> untuk menentukan jumlah wajah yang akan dideteksi
        # min_tracking_confidence -> untuk mengatur ambang kepercayaan pelacakan wajah
        # min_face_detection_confidence -> untuk mengatur ambang kepercayaan deteksi wajah
        # min_face_presence_confidence -> untuk mengatur ambang kepercayaan keberadaan wajah
        # running_mode -> untuk menentukan mode operasi (IMAGE, VIDEO, LIVE_STREAM)
        self.options = vision.FaceLandmarkerOptions(base_options=self.base_options,
                                                    num_faces=1,
                                                    min_tracking_confidence=0.7,
                                                    min_face_detection_confidence=0.7,
                                                    min_face_presence_confidence=0.7,
                                                    running_mode=vision.RunningMode.IMAGE)
        
        # Membuat instance Face Landmarker dengan opsi yang telah ditentukan
        self.landmarker = vision.FaceLandmarker.create_from_options(self.options)

        # Membuat attribut landmark untuk menyimpan hasil deteksi
        self.landmark = None


    def detect(self, image: np.ndarray) -> vision.FaceLandmarkerResult: #type: ignore
        """
        Melakukan deteksi landmarks dengan citra input

        Args:
            image: Citra input dalam format numpy ndarray

        Returns:
            FaceLandmarkerResult: Hasil deteksi dari FaceLandmarker

        Raises:
            ValueError: Jika citra input tidak valid
        """

        # Jika citra bukan merupakan instance dari numpy ndarray
        # Maka raise dengan ValueError yang menunjukkan image harus berupa numpy ndarray
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy ndarray.")

        # Jika citra tidak memiliki value apapun meskipun instance dari numpy ndarray
        # Maka raise dengan ValueError yang menunjukkan image tidak boleh kosong (invalid)
        if image.size == 0:
            raise ValueError("Input image is empty.")

        # Memastikan citra dalam format RGB sebelum diproses
        # Hal ini diperlukan oleh MediaPipe Face Landmarker
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Melakukan deteksi landmarks pada citra RGB
        self.landmark = self.landmarker.detect(mp_image)
        return self.landmark


    def crop(self, image: np.ndarray, margin: float = 0.2) -> np.ndarray | None: #type: ignore
        """
        Melakukan cropping wajah berdasarkan landmark yang terdeteksi
        
        Args:
            landmark: Hasil deteksi dari FaceLandmarker

        Returns:
            np.ndarray | None: Citra wajah yang telah di-crop atau None jika tidak ada wajah terdeteksi
        """

        # Memastikan bahwa telah memiliki titik landmark sebelum melakukan cropping
        # Jika belum melakukan deteksi landmark, maka raise dengan ValueError
        if self.landmark is None:
            raise ValueError("Landmark detection has not been performed yet due to missing landmark data.")

        # Memastikan bahwa ada wajah yang terdeteksi sebelum melakukan cropping
        # Jika tidak ada wajah yang terdeteksi, maka raise dengan ValueError
        if not self.landmark.face_landmarks:
            raise ValueError("No face landmarks detected to perform cropping.")

        # Melakukan crop wajah berdasarkan landmark yang terdeteksi
        h, w, _ = image.shape

        landmarks = self.landmark.face_landmarks[0]

        # Mendapatkan koordinat x dan y dari semua landmark
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]

        # Menghitung bounding box dari koordinat landmark
        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

        # Menambahkan margin pada bounding box untuk mengakomodasi area sekitar wajah
        x_margin = int((x_max - x_min) * margin)
        y_margin = int((y_max - y_min) * margin)

        # Memastikan bounding box tetap berada dalam batas citra yang asli
        x_min = max(0, x_min - x_margin)
        x_max = min(w, x_max + x_margin)
        y_min = max(0, y_min - y_margin)
        y_max = min(h, y_max + y_margin)

        return image[y_min:y_max, x_min:x_max]
        