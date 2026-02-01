import os
import cv2
import numpy as np
import absl.logging
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Tuple, List


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

    FACE_OVAL: List[int] = [
        10, 338, 297, 332, 284, 251, 389, 356,
        454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109
    ]

    STABLE_POINTS: List[int] = [
        1,    # nose tip
        33,   # left eye outer
        263,  # right eye outer
        61,   # mouth left
        291   # mouth right
    ]


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


    def crop(self,
             image: np.ndarray,
             landmarks: vision.FaceLandmarkerResult = None,  # type: ignore
             landmark_indices: List[int] = None,
             margin: float = 0.05,
             output_size: Tuple[int, int] = (240, 240)) -> np.ndarray:
        """
        Melakukan cropping wajah berdasarkan landmark yang terdeteksi

        Args:
            image: Citra input dalam format numpy ndarray
            landmarks: Hasil deteksi landmark (opsional)
            landmark_indices: Indeks landmark yang akan digunakan untuk cropping (opsional)
            margin: Margin tambahan untuk bounding box (default: 0.05)
            output_size: Ukuran output citra setelah cropping dan resizing (default: (240, 240))

        Returns:
            np.ndarray: Citra wajah yang telah di-crop dan di-resize

        Raises:
            ValueError: Jika landmark tidak tersedia atau tidak ada landmark yang terdeteksi
        """

        # Validasi landmark yang diberikan telah sesuai atau tidak
        # Jika tidak ada landmark yang diberikan dan juga tidak ada landmark yang tersimpan
        if landmarks is None and self.landmark is None:
            raise ValueError("Landmark detection has not been performed.")

        # Gunakan landmark yang diberikan atau yang tersimpan
        final_landmarks = landmarks if landmarks is not None else self.landmark

        # Jika tidak ada landmark yang terdeteksi
        # Maka raise dengan ValueError yang menunjukkan tidak ada landmark yang terdeteksi
        if not final_landmarks.face_landmarks:
            raise ValueError("No face landmarks detected.")

        # Gunakan indeks landmark default jika tidak diberikan
        if landmark_indices is None:
            landmark_indices = self.FACE_OVAL

        # Gabungkan dengan titik stabil untuk memastikan cropping yang konsisten
        effective_indices = landmark_indices + self.STABLE_POINTS

        h, w, _ = image.shape
        face_landmarks = final_landmarks.face_landmarks[0]

        # Ambil koordinat landmark subset yang diinginkan
        xs = [face_landmarks[i].x * w for i in effective_indices]
        ys = [face_landmarks[i].y * h for i in effective_indices]

        # Menghitung bounding box dari landmark yang diberikan
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        # Menghitung margin tambahan untuk bounding box
        dx = int((x_max - x_min) * margin)
        dy = int((y_max - y_min) * margin)

        x_min = max(0, x_min - dx)
        x_max = min(w, x_max + dx)
        y_min = max(0, y_min - dy)
        y_max = min(h, y_max + dy)

        # Memotong citra berdasarkan bounding box yang dihitung
        face_crop = image[y_min:y_max, x_min:x_max]
        face_crop = cv2.resize(face_crop, output_size)

        return face_crop
