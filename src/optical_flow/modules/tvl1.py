import cv2
import numpy as np
from src.image.modules import Image


class TVL1:

    # Image instance untuk pemrosesan gambar
    image: Image = None

    def __init__(self, fast_mode: bool = True):
        
        self.image = Image()
        
        self.flow = None
        
        self.use_cuda = False

        # Deteksi cuda saat pertama kali inisialisasi untuk menentukan apakah akan menggunakan GPU atau CPU
        # Jika tidak ada dukungan CUDA, akan menggunakan CPU dengan OpenCV OptFlow TV-L1
        # Jika ada dukungan CUDA, akan menggunakan OpenCV CUDA TV-L1 untuk performa yang lebih baik
        has_cuda = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        has_cuda_tvl1 = has_cuda and hasattr(cv2.cuda, "OpticalFlowDual_TVL1_create")

        if has_cuda_tvl1:
            self.use_cuda = True
            self.tvl1 = cv2.cuda.OpticalFlowDual_TVL1_create()
            self.gpumat_prev = cv2.cuda_GpuMat()
            self.gpumat_next = cv2.cuda_GpuMat()
            self.gpumat_flow = cv2.cuda_GpuMat()
        else:
            # Fallback aman ke CPU
            self.tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

        # Parameter dasar untuk TV-L1
        # Ini adalah nilai default yang dapat disesuaikan sesuai kebutuhan
        self.tvl1.setLambda(0.15)
        self.tvl1.setTheta(0.3)
        self.tvl1.setTau(0.25)

        # Penyesuaian parameter untuk improve performa dengan mengorbankan akurasi
        # Jika fast_mode diaktifkan, kita akan mengurangi jumlah iterasi dan skala
        # Ini akan sedikit mempercepat komputasi tetapi mungkin mengurangi akurasi optical flow
        if fast_mode:
            if hasattr(self.tvl1, "setScalesNumber"):
                self.tvl1.setScalesNumber(3)
            if hasattr(self.tvl1, "setWarpingsNumber"):
                self.tvl1.setWarpingsNumber(2)
            if hasattr(self.tvl1, "setInnerIterations"):
                self.tvl1.setInnerIterations(20)
            if hasattr(self.tvl1, "setOuterIterations"):
                self.tvl1.setOuterIterations(5)
            if hasattr(self.tvl1, "setMedianFiltering"):
                self.tvl1.setMedianFiltering(1)


    def compute(self, prev: np.ndarray, next: np.ndarray, download: bool = True):
        """
        Menghitung optical flow antara dua frame menggunakan algoritma TV-L1.

        Args:
            prev (numpy.ndarray): Frame sebelum.
            next (numpy.ndarray): Frame saat ini.
            download (bool): Jika True hasil di-download ke CPU (numpy array).
                             Jika False (GPU mode), kembalikan GpuMat.

        Returns:
            numpy.ndarray | cv2.cuda_GpuMat: Optical flow yang dihitung.
        """
        if prev is None or next is None:
            raise ValueError("Input frame is None.")
        if prev.shape != next.shape:
            raise ValueError("Input frames must have the same dimensions.")

        # Jika input bukan grayscale, konversi ke grayscale
        # Memastikan input adalah 2D (grayscale)
        gray_prev = prev if prev.ndim == 2 else self.image.grayscale(prev)
        gray_next = next if next.ndim == 2 else self.image.grayscale(next)

        # Pastikan tipe data adalah uint8 untuk kompatibilitas dengan OpenCV
        if gray_prev.dtype != np.uint8:
            gray_prev = gray_prev.astype(np.uint8, copy=False)
        if gray_next.dtype != np.uint8:
            gray_next = gray_next.astype(np.uint8, copy=False)

        # Pastikan array adalah contiguous untuk performa yang lebih baik
        # OpenCV membutuhkan array yang contiguous untuk operasi GPU
        gray_prev = np.ascontiguousarray(gray_prev)
        gray_next = np.ascontiguousarray(gray_next)

        if self.use_cuda:
            self.gpumat_prev.upload(gray_prev)
            self.gpumat_next.upload(gray_next)

            # Reuse output buffer, tidak perlu release tiap frame
            self.gpumat_flow = self.tvl1.calc(self.gpumat_prev, self.gpumat_next, self.gpumat_flow)
            flow = self.gpumat_flow.download() if download else self.gpumat_flow
        else:
            flow = self.tvl1.calc(gray_prev, gray_next, None)

        self.flow = flow
        return flow


    def close(self):
        if self.use_cuda:
            self.gpumat_prev.release()
            self.gpumat_next.release()
            self.gpumat_flow.release()