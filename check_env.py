import sys

import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
import matplotlib
import sklearn
import PIL
import ultralytics
import onnx
import onnxruntime as ort

print("=== Environment Check ===")
print("Python      :", sys.version.split()[0])
print("Executable  :", sys.executable)
print("PyTorch     :", torch.__version__)
print("Torchvision :", torchvision.__version__)
print("OpenCV      :", cv2.__version__)
print("NumPy       :", np.__version__)
print("Pandas      :", pd.__version__)
print("Matplotlib  :", matplotlib.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Pillow      :", PIL.__version__)
print("Ultralytics :", ultralytics.__version__)
print("ONNX        :", onnx.__version__)
print("ONNX Runtime:", ort.__version__)
print("CUDA usable :", torch.cuda.is_available())

print("\nEnvironment check completed successfully.")