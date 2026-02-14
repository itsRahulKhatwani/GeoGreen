import torch
print(f"Torch imported: {torch.__version__}")
try:
    import easyocr
    print("EasyOCR imported successfully")
except Exception as e:
    print(f"EasyOCR import failed: {e}")
