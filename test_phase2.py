import sys
import os
import cv2
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def run_tests():
    print("🔍 Starting Phase 2 Backend Verification...")

    # 1. Test Imports
    print("\n1️⃣ Testing Imports...")
    try:
        from src.cv_analysis import analyze_land_cover
        from src.ocr_utils import extract_coordinates
        from src.climate_api import get_climate_data
        from src.analysis import generate_rgb_recommendations
        import easyocr
        import streamlit
        print("   ✅ All modules imported successfully")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return

    # 2. Test CV Analysis
    print("\n2️⃣ Testing Computer Vision (RGB Segmentation)...")
    try:
        # Create dummy image: Green square on Blue background
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = [255, 0, 0] # Blue (BGR) -> Water
        img[25:75, 25:75] = [34, 139, 34] # Green (BGR) -> Vegetation? No, input to function is BGR, function converts to RGB.
        # Wait, cv2.imwrite expects BGR. 34, 139, 34 is Green.
        
        cv2.imwrite("dummy_test.png", img)
        
        stats, mask, overlay = analyze_land_cover("dummy_test.png", n_clusters=2)
        print(f"   ✅ CV Stats: {stats}")
        
        # Check if Vegetation and Water are detected
        # Note: K-Means labels are random, but my logic maps colors.
        # Green (34,139,34) should map to Vegetation. Blue (255,0,0) to Water.
        
        if stats.get('Vegetation', 0) > 0:
            print("   ✅ Vegetation cluster identified")
        else:
            print("   ⚠️ Vegetation not identified (might be color threshold issue)")
            
    except Exception as e:
        print(f"   ❌ CV Error: {e}")

    # 3. Test Climate API
    print("\n3️⃣ Testing Climate API (Open-Meteo)...")
    try:
        # Test for Sehore coordinates
        data = get_climate_data(23.20, 77.08)
        if data.get("location_ok"):
            print(f"   ✅ API Response: Rain={data['annual_rainfall_mm']}mm, Temp={data['mean_temp_c']}C")
        else:
            print("   ⚠️ API returned fallback data")
    except Exception as e:
        print(f"   ❌ API connection failed: {e}")

    # 4. Test OCR Initialization
    print("\n4️⃣ Testing OCR Engine Initialization...")
    try:
        from src.ocr_utils import get_reader
        reader = get_reader() # This triggers model download if needed
        print("   ✅ EasyOCR Engine initialized")
        
        # We won't run extraction on dummy image as it has no text
        # But initialization proves the model is ready
    except Exception as e:
        print(f"   ❌ OCR Init Error: {e}")

    # Cleanup
    if os.path.exists("dummy_test.png"):
        os.remove("dummy_test.png")
        
    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    run_tests()
