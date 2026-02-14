import re
import os
import sys

# Lazy loading for EasyOCR to prevent app triggers on import
_reader = None
_easyocr_available = None

def get_reader():
    global _reader, _easyocr_available
    
    if _easyocr_available is False:
        return None

    if _reader is None:
        try:
            print("  📝 Initializing EasyOCR engine...")
            import easyocr
            # Use CPU mode for compatibility
            _reader = easyocr.Reader(['en'], gpu=False) 
            _easyocr_available = True
        except Exception as e:
            print(f"  ❌ OCR Engine Failed to Load: {e}")
            _easyocr_available = False
            return None
            
    return _reader

def extract_coordinates(image_path):
    """
    Extract latitude and longitude from image text.

    Parameters
    ----------
    image_path : str
        Path to the image.

    Returns
    -------
    tuple : (latitude, longitude) or (None, None) if not found.
    """
    if not os.path.exists(image_path):
        return None, None

    try:
        reader = get_reader()
        if reader is None:
            # OCR engine failed to load (e.g. DLL error)
            # Fail silently so the rest of the app works
            print("  ⚠️ OCR Engine unavailable. Skipping coordinate extraction.")
            return None, None
            
        results = reader.readtext(image_path, detail=0)
        
        # Combine all text into one string for easier regex
        full_text = " ".join(results)
        # print(f"     OCR Text: {full_text[:100]}...") # Debug

        # Regex patterns for coordinates
        # Pattern 1: Decimal degrees (e.g., "Lat: 23.25, Long: 77.15")
        decimal_pattern = r"(?:Lat|Latitude)[:\s]*([+-]?\d+\.?\d*)[,\s]*(?:Lon|Long|Longitude)[:\s]*([+-]?\d+\.?\d*)"
        
        # Pattern 2: BMS format (e.g., "23°15'N 77°05'E")
        dms_pattern = r"(\d+)°(\d+)'?([NS])[\s,]*(\d+)°(\d+)'?([EW])"
        
        # Try Pattern 1 (Decimal)
        match = re.search(decimal_pattern, full_text, re.IGNORECASE)
        if match:
            lat = float(match.group(1))
            lon = float(match.group(2))
            return lat, lon

        # Try Pattern 2 (DMS)
        match = re.search(dms_pattern, full_text, re.IGNORECASE)
        if match:
            d_lat = float(match.group(1))
            m_lat = float(match.group(2))
            hem_lat = match.group(3).upper()
            
            d_lon = float(match.group(4))
            m_lon = float(match.group(5))
            hem_lon = match.group(6).upper()
            
            lat = d_lat + (m_lat / 60.0)
            if hem_lat == 'S': lat = -lat
            
            lon = d_lon + (m_lon / 60.0)
            if hem_lon == 'W': lon = -lon
            
            return lat, lon

        # Fallback: Just look for numbers near "N" and "E" if explicit "Lat" missing
        # e.g. "23.25 N 77.15 E"
        simple_pattern = r"(\d+\.\d+)\s*N[\s,]*(\d+\.\d+)\s*E"
        match = re.search(simple_pattern, full_text, re.IGNORECASE)
        if match:
            return float(match.group(1)), float(match.group(2))

        return None, None

    except Exception as e:
        print(f"  ❌ OCR Error: {e}")
        return None, None
