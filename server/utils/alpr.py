import cv2
import easyocr
import logging

log = logging.getLogger(__name__)

# Initialize the OCR reader once (it will download models on first run)
log.info("Initializing EasyOCR Model for License Plates...")
reader = easyocr.Reader(['en'], gpu=True) # Set gpu=False if you run into memory errors

def extract_license_plate(raw_frame, bbox) -> str:
    """
    Crops the vehicle from the frame and uses OCR to find the license plate text.
    """
    try:
        x1, y1, x2, y2 = bbox
        
        # Crop the vehicle from the full high-res frame
        vehicle_crop = raw_frame[y1:y2, x1:x2]
        
        # Convert to grayscale to help the OCR
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        
        # Run OCR
        results = reader.readtext(gray)
        
        if not results:
            return "UNKNOWN_PLATE"
            
        # Filter for the largest/most confident text block
        # (This is a simplified heuristic; real ALPR models are trained specifically on plates)
        best_text = ""
        highest_conf = 0.0
        
        for (coord, text, prob) in results:
            # Clean the text (remove spaces and special chars)
            clean_text = ''.join(e for e in text if e.isalnum()).upper()
            if prob > highest_conf and len(clean_text) > 4:
                highest_conf = prob
                best_text = clean_text
                
        return best_text if best_text else "UNKNOWN_PLATE"
        
    except Exception as e:
        log.error("ALPR Error: %s", e)
        return "UNKNOWN_PLATE"