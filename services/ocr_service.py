import os
import shutil
from typing import Dict, Any, Optional
from PIL import Image

# Candidate paths for tesseract binary on Windows
TESSERACT_CANDIDATES = [
    r"C:\Users\vk980\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
]

def get_tesseract_cmd() -> Optional[str]:
    """Find tesseract binary on Windows or system PATH."""
    which_path = shutil.which("tesseract")
    if which_path and os.path.exists(which_path):
        return which_path
    for candidate in TESSERACT_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None

def extract_text_from_image(image_input: Any) -> Dict[str, Any]:
    """
    Extract text from an uploaded image using pytesseract.
    Returns:
        dict: {
            "success": bool,
            "text": str,
            "error": Optional[str],
            "word_count": int
        }
    """
    try:
        import pytesseract
    except ImportError:
        return {
            "success": False,
            "text": "",
            "error": "pytesseract library is not installed in the environment.",
            "word_count": 0
        }

    tess_cmd = get_tesseract_cmd()
    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd
    else:
        return {
            "success": False,
            "text": "",
            "error": "Tesseract OCR engine executable not found. Please ensure Tesseract is installed.",
            "word_count": 0
        }

    try:
        # Load image if bytes or file object
        if isinstance(image_input, (str, bytes)) or hasattr(image_input, "read"):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            return {
                "success": False,
                "text": "",
                "error": "Unsupported image format provided.",
                "word_count": 0
            }

        # Convert to RGB if needed
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        extracted_text = pytesseract.image_to_string(image)
        cleaned_text = extracted_text.strip()

        if not cleaned_text:
            return {
                "success": False,
                "text": "",
                "error": "No readable text could be recognized in this image. Please upload a clearer image.",
                "word_count": 0,
                "confidence": 0.0,
                "is_low_confidence": True,
                "warning": "Low OCR confidence. Results may be unreliable."
            }

        words = cleaned_text.split()
        
        # Calculate word confidence scores
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit() and int(c) >= 0]
            avg_conf = (sum(confs) / len(confs)) if confs else 0.0
        except Exception:
            avg_conf = 70.0 if len(words) >= 5 else 45.0

        is_low_conf = (avg_conf < 60.0) or (len(words) < 3)
        warning_msg = "Low OCR confidence. Results may be unreliable." if is_low_conf else None

        return {
            "success": True,
            "text": cleaned_text,
            "error": None,
            "word_count": len(words),
            "confidence": round(avg_conf, 1),
            "is_low_confidence": is_low_conf,
            "warning": warning_msg
        }

    except Exception as e:
        return {
            "success": False,
            "text": "",
            "error": f"OCR processing encountered an error: {str(e)}",
            "word_count": 0,
            "confidence": 0.0,
            "is_low_confidence": True,
            "warning": "OCR extraction failed."
        }
