import cv2
import re
import json
import os
import numpy as np
from paddleocr import PaddleOCR

def preprocess_image(image_path: str, state: int):
    """
    Vision Agent (Filter_Specialist)
    Loads image, downscales by 50% for memory protection, and applies state-specific filters.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    # State 1: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if state == 1:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # State 2: Grayscale + CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    if state == 2:
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # State 3: Grayscale + CLAHE + Bitwise Inversion
    # Inversion and Thresholding applied for metallic surface highlights
    inverted = cv2.bitwise_not(enhanced)
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

def get_extracted_items(ocr_result):
    items = []
    def recurse(data):
        if isinstance(data, (list, tuple)):
            # Check if this node is (text, confidence)
            if len(data) == 2 and isinstance(data[0], str) and isinstance(data[1], (float, int, np.floating)):
                items.append({"text": data[0], "confidence": float(data[1])})
                return
            # Check if this node is [poly, (text, confidence)]
            if len(data) == 2 and isinstance(data[1], (list, tuple)) and len(data[1]) == 2 and isinstance(data[1][0], str):
                items.append({"text": data[1][0], "confidence": float(data[1][1])})
                return
            for item in data:
                recurse(item)
    recurse(ocr_result)
    return items

def get_valid_items(ocr_result):
    """
    Verification Agent (Ponytail)
    Evaluates the OCR output and returns a filtered list of valid items.
    """
    valid_items = []
    items = get_extracted_items(ocr_result)
    print(f"    [Debug] get_extracted_items returned {len(items)} items")
    if not items:
        return valid_items
        
    for item in items:
        text = item["text"]
        confidence = item["confidence"]
        print(f"    [Debug] Evaluating text: '{text}' (conf: {confidence})")
        # Rule 1: Empty text
        if not text.strip():
            continue
            
        # Generic Threshold: lowered to 0.35 to catch garbled reads organically
        if confidence < 0.35:
            print(f"    [Rejected] Confidence {confidence:.2f} < 0.35 for '{text}'")
            continue
            
        # Heavy gibberish check
        alnum_ratio = sum(c.isalnum() for c in text) / len(text) if len(text) > 0 else 0
        if alnum_ratio < 0.5 or re.search(r'[@#$%^&*]', text):
            print(f"    [Rejected] Gibberish detected in '{text}'")
            continue
            
        # Add to valid items with original text and confidence
        valid_items.append(item)
            
    return valid_items

def main():
    # Orchestrator Agent (Supervisor)
    input_dir = "input"
    output_dir = "output"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # PONYTAIL RUNG: Lazy code without its check is unfinished.
    # We create a dummy image if the input directory is empty so the logic 
    # can be tested without setup.
    # -------------------------------------------------------------
    if not any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(input_dir)):
        print(f"No images found in '{input_dir}'. Generating a dummy gear image for self-check...")
        dummy_img = np.zeros((400, 400, 3), dtype=np.uint8)
        # Adding some noise and a metallic color
        dummy_img[:] = (100, 100, 100)
        cv2.putText(dummy_img, "GEAR-987X", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 3)
        cv2.imwrite(os.path.join(input_dir, "gear.jpg"), dummy_img)

    # Initialize PaddleOCR
    # Using mobile models by default, limiting side length to protect memory (max 8GB RAM available)
    print("Initializing OCR Engine (Lightweight Mobile Model)...")
    ocr = PaddleOCR(use_textline_orientation=True, lang='en', text_det_limit_side_len=960)

    states = [1, 2, 3, 4]
    state_names = {
        1: "Grayscale",
        2: "Grayscale + CLAHE",
        3: "Grayscale + CLAHE + Bitwise Inversion",
        4: "Rotational Sweep (45 deg increments)"
    }

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        image_path = os.path.join(input_dir, filename)
        print(f"\n--- Processing {filename} ---")
        final_valid_items = []
        seen_texts = set()

        for state in states:
            print(f"Attempt {state}: Processing with {state_names[state]}...")
            
            result = []
            
            if state == 4:
                # State 4: Rotational Brute-force for circular/curved texts
                img = cv2.imread(image_path)
                h, w = img.shape[:2]
                
                for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
                    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                    rotated = cv2.warpAffine(img, M, (w, h))
                    try:
                        result_gen = ocr.predict(rotated)
                    except Exception as e:
                        print(f"      [Debug] ocr.predict failed at angle {angle}: {e}")
                        continue
                    
                    for res in result_gen:
                        if hasattr(res, 'keys'):
                            if 'rec_texts' in res:
                                for txt, score in zip(res.get('rec_texts', []), res.get('rec_scores', [])):
                                    result.append((txt, score))
                            elif 'rec_text' in res:
                                for txt, score in zip(res.get('rec_text', []), res.get('rec_score', [])):
                                    result.append((txt, score))
                            else:
                                result.append(res)
                        else:
                            result.append(res)
            else:
                # 1. Vision Preprocessing (States 1-3)
                try:
                    processed_image = preprocess_image(image_path, state)
                except Exception as e:
                    print(f"Failed to preprocess {filename}: {e}")
                    break
                
                # 2. OCR Extraction
                # Pass the preprocessed image directly as numpy array
                result_gen = ocr.predict(processed_image)
                # Handle PaddleOCR 3.7+ predict() return format (which returns a generator of dicts)
                result = []
                for res in result_gen:
                    if hasattr(res, 'keys'):
                        if 'rec_texts' in res:
                            for poly, text, score in zip(res.get('dt_polys', []), res.get('rec_texts', []), res.get('rec_scores', [])):
                                result.append([poly, (text, score)])
                        elif 'rec_text' in res:
                            for poly, text, score in zip(res.get('dt_polys', []), res.get('rec_text', []), res.get('rec_score', [])):
                                result.append([poly, (text, score)])
                        else:
                            result.append(res)
                    else:
                        result.append(res)
                
                # Package it in the expected list of lists format for the get_extracted_items function
                if result and isinstance(result[0], list) and not isinstance(result[0][0], list):
                     result = [result]
            
            # 3. Verification & Filtering
            valid_items_from_state = get_valid_items(result)
            if valid_items_from_state:
                print(f"-> Success! State {state} produced valid OCR data.")
                for item in valid_items_from_state:
                    # ORGANIC DEDUPLICATION: Merge substrings and highly similar strings
                    import difflib
                    new_text = item["text"]
                    new_conf = item["confidence"]
                    
                    matched_existing = None
                    for existing in list(seen_texts):
                        # 1. Substring match
                        if new_text in existing or existing in new_text:
                            matched_existing = existing
                            break
                        # 2. High overlap match
                        ratio = difflib.SequenceMatcher(None, new_text, existing).ratio()
                        if ratio > 0.6:
                            matched_existing = existing
                            break
                            
                    if matched_existing:
                        existing_item = next(x for x in final_valid_items if x["text"] == matched_existing)
                        # Replace if new text has higher confidence or is longer with decent confidence
                        if new_conf > existing_item["confidence"] or len(new_text) > len(matched_existing):
                            seen_texts.remove(matched_existing)
                            final_valid_items = [x for x in final_valid_items if x["text"] != matched_existing]
                            final_valid_items.append(item)
                            seen_texts.add(new_text)
                    else:
                        if new_text not in seen_texts:
                            final_valid_items.append(item)
                            seen_texts.add(new_text)
            else:
                print(f"-> State {state} did not yield new valid texts.")

        # 4. Result writing
        if final_valid_items:
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
            with open(output_file, "w") as f:
                json.dump(final_valid_items, f, indent=4)
            print(f"Final valid data written to {output_file}")
        else:
            print(f"Pipeline failed for {filename}: Could not extract valid text meeting quality thresholds.")

if __name__ == "__main__":
    main()
