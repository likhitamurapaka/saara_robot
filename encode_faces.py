"""
encode_faces.py - Uses dlib directly for maximum compatibility
"""

import os
import pickle
import numpy as np
import cv2
import dlib

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FACES_DIR   = os.path.join(BASE_DIR, "faces")
OUTPUT_FILE = os.path.join(BASE_DIR, "encodings.pkl")

SUPPORTED = (".jpg", ".jpeg", ".png", ".bmp")

# Download these model files automatically if not present
import urllib.request

MODELS_DIR         = os.path.join(BASE_DIR, "dlib_models")
PREDICTOR_PATH     = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
RECOGNITION_PATH   = os.path.join(MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat")

PREDICTOR_URL    = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
RECOGNITION_URL  = "https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2"

def download_models():
    import bz2
    os.makedirs(MODELS_DIR, exist_ok=True)

    for url, out_path in [
        (PREDICTOR_URL,   PREDICTOR_PATH),
        (RECOGNITION_URL, RECOGNITION_PATH),
    ]:
        if os.path.isfile(out_path):
            continue
        bz2_path = out_path + ".bz2"
        print(f"  Downloading {os.path.basename(out_path)}...")
        urllib.request.urlretrieve(url, bz2_path)
        print(f"  Extracting...")
        with bz2.open(bz2_path, "rb") as f_in, open(out_path, "wb") as f_out:
            f_out.write(f_in.read())
        os.remove(bz2_path)
        print(f"  [OK] {os.path.basename(out_path)}")

def encode_all():
    print("[INFO] Checking dlib models...")
    download_models()

    detector   = dlib.get_frontal_face_detector()
    predictor  = dlib.shape_predictor(PREDICTOR_PATH)
    face_rec   = dlib.face_recognition_model_v1(RECOGNITION_PATH)

    if not os.path.isdir(FACES_DIR):
        os.makedirs(FACES_DIR)
        print(f"[INFO] Created faces/ folder. Add photos then run again.")
        return

    files = [f for f in os.listdir(FACES_DIR)
             if os.path.splitext(f)[1].lower() in SUPPORTED]

    if not files:
        print("[INFO] No photos found in faces/ folder.")
        return

    known_encodings = []
    known_names     = []

    print(f"\n[INFO] Found {len(files)} photo(s). Encoding...\n")

    for filename in files:
        name = os.path.splitext(filename)[0]
        path = os.path.join(FACES_DIR, filename)
        print(f"  Processing: {filename}  (name = {name})")

        try:
            # Load with OpenCV → convert to RGB
            bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"  [ERROR] Could not read {filename} - skipping.")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Detect faces
            dets = detector(rgb, 1)
            if len(dets) == 0:
                print(f"  [WARNING] No face found in {filename} - skipping.")
                print(f"            Make sure face is clearly visible and well lit.")
                continue

            if len(dets) > 1:
                print(f"  [WARNING] Multiple faces in {filename} - using first only.")

            # Get face landmarks and encoding
            shape = predictor(rgb, dets[0])
            enc   = np.array(face_rec.compute_face_descriptor(rgb, shape))

            known_encodings.append(enc)
            known_names.append(name)
            print(f"  [OK] Encoded {name}")

        except Exception as e:
            print(f"  [ERROR] {filename}: {e} - skipping.")
            continue

    if not known_encodings:
        print("\n[ERROR] No valid faces encoded. Check your photos.")
        return

    data = {"encodings": known_encodings, "names": known_names}
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"\n[DONE] Encoded {len(known_names)} member(s):")
    for n in known_names:
        print(f"       - {n}")
    print(f"\n[SAVED] encodings.pkl saved!")
    print("[READY] You can now run:  python robot_eyes.py")

if __name__ == "__main__":
    encode_all()