import os
import json
import cv2
import numpy as np

# Suppress TensorFlow verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

MODEL_PATH = 'isl_mobilenet.keras'
LABELS_PATH = 'class_indices.json'
IMG_SIZE = (128, 128)

def load_resources():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found!")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Labels file '{LABELS_PATH}' not found!")
    
    print(f"Loading model from '{MODEL_PATH}'...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print(f"Loading class indices from '{LABELS_PATH}'...")
    with open(LABELS_PATH, 'r') as f:
        labels_map = json.load(f)
    
    labels_dict = {int(k): v for k, v in labels_map.items()}
    return model, labels_dict

def main():
    try:
        model, labels_map = load_resources()
        print("Model and labels loaded successfully.")
    except Exception as e:
        print(f"Error loading resources: {e}")
        return

    # Initialize video capture (0 for default webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the webcam (or running in headless environment).")
        print("Running a dry-run test on a sample frame to verify model inference...")
        dummy_frame = np.zeros((128, 128, 3), dtype=np.float32)
        dummy_input = np.expand_dims(dummy_frame / 255.0, axis=0)
        preds = model.predict(dummy_input, verbose=0)
        top_idx = np.argmax(preds[0])
        print(f"Dry-run inference test successful!")
        print(f"Predicted class index: {top_idx} -> Label: '{labels_map.get(top_idx, str(top_idx))}', Confidence: {preds[0][top_idx]:.2f}")
        return

    print("Webcam feed started successfully! Place your hand inside the green box.")
    print("Press 'q' in the video window to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting...")
            break

        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Define Region of Interest (ROI) box
        roi_top, roi_bottom = int(h * 0.2), int(h * 0.7)
        roi_left, roi_right = int(w * 0.5), int(w * 0.9)

        # Draw ROI box on screen
        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
        cv2.putText(frame, "Place Sign Here", (roi_left, roi_top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Preprocess ROI
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        if roi.size != 0:
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_resized = cv2.resize(roi_rgb, IMG_SIZE)
            roi_normalized = roi_resized.astype("float32") / 255.0
            roi_input = np.expand_dims(roi_normalized, axis=0)

            # Predict ISL gesture
            predictions = model.predict(roi_input, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx]
            predicted_label = labels_map.get(class_idx, str(class_idx))

            # Render overlay text
            display_text = f"Sign: {predicted_label} ({confidence * 100:.1f}%)"
            cv2.putText(frame, display_text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        cv2.imshow("Indian Sign Language (ISL) Detector", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()