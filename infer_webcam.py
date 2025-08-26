import cv2
import numpy as np
import tensorflow as tf
import time
import paho.mqtt.client as mqtt
from pathlib import Path
from scipy.io import wavfile
import sounddevice as sd

IMG_SIZE = 64
CLOSED_FRAMES_THRESHOLD = 48  # ~2s at ~24 FPS
MQTT_ENABLED = True
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "tinyml/drowsy/alert"

ALARM_WAV = Path("alarms/alarm.wav")

def play_alarm():
    if not ALARM_WAV.exists():
        print("Alarm WAV not found, printing alert instead!")
        return
    sr, data = wavfile.read(str(ALARM_WAV))
    sd.play(data, sr)

def preprocess_eye(gray_crop):
    eye = cv2.resize(gray_crop, (IMG_SIZE, IMG_SIZE))
    eye = eye.astype(np.float32) / 255.0
    eye = np.expand_dims(eye, axis=(0, -1))
    return eye

def main():
    print("Loading model...")
    model = tf.keras.models.load_model("models/best_model.keras")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    mqtt_client = None
    if MQTT_ENABLED:
        try:
            mqtt_client = mqtt.Client()
            mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            mqtt_client.loop_start()
            print(f"MQTT connected to {MQTT_BROKER}:{MQTT_PORT}")
        except Exception as e:
            print("MQTT init failed:", e)
            mqtt_client = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    closed_run = 0
    alarm_active = False
    label_map = {0: "open", 1: "close"}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(80,80))
        both_closed = False

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color= frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 8, minSize=(20,20))
            eyes = sorted(eyes, key=lambda e: e[0])[:2]

            eye_states = []
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                eye_crop = roi_gray[ey:ey+eh, ex:ex+ew]
                eye_in = preprocess_eye(eye_crop)
                probs = model.predict(eye_in, verbose=0)[0]
                pred = int(np.argmax(probs))
                conf = float(np.max(probs))
                eye_states.append((pred,conf))
                cv2.putText(roi_color, f"{label_map[pred]} {conf:.2f}", (ex,ey-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)

            if len(eye_states) >= 2:
                both_closed = all(p==1 for p,c in eye_states)

        if both_closed:
            closed_run += 1
        else:
            closed_run = 0
            alarm_active = False

        status = f"closed_frames={closed_run}/{CLOSED_FRAMES_THRESHOLD}"
        cv2.putText(frame, status, (12,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

        if closed_run >= CLOSED_FRAMES_THRESHOLD and not alarm_active:
            alarm_active = True
            print("DROWSINESS DETECTED!")
            play_alarm()
            if mqtt_client:
                try:
                    mqtt_client.publish(MQTT_TOPIC, "DROWSINESS_DETECTED")
                    print(f"MQTT published to {MQTT_TOPIC}")
                except Exception as e:
                    print("MQTT publish failed:", e)

        cv2.imshow("Drowsiness detector (TinyML-ready)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()

if __name__ == "__main__":
    main()