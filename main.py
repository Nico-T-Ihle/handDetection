import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Finger-/Landmark-Namen für schöneren Log
landmark_names = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

# Globale Variable zum Speichern der letzten Ergebnisse
latest_result = None

# Callback, wenn neue Hand-Ergebnisse da sind
def print_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

    if result.hand_landmarks:
        for hand_idx, hand in enumerate(result.hand_landmarks):
            print(f"\n=== Hand {hand_idx} erkannt ===")
            for lm_idx, lm in enumerate(hand):
                name = landmark_names[lm_idx] if lm_idx < len(landmark_names) else f"Point {lm_idx}"
                print(f"  [{name:20}] Index: {lm_idx:2} | x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:.4f}")

# Landmarker konfigurieren
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,  # bis zu 2 Hände tracken
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=print_result
)

# Hauptloop
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Fehler: Konnte Webcam nicht öffnen")
        exit()

    timestamp = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fehler: Frame konnte nicht gelesen werden")
            break

        # Mediapipe erwartet RGB, OpenCV liefert BGR → konvertieren
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Hände asynchron erkennen
        landmarker.detect_async(mp_image, timestamp)
        timestamp += 33  # ungefähr 30 FPS

        # Skelette auf Frame zeichnen, falls Resultat da
        if latest_result and latest_result.hand_landmarks:
            image_h, image_w, _ = frame.shape
            for hand in latest_result.hand_landmarks:
                # Punkte verbinden (z.B. Handfläche & Finger)
                connections = [
                    (0,1),(1,2),(2,3),(3,4),             # Daumen
                    (0,5),(5,6),(6,7),(7,8),             # Zeigefinger
                    (0,9),(9,10),(10,11),(11,12),        # Mittelfinger
                    (0,13),(13,14),(14,15),(15,16),      # Ringfinger
                    (0,17),(17,18),(18,19),(19,20),      # Kleiner Finger
                    (5,9),(9,13),(13,17),(17,5)          # Handfläche umranden
                ]
                # Punkte in Pixelkoordinaten umrechnen
                landmark_points = []
                for lm in hand:
                    x_px, y_px = int(lm.x * image_w), int(lm.y * image_h)
                    landmark_points.append((x_px, y_px))

                # Linien zeichnen
                for start_idx, end_idx in connections:
                    cv2.line(frame, landmark_points[start_idx], landmark_points[end_idx], (0,255,0), 2)

                # Optional: Punkte zeichnen
                for point in landmark_points:
                    cv2.circle(frame, point, 4, (0,0,255), -1)

        cv2.imshow('Hand Tracking + Skeleton', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC zum Beenden
            break

    cap.release()
    cv2.destroyAllWindows()
