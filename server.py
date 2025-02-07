import os
from flask import Flask, Response, request, jsonify
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("yolov8n.pt")

def generate_frames(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Ошибка подключения к видеопотоку")
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        for result in results[0].boxes:
            class_id = int(result.cls[0].item())
            if class_id == 0:  # Человек
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    url = data.get("url")
    protocol = data.get("protocol")

    if not url or not protocol:
        return jsonify({"error": "URL или протокол не указан"}), 400

    return Response(generate_frames(url), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
