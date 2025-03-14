from flask import Flask, render_template, Response
import cv2
import numpy as np
from threading import Thread, Lock

app = Flask(__name__)

# Import face recognition functions from your existing code
from FaceNet_Implementation import (
    detector, 
    get_face,
    get_embedding,
    load_embeddings,
    recognize_face
)

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.registered_faces = load_embeddings() or {}
        self.lock = Lock()
        self.running = True
        self.thread = Thread(target=self.update)
        self.thread.start()

    def __del__(self):
        self.running = False
        self.thread.join()
        self.video.release()

    def get_frame(self):
        with self.lock:
            _, frame = self.video.read()
            if frame is not None:
                # Perform face recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(rgb_frame)
                
                for face in faces:
                    face_img, (x1, y1), (x2, y2) = get_face(frame, face['box'])
                    try:
                        embedding = get_embedding(face_img)
                        name, confidence = recognize_face(embedding, self.registered_faces)
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{name} ({confidence:.2f})", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.8, color, 2)
                    except Exception as e:
                        print(f"Processing error: {str(e)}")
                
                _, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes()
            return None

    def update(self):
        while self.running:
            with self.lock:
                self.video.grab()

vcam = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = vcam.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    return Response(generate(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)