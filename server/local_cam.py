import cv2
from flask import Flask, Response

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # 0 = default laptop webcam

def generate():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Resize to match Pi frame (important for zone consistency)
        frame = cv2.resize(frame, (640, 480))

        # --- INVERT THE IMAGE HERE ---
        # 1  = Horizontal flip (mirror effect, common for webcams)
        # 0  = Vertical flip (upside down)
        # -1 = Both horizontal and vertical flip
        # (Note: if you meant invert COLORS, use `frame = cv2.bitwise_not(frame)`)
        frame = cv2.flip(frame, 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5123, threaded=True)