from flask import Flask, request, jsonify, render_template
import base64
import cv2

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# 🔥 THIS IS THE MAIN API YOUR UI CALLS
@app.route('/enhance', methods=['POST'])
def enhance():
    file = request.files['file']

    # Save file
    file_path = "input.jpg"
    file.save(file_path)

    # 🔥 CALL YOUR MODEL HERE
    # Replace with your inference logic
    img = cv2.imread(file_path)
    enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=30)

    # Convert to base64
    _, buffer1 = cv2.imencode('.png', img)
    _, buffer2 = cv2.imencode('.png', enhanced)

    original_base64 = base64.b64encode(buffer1).decode('utf-8')
    enhanced_base64 = base64.b64encode(buffer2).decode('utf-8')

    return jsonify({
        "success": True,
        "original": original_base64,
        "enhanced": enhanced_base64,
        "method": "demo"
    })

# VIDEO ROUTE
@app.route('/video', methods=['POST'])
def video():
    file = request.files['video']
    file.save("static/output.mp4")

    return render_template("index.html",
        original_video="output.mp4",
        output_video="output.mp4",
        metrics={"time_taken":1,"frames":100,"fps":24}
    )

if __name__ == "__main__":
    app.run()
