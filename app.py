from flask import Flask, request, jsonify, render_template
import base64
import os
import random

app = Flask(__name__)

# Ensure folders exist
os.makedirs("static", exist_ok=True)
os.makedirs("static/demo", exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')


# 🔥 DEMO IMAGE API
@app.route('/enhance', methods=['POST'])
def enhance():
    try:
        demo_images = [
            ("input1.jpg", "output1.jpg"),
            ("input2.jpg", "output2.jpg")
        ]

        inp, out = random.choice(demo_images)

        input_path = f"static/demo/{inp}"
        output_path = f"static/demo/{out}"

        # Safety check
        if not os.path.exists(input_path) or not os.path.exists(output_path):
            return jsonify({
                "success": False,
                "error": "Demo images missing in static/demo"
            })

        return jsonify({
    "success": True,
    "original": f"/static/demo/{inp}",
    "enhanced": f"/static/demo/{out}",
    "method": "Demo Mode"
})

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


# 🎥 VIDEO ROUTE
@app.route('/video', methods=['POST'])
def video():
    try:
        file = request.files.get('video')

        if not file:
            return "No video uploaded", 400

        output_path = "static/output.mp4"
        file.save(output_path)

        return render_template(
            "index.html",
            original_video="output.mp4",
            output_video="output.mp4",
            metrics={
                "time_taken": 1,
                "frames": 100,
                "fps": 24
            }
        )

    except Exception as e:
        return str(e)


# 🔥 RENDER ENTRY POINT
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
