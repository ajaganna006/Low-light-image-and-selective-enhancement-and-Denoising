from flask import Flask, request, render_template, send_file
import os
from inference import inference  # your existing file

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]

        if file:
            input_path = os.path.join(UPLOAD_FOLDER, file.filename)
            output_path = os.path.join(OUTPUT_FOLDER, "out_" + file.filename)

            file.save(input_path)

            # 🔥 Call your ML model
            inference(input_path, output_path)

            return render_template("index.html", output_image=output_path)

    return render_template("index.html", output_image=None)


@app.route("/output/<filename>")
def output_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
