from flask import Flask, request, jsonify
import werkzeug
import sklearn
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
import cv2

app = Flask(__name__)

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def predict_colors(image_path, number_of_colors):
    image = get_image(image_path)
    modified_image = image.reshape(image.shape[0]*image.shape[1], 3)

    clf = MiniBatchKMeans(n_clusters=number_of_colors).fit(modified_image)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]

    return hex_colors

@app.route('/upload', methods=["POST"])
def upload():
    if request.method == "POST":
        imageFile = request.files['image']
        fileName = werkzeug.utils.secure_filename(imageFile.filename)
        imageFile.save("./uploadedimages/" + fileName)

        # Set the desired number of colors
        number_of_colors = 5

        # Predict colors
        image_path = "./uploadedimages/" + fileName
        predicted_colors = predict_colors(image_path, number_of_colors)

        return jsonify({
            "colors": predicted_colors
        })

if __name__ == "__main__":
    app.run(debug=True, port=8000)


