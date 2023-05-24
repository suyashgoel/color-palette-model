from flask import Flask, request, jsonify
import werkzeug
import tensorflow as tf
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

    def input_fn():
        return tf.compat.v1.train.limit_epochs(tf.convert_to_tensor(modified_image, dtype=tf.float32), num_epochs=1)

    kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=number_of_colors, use_mini_batch=False)

    previousCenters = None
    numIter = 10
    for _ in range(numIter):
        kmeans.train(input_fn)
        clusterCenters = kmeans.cluster_centers()
        previousCenters = clusterCenters

    clusterCenters = kmeans.cluster_centers()
    clusterLabels = list(kmeans.predict_cluster_index(input_fn))

    return clusterCenters

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

        # Convert predicted colors to HEX format
        hex_colors = [RGB2HEX(color) for color in predicted_colors]

        return jsonify({
            "message": "Image Uploaded Successfully",
            "colors": hex_colors
        })

if __name__ == "__main__":
    app.run(debug=True, port=6000)


