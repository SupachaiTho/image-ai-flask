from flask import Flask, request, jsonify
from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()
prediction = ImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(execution_path, "resnet50_imagenet_tf.2.0.h5"))
prediction.loadModel()

app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'hello test'

@app.route('/object-detection', methods=['POST'])
def object_detection():
    file = request.files['image']
    file_name = file.filename
    file.save(file_name)
    file_path = os.path.join(execution_path, file_name)
    predictions, probabilities = prediction.classifyImage(file_path, result_count=100)
    data = {}
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        data[eachPrediction] = eachProbability
    print(file_name, file_path)
    os.remove(file_path)

    return jsonify({'msg': 'success', 'name': file.filename, 'data': data})

if __name__ == '__main__':
    app.run(debug=False)