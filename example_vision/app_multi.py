from service_streamer import ManagedModel
from flask import Flask, jsonify, request
from model_new import ClsModel
from service_streamer import Streamer
#框架中已经包含了多线程队列，因此不要再使用gevent
# from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = None
streamer = None

class ManagedClstModel(ManagedModel):

    def init_model(self):
        self.model = ClsModel()

    def predict(self, batch):
        return self.model.batch_prediction(batch)



@app.route('/stream', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()

        class_id, class_name = streamer.predict([img_bytes])[0]
        return jsonify({'class_id': class_id, 'class_name': class_name})


@app.route('/t')
def hello():
    return 'Hello World'


if __name__ == "__main__":
    streamer = Streamer(ManagedClstModel, batch_size=32, max_latency=0.1, worker_num=2, cuda_devices=(0,))
    app.run(host="0.0.0.0", port=5005)



    #通过图片测试
    # with open(r"cat.jpg", 'rb') as f:
    #     image_bytes = f.read()
    # with open(r"truck.jpg", 'rb') as f:
    #     image_bytes2 = f.read()
    # outputs = streamer.predict([image_bytes,image_bytes2])
    # print(outputs)
