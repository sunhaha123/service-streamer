# coding=utf-8
# Created by Meteorix at 2019/8/9
import io
import json
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import time


imagenet_class_index = json.load(open('imagenet_class_index.json'))
device = "cuda"
# Make sure to pass `pretrained` as `True` to use the pretrained weights:
# model = models.densenet121(pretrained=True)
# model.to(device)
# # Since we are using our model only for inference, switch to `eval` mode:
# model.eval()
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        print('photo error!')
    # print("image mode:",image.mode)
    # print("image type:",type(image))
    if image.mode == "RGBA":
        image =  image.convert("RGB")
    return my_transforms(image).unsqueeze(0)


class ClsModel(object):
    def __init__(self):
        self.model = models.densenet121(pretrained=True)
        self.model.to(device)
        self.model.eval()

    def get_prediction(self,image_bytes):
        tensor = transform_image(image_bytes=image_bytes).to(device)
        outputs = self.model.forward(tensor)
        _, y_hat = outputs.max(1)
        predicted_idx = str(y_hat.item())
        return imagenet_class_index[predicted_idx]


    def batch_prediction(self,image_bytes_batch):
        image_tensors = [transform_image(image_bytes=image_bytes) for image_bytes in image_bytes_batch]
        tensor = torch.cat(image_tensors).to(device)
        outputs = self.model.forward(tensor)
        _, y_hat = outputs.max(1)
        predicted_ids = y_hat.tolist()
        return [imagenet_class_index[str(i)] for i in predicted_ids]


if __name__ == "__main__":
    with open(r"cat.jpg", 'rb') as f:
        image_bytes = f.read()
    with open(r"truck.jpg", 'rb') as f:
        image_bytes2 = f.read()
    #测试时间
    model = ClsModel()
    result = model.get_prediction(image_bytes)
    t0 = time.time()
    result = model.get_prediction(image_bytes)
    print(result)
    wasted_time = float(time.time()-t0)*6
    print('wasted time %5f\n'%wasted_time)
    #测试batch
    t0 = time.time()
    batch_result = model.batch_prediction([image_bytes,image_bytes2])
    # assert batch_result == [result] * 6
    print(batch_result)
    print('batch wasted time %5f'%(time.time()-t0))
