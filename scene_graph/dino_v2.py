# from transformers import AutoImageProcessor, Dinov2Model
# import torch


# image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
# model = Dinov2Model.from_pretrained("facebook/dinov2-large")


# if __name__ == "__main__":
#     x = torch.rand(1, 3, 768, 768)
#     output = model(x)
#     print(output.last_hidden_state.shape)



from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large')

inputs = processor(images=image, return_tensors="pt")

print(inputs.pixel_values.shape)
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state



