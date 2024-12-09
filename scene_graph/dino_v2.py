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
from torchvision import transforms, utils


transforms = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor()
])

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant', size=518)
model = AutoModel.from_pretrained('facebook/dinov2-giant')

inputs = transforms(image).unsqueeze(0)


# inputs = processor(images=image, 
#                   size={'height': 518, 'width': 518},  # Explicit size dict
#                   return_tensors="pt")


outputs = model(inputs)
last_hidden_states = outputs.last_hidden_state
# remove the class token
last_hidden_states = last_hidden_states[:, 1:, :]

print(last_hidden_states.shape)



