from transformers import AutoImageProcessor, Dinov2Model
import torch


image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = Dinov2Model.from_pretrained("facebook/dinov2-base")


if __name__ == "__main__":
    x = torch.rand(1, 3, 256, 256)
    output = model(x)
    print(output.last_hidden_state.shape)



