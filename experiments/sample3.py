# tensor([[[0, 0, 1],
#          [0, 0, 2],
#          [0, 1, 1],
#          [0, 1, 2],
#          [0, 2, 0],
#          [0, 2, 1]],

#         [[1, 0, 0],
#          [1, 0, 2],
#          [1, 1, 0],
#          [1, 1, 1],
#          [1, 2, 0],
#          [1, 2, 2]]])



# org_indices
# tensor([[1, 2, 3],
#         [1, 2, 3]])



import torch
from einops import rearrange, repeat, pack 

indices = torch.tensor([[[0, 0, 1],
                        [0, 0, 2],
                        [0, 1, 1],
                        [0, 1, 2],
                        [0, 2, 0],
                        [0, 2, 1]],

                        [[1, 0, 0],
                        [1, 0, 2],
                        [1, 1, 0],
                        [1, 1, 1],
                        [1, 2, 0],
                        [1, 2, 2]]])


top_k_indices = torch.tensor([[1, 2, 3],
                              [0, 2, 3]])



print(top_k_indices.shape)


top_k_indices = repeat(top_k_indices, 'b k -> b n k', n=6)


print(indices.shape)
print(top_k_indices)

index = top_k_indices.gather(-1, indices)
print(index)


