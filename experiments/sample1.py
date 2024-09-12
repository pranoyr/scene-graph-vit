import torch

from einops import rearrange, repeat, pack

scores = torch.randint(10, (2, 4, 4))


print(scores)

#  get diagonal
diag = scores.diagonal(dim1=-2, dim2=-1)


print("diag")
print(diag)

# get top k relationships
top_k_scores, top_k_indices = torch.topk(diag, k=3, dim=-1)
top_k_indices = torch.sort(top_k_indices, dim=-1, descending=False)[0]

# copt to top_k_indices1
top_k_indices1 = top_k_indices.clone()

scores = scores[torch.arange(scores.size(0)).unsqueeze(1), top_k_indices]


# print("scores")
# print(scores)

top_k_indices = repeat(top_k_indices, 'b n ->  b r n', r=scores.shape[1])

print("top_k_indices")
print(top_k_indices)


relationship_scores = scores.gather(-1, top_k_indices)

print("relationship_scores")
print(relationship_scores)



top_k_rel_indices = torch.topk(relationship_scores, k=2, dim=-1)[1]


print("top_k_indices_for_rel")
print(top_k_rel_indices)

split_shape = top_k_rel_indices.shape[-1] * top_k_rel_indices.shape[-2]


top_k_rel_indices = relationship_scores.scatter(-1, top_k_rel_indices, -1)


indices = torch.where(top_k_rel_indices == -1)



# stack indices
indices = torch.stack(indices, dim=-1)


print(indices)
print("split_shape")
print(split_shape)

indices = indices.split(split_shape, dim=-2)

indices = torch.stack(indices)



print("indices")
print(indices)


print("org_indices")
print(top_k_indices1)




top_k_indices1 = repeat(top_k_indices1, 'b k -> b n k', n=split_shape)

subject_object_indices = top_k_indices1.gather(-1, indices)

# Replace the first index in subject_object_indices with batch id
batch_size = subject_object_indices.shape[0]

print(subject_object_indices)

batch_ids = torch.arange(batch_size).unsqueeze(-1).unsqueeze(-1).expand_as(subject_object_indices[:, :, :1])
# subject_object_indices = torch.cat([batch_ids, subject_object_indices[:, :, 1:]], dim=-1)

print("final_indices")


 # Replace the first index in subject_object_indices with batch id
batch_size = subject_object_indices.shape[0]

batch_ids = torch.arange(batch_size).unsqueeze(-1).unsqueeze(-1).expand_as(subject_object_indices[:, :, :1])



subject_indices = torch.cat([batch_ids, subject_object_indices[:, :, 1:2]], dim=-1)
object_indices = torch.cat([batch_ids, subject_object_indices[:, :, 2:3]], dim=-1)


print(subject_indices)

q = torch.randn(2, 4, 512)

print(q)



# Extract corresponding q values using subject_indices
subject_embeds = q[subject_indices[:, :, 0], subject_indices[:, :, 1]]
object_embeds = q[object_indices[:, :, 0], object_indices[:, :, 1]]



print(subject_embeds)
