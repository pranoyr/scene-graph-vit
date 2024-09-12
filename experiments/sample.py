import torch


scores = torch.randint(10, (1, 4, 4))

scores = [scores, scores]

scores = torch.cat(scores, dim=0)


print(scores)
print(scores.shape)

diag = scores.diagonal(dim1=-2, dim2=-1)

top_k_scores, top_k_indices = torch.topk(diag, k=3, dim=-1)

print(top_k_indices)

relationship_scores_filtered = scores.gather(2, top_k_indices.unsqueeze(1))

print(relationship_scores_filtered)
