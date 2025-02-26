import torch
from fast_soft_sort.pytorch_ops import soft_sort, soft_rank
import matplotlib.pyplot as plt

values = torch.tensor([[5., 1., 2.],
                       [2., 1., 5.]], dtype=torch.float32)

print("Original values:")
print(values)


print("\n=== Testing soft_sort ===")

#more soft
sorted_rs1 = soft_sort(values, regularization_strength=1.0)
print("\nSoft sort (regularization_strength = 1.0):")
print(sorted_rs1)

#less soft
sorted_rs2 = soft_sort(values, regularization_strength=0.1)
print("\nSoft sort (regularization_strength = 0.1):")
print(sorted_rs2)

exact_sorted = torch.stack([torch.sort(row)[0] for row in values])
print("\nExact sorted (using torch.sort):")
print(exact_sorted)


print("\n=== Testing soft_rank ===")

rank_sr1 = soft_rank(values, regularization_strength=2.0)
print("\nSoft rank (regularization_strength = 2.0):")
print(rank_sr1)

rank_sr2 = soft_rank(values, regularization_strength=1.0)
print("\nSoft rank (regularization_strength = 1.0):")
print(rank_sr2)


print("\nPlotting soft_rank vs. regularization strength (for first row)...")

reg_strengths = [0.1 * i for i in range(1, 11)]
ranks_first_row = []

for reg in reg_strengths:
    sr = soft_rank(values, regularization_strength=reg)
    ranks_first_row.append(sr[0].detach().numpy())

plt.figure(figsize=(8, 6))
for i in range(values.shape[1]):
    plt.plot(reg_strengths, [rank[i] for rank in ranks_first_row],
             marker='o', label=f'Element {i} (original value {values[0, i].item()})')

plt.xlabel("Regularization Strength")
plt.ylabel("Soft Rank (first row)")
plt.title("Soft Rank of first row vs. Regularization Strength")
plt.legend()
plt.grid(True)
plt.show()



