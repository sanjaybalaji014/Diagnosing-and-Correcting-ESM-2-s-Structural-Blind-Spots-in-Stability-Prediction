import numpy as np

path = "data/processed/multilayer_embeddings.npz"
data = np.load(path, allow_pickle=True)
print("keys in npz:", list(data.keys()))
for k in data.keys():
    arr = data[k]
    print(f"  {k}: shape={arr.shape} dtype={arr.dtype}")
    if k == "keys":
        print("    sample:", arr[:3])