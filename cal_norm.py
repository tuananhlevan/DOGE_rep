import torch
from tqdm.auto import tqdm

def compute_norm(merged_weights):
    layers = list(merged_weights.keys())
    num_layers = len(layers)

    # We'll compute the average Frobenius norm across layers
    fro_norm_sum = 0.0

    for layer_key in tqdm(layers, desc="Computing norms"):
        matrix = merged_weights[layer_key]
        # Frobenius norm (most common for weight matrices)
        fro_norm = torch.norm(matrix, p='fro').item()
        fro_norm_sum += fro_norm

    avg_fro_norm = fro_norm_sum / num_layers

    print(f"Average Frobenius norm of weight across {num_layers} layers: {avg_fro_norm:.6f}")

    # Optional: also compute other useful norms
    # Reset sum for clarity
    l1_norm_sum = 0.0
    l2_norm_sum = 0.0
    inf_norm_sum = 0.0

    for layer_key in layers:
        matrix = merged_weights[layer_key]
        l1_norm_sum += torch.norm(matrix, p=1).item()        # L1 norm
        l2_norm_sum += torch.norm(matrix, p=2).item()        # Spectral norm (largest singular value)
        inf_norm_sum += torch.norm(matrix, p=float('inf')).item()  # Max norm

    avg_l1_norm = l1_norm_sum / num_layers
    avg_spectral_norm = l2_norm_sum / num_layers
    avg_max_norm = inf_norm_sum / num_layers

    print(f"Average L1 norm:              {avg_l1_norm:.6f}")
    print(f"Average spectral norm (L2):   {avg_spectral_norm:.6f}")
    print(f"Average max norm (L-inf):     {avg_max_norm:.6f}")