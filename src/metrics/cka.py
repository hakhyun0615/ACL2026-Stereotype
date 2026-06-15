import numpy as np
import torch


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    device = _get_device()
    X = torch.tensor(X, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.float32, device=device)

    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)

    hsic_xy = torch.norm(X.T @ Y, "fro") ** 2
    hsic_xx = torch.norm(X.T @ X, "fro") ** 2
    hsic_yy = torch.norm(Y.T @ Y, "fro") ** 2

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float((hsic_xy / denom).item())


def kernel_cka(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    device = _get_device()
    X = torch.tensor(X, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.float32, device=device)

    dists_x = torch.cdist(X, X)
    dists_y = torch.cdist(Y, Y)

    if sigma is None:
        all_dists = torch.cat([dists_x.view(-1), dists_y.view(-1)])
        sigma = float(torch.median(all_dists).item())
        if sigma < 1e-8:
            sigma = 1.0

    K_x = torch.exp(-dists_x ** 2 / (2 * sigma ** 2))
    K_y = torch.exp(-dists_y ** 2 / (2 * sigma ** 2))

    n = K_x.shape[0]
    H = torch.eye(n, device=device) - torch.ones(n, n, device=device) / n
    K_x = H @ K_x @ H
    K_y = H @ K_y @ H

    hsic_xy = (K_x * K_y).sum()
    hsic_xx = (K_x * K_x).sum()
    hsic_yy = (K_y * K_y).sum()

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float((hsic_xy / denom).item())


def bootstrap_cka(X: np.ndarray, Y: np.ndarray, n_boot: int = 1000,
                   ci: float = 0.95, kernel: str = "linear") -> dict:
    cka_fn = linear_cka if kernel == "linear" else kernel_cka
    n = X.shape[0]
    boot_values = []

    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        boot_values.append(cka_fn(X[idx], Y[idx]))

    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(boot_values)),
        "std": float(np.std(boot_values)),
        "ci_lower": float(np.percentile(boot_values, alpha * 100)),
        "ci_upper": float(np.percentile(boot_values, (1 - alpha) * 100)),
        "values": boot_values,
    }


def layerwise_cka(hidden_states_npz: dict, layers: list[int],
                   kernel: str = "linear", n_boot: int = None) -> dict:
    cka_fn = linear_cka if kernel == "linear" else kernel_cka
    results = {}

    for layer in layers:
        X_en = hidden_states_npz[f"layer_{layer}_en"]
        X_ja = hidden_states_npz[f"layer_{layer}_ja"]
        cka_val = cka_fn(X_en, X_ja)

        entry = {"cka": cka_val, "layer": layer}

        if n_boot and X_en.shape[0] < 30:
            boot = bootstrap_cka(X_en, X_ja, n_boot=n_boot, kernel=kernel)
            entry.update(boot)

        results[layer] = entry

    return results
