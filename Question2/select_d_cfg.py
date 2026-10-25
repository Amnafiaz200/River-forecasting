"""Candidate designs for model D (screened on the validation split only)."""
CANDS = {
    "p8_k3": dict(patch=8, hidden=64, kernels=(3, 3), dilations=(1, 1)),   # 8x8 grid, 8 px per cell
    "p4_k5_h32": dict(patch=4, hidden=32, kernels=(5, 5), dilations=(1, 1)),  # 16x16 grid, 5x5 kernels
}
