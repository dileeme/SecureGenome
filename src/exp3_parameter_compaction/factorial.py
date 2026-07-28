import os
import csv
import time
import numpy as np
import tenseal as ts
from tqdm import tqdm

N_INDIVIDUALS = 2504
N_SNPS = 200

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp3")

CONFIGS = [
    {
        "name": "A",
        "label": "Baseline (L=2, scale=2^40)",
        "depth": 2,
        "coeff_mod_bit_sizes": [60, 40, 40, 60],
        "scale_bits": 40,
    },
    {
        "name": "B",
        "label": "Depth reduction only (L=1, scale=2^40)",
        "depth": 1,
        "coeff_mod_bit_sizes": [60, 40, 60],  # outer primes must exceed scale bits
        "scale_bits": 40,
    },
    {
        "name": "C",
        "label": "Scale reduction only (L=2, scale=2^21)",
        "depth": 2,
        "coeff_mod_bit_sizes": [60, 21, 21, 60],
        "scale_bits": 21,
    },
    {
        "name": "D",
        "label": "Compacted (L=1, scale=2^21)",
        "depth": 1,
        "coeff_mod_bit_sizes": [40, 21, 40],
        "scale_bits": 21,
    },
]


def benchmark_config(cfg: dict, data: np.ndarray, weights: np.ndarray, pt_results: np.ndarray) -> dict:
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=cfg["coeff_mod_bit_sizes"],
    )
    context.global_scale = 2 ** cfg["scale_bits"]
    context.generate_galois_keys()

    enc_start = time.time()
    enc_rows = [ts.ckks_vector(context, row.tolist()) for row in data]
    enc_time = time.time() - enc_start

    comp_start = time.time()
    enc_scores = []
    for row in tqdm(enc_rows, desc=f"  Config {cfg['name']}"):
        enc_scores.append(row.dot(weights.tolist()))
    comp_time = time.time() - comp_start

    dec_results = [s.decrypt()[0] for s in enc_scores]
    mae = float(np.mean(np.abs(pt_results - dec_results)))
    ct_size_kb = len(enc_rows[0].serialize()) / 1024

    return {
        "Config": cfg["name"],
        "Description": cfg["label"],
        "Depth (L)": cfg["depth"],
        "Scale (bits)": cfg["scale_bits"],
        "Enc Time (s)": round(enc_time, 4),
        "Comp Time (s)": round(comp_time, 4),
        "Total FHE Time (s)": round(enc_time + comp_time, 4),
        "Ciphertext Size (KB)": round(ct_size_kb, 2),
        "MAE": round(mae, 10),
    }


def run_factorial():
    print(f"--- 2x2 Factorial: {{L=1,L=2}} x {{scale=2^21,2^40}}  (N={N_INDIVIDUALS}, K={N_SNPS}) ---")
    rng = np.random.default_rng(42)
    data = rng.integers(0, 3, size=(N_INDIVIDUALS, N_SNPS))
    weights = rng.uniform(0.001, 0.05, size=(N_SNPS,))
    pt_results = data.dot(weights)

    results = []
    for cfg in CONFIGS:
        print(f"\nRunning Config {cfg['name']}: {cfg['label']}")
        r = benchmark_config(cfg, data, weights, pt_results)
        results.append(r)
        print(f"  Comp Time: {r['Comp Time (s)']:.2f}s  |  MAE: {r['MAE']:.6f}  |  CT size: {r['Ciphertext Size (KB)']:.1f} KB")

    r = {row["Config"]: row for row in results}
    baseline_comp = r["A"]["Comp Time (s)"]

    depth_effect = r["B"]["Comp Time (s)"] - r["A"]["Comp Time (s)"]  # depth: A->B
    scale_effect = r["C"]["Comp Time (s)"] - r["A"]["Comp Time (s)"]  
    interaction = (r["D"]["Comp Time (s)"] - r["A"]["Comp Time (s)"]) - depth_effect - scale_effect
    total_gain = r["D"]["Comp Time (s)"] - r["A"]["Comp Time (s)"]

    mae_depth_effect = r["B"]["MAE"] - r["A"]["MAE"]
    mae_scale_effect = r["C"]["MAE"] - r["A"]["MAE"]
    mae_interaction = (r["D"]["MAE"] - r["A"]["MAE"]) - mae_depth_effect - mae_scale_effect

    print("\n" + "=" * 65)
    print("FACTORIAL RESULTS")
    print("-" * 65)
    print(f"{'Config':<6}  {'Desc':<38}  {'Comp(s)':>7}  {'MAE':>12}")
    for row in results:
        print(f"{row['Config']:<6}  {row['Description']:<38}  {row['Comp Time (s)']:>7.2f}  {row['MAE']:>12.8f}")

    print("\nMARGINAL ATTRIBUTION (computation time, seconds):")
    print(f"  Depth reduction (A→B):   {depth_effect:+.2f}s  ({100*depth_effect/baseline_comp:+.1f}%)")
    print(f"  Scale reduction (A→C):   {scale_effect:+.2f}s  ({100*scale_effect/baseline_comp:+.1f}%)")
    print(f"  Interaction (B+C - A→D): {interaction:+.2f}s  ({100*interaction/baseline_comp:+.1f}%)")
    print(f"  Total (A→D):             {total_gain:+.2f}s  ({100*total_gain/baseline_comp:+.1f}%)")
    print("\nMARGINAL ATTRIBUTION (MAE):")
    print(f"  Depth reduction (A→B):   {mae_depth_effect:+.6f}")
    print(f"  Scale reduction (A→C):   {mae_scale_effect:+.6f}")
    print(f"  Interaction (B+C - A→D): {mae_interaction:+.6f}")
    print("=" * 65)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "factorial_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    marginals_path = os.path.join(RESULTS_DIR, "factorial_marginals.csv")
    with open(marginals_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Effect", "Comp Time Delta (s)", "Comp Time Delta (%)", "MAE Delta"])
        writer.writerow(["Depth (A->B)", round(depth_effect, 4), round(100*depth_effect/baseline_comp, 2), round(mae_depth_effect, 8)])
        writer.writerow(["Scale (A->C)", round(scale_effect, 4), round(100*scale_effect/baseline_comp, 2), round(mae_scale_effect, 8)])
        writer.writerow(["Interaction", round(interaction, 4), round(100*interaction/baseline_comp, 2), round(mae_interaction, 8)])
        writer.writerow(["Total (A->D)", round(total_gain, 4), round(100*total_gain/baseline_comp, 2), round(r["D"]["MAE"] - r["A"]["MAE"], 8)])

    print(f"\nResults saved: {csv_path}")
    print(f"Marginals saved: {marginals_path}")


if __name__ == "__main__":
    run_factorial()
