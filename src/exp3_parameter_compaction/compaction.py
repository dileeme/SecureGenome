import os
import csv
import time
import numpy as np
import tenseal as ts
from tqdm import tqdm

N_INDIVIDUALS = 2504
N_SNPS = 200

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp3")


def experiment_compaction():
    print(f"--- [COMPACTION] Parameter-Compacted CKKS (N={N_INDIVIDUALS}, K={N_SNPS}) ---")
    data = np.random.randint(0, 3, size=(N_INDIVIDUALS, N_SNPS))
    weights = np.random.uniform(0.001, 0.05, size=(N_SNPS,))

    pt_results = data.dot(weights)

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[40, 21, 40],
    )
    context.global_scale = 2**21
    context.generate_galois_keys()

    enc_start = time.time()
    enc_rows = [ts.ckks_vector(context, row.tolist()) for row in data]
    enc_time = time.time() - enc_start

    comp_start = time.time()
    enc_scores = []
    for row in tqdm(enc_rows, desc="Processing Compacted PRS"):
        enc_scores.append(row.dot(weights.tolist()))
    comp_time = time.time() - comp_start

    dec_results = [s.decrypt()[0] for s in enc_scores]
    mae = float(np.mean(np.abs(pt_results - dec_results)))
    serialized_size_kb = len(enc_rows[0].serialize()) / 1024

    print("\n" + "=" * 45)
    print(f"COMPACTION RESULTS  (L=1, scale=2^21)")
    print("-" * 45)
    print(f"Encryption Time (Total):  {enc_time:>8.2f} s")
    print(f"Computation Time (Total): {comp_time:>8.2f} s")
    print(f"Ciphertext Size:          {serialized_size_kb:>8.1f} KB")
    print(f"MAE:                      {mae:>8.10f}")
    print(f"Target Latency Met:       {'YES' if comp_time < 70 else 'NO'}")
    print("=" * 45)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "tuned_benchmark_results.csv")
    rows = [
        ("N Individuals", N_INDIVIDUALS),
        ("N SNPs", N_SNPS),
        ("Config", "D: L=1, scale=2^21"),
        ("Encryption Time (Total) (s)", round(enc_time, 4)),
        ("Computation Time (Total) (s)", round(comp_time, 4)),
        ("Ciphertext Size (KB)", round(serialized_size_kb, 2)),
        ("MAE", round(mae, 10)),
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerows(rows)
    print(f"\nResults saved: {csv_path}")


if __name__ == "__main__":
    experiment_compaction()
