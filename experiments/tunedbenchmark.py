import tenseal as ts
import numpy as np
import time
from tqdm import tqdm


def experiment_compaction():
    n_individuals = 2500
    n_snps = 200
    print(f"--- [COMPACTION] Optimizing for Genomic Depth ---")
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[40, 21, 40])
    context.global_scale = 2**21
    context.generate_galois_keys()

    data = np.random.randint(0, 3, size=(n_individuals, n_snps))
    weights = np.random.uniform(0.001, 0.05, size=(n_snps,))

    enc_start = time.time()
    enc_rows = [ts.ckks_vector(context, row.tolist()) for row in data]
    enc_time = time.time() - enc_start

    comp_start = time.time()
    for row in tqdm(enc_rows, desc="Processing Compacted PRS"):
        _ = row.dot(weights.tolist())
    comp_time = time.time() - comp_start

    serialized_size = len(enc_rows[0].serialize()) / 1024 

    print("\n" + "="*40)
    print(f"OPTIMIZATION RESULTS (Tuned CKKS)")
    print("-" * 40)
    print(f"Encryption Time (Total): {enc_time:.2f} s")
    print(f"Computation Time (Total): {comp_time:.2f} s")
    print(f"Ciphertext Size:         {serialized_size:.1f} KB")
    print(f"Target Latency Met:      {'YES' if comp_time < 70 else 'NO'}")
    print("="*40)

if __name__ == "__main__":
    experiment_compaction()                                       
