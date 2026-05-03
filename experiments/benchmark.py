import tenseal as ts
import time
import numpy as np
from tqdm import tqdm
import sys

def run_full_scale_study():
    n_individuals = 2500
    n_snps = 200
    print(f"--- Scaling Study: {n_individuals} Samples x {n_snps} SNPs ---")

    data = np.random.randint(0, 3, size=(n_individuals, n_snps))
    weights = np.random.uniform(0.001, 0.05, size=(n_snps,))

    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    context.generate_galois_keys()

    start_pt = time.time()
    pt_results = data.dot(weights)
    end_pt = time.time()
    pt_total_time = end_pt - start_pt

    en_start = time.time()
    encrypted_rows = []
    for row in tqdm(data.tolist(), desc="Encrypting Patient Data"):
        encrypted_rows.append(ts.ckks_vector(context, row))
    en_end = time.time()
    total_enc_time = en_end - en_start

    comp_start = time.time()
    encrypted_scores = []
    for enc_row in tqdm(encrypted_rows, desc="Computing Encrypted PRS"):
        encrypted_scores.append(enc_row.dot(weights.tolist()))
    comp_end = time.time()
    total_comp_time = comp_end - comp_start

    dec_results = [score.decrypt()[0] for score in encrypted_scores]
    mae = np.mean(np.abs(pt_results - dec_results))

    print("\n" + "="*50)
    print("FINAL RESEARCH RESULTS: N=2500")
    print("-" * 50)
    print(f"Plaintext Processing:   {pt_total_time*1000:,.2f} ms")
    print(f"Total Encryption Time:  {total_enc_time:,.2f} s")
    print(f"Total FHE Computation:  {total_comp_time:,.2f} s")
    print(f"Accuracy (MAE):         {mae:.10f}")

    overhead = total_comp_time / pt_total_time
    print(f"Performance Overhead:   {overhead:,.0f}x slowest")
    print("="*50)

if __name__ == "__main__":
    run_full_scale_study()
~                                          
