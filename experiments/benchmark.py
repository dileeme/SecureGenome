import tenseal as ts
import time
import numpy as np
from tqdm import tqdm
import sys

def run_full_scale_study():
    # 1. Dataset Setup
    n_individuals = 2500
    n_snps = 200
    print(f"--- Scaling Study: {n_individuals} Samples x {n_snps} SNPs ---")

    # Simulating your genotype matrix (0, 1, 2)
    data = np.random.randint(0, 3, size=(n_individuals, n_snps))
    # Simulating PGS weights (e.g., from your Bioinformatics research)
    weights = np.random.uniform(0.001, 0.05, size=(n_snps,))

    # 2. Context Setup (CKKS for Polygenic Risk Scores)
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    context.generate_galois_keys()

    # 3. Plaintext Baseline
    start_pt = time.time()
    pt_results = data.dot(weights)
    end_pt = time.time()
    pt_total_time = end_pt - start_pt

    # 4. Encrypted Pipeline (The "Defense")
    # Step A: Encryption (Client-side)
    en_start = time.time()
    encrypted_rows = []
    for row in tqdm(data.tolist(), desc="Encrypting Patient Data"):
        encrypted_rows.append(ts.ckks_vector(context, row))
    en_end = time.time()
    total_enc_time = en_end - en_start

    # Step B: Computation (Server-side)
    comp_start = time.time()
    encrypted_scores = []
    for enc_row in tqdm(encrypted_rows, desc="Computing Encrypted PRS"):
        # Dot product: Multiply patient SNPs by Plaintext weights and sum
        encrypted_scores.append(enc_row.dot(weights.tolist()))
    comp_end = time.time()
    total_comp_time = comp_end - comp_start

    # Step C: Decryption & Accuracy Check
    dec_results = [score.decrypt()[0] for score in encrypted_scores]
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(pt_results - dec_results))

    # 5. Final Report
    print("\n" + "="*50)
    print("FINAL RESEARCH RESULTS: N=2500")
    print("-" * 50)
    print(f"Plaintext Processing:   {pt_total_time*1000:,.2f} ms")
    print(f"Total Encryption Time:  {total_enc_time:,.2f} s")
    print(f"Total FHE Computation:  {total_comp_time:,.2f} s")
    print(f"Accuracy (MAE):         {mae:.10f}")

    # Calculate Overhead
    overhead = total_comp_time / pt_total_time
    print(f"Performance Overhead:   {overhead:,.0f}x slowest")
    print("="*50)

if __name__ == "__main__":
    run_full_scale_study()
~                                          
