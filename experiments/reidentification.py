import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from tqdm import tqdm


print("Loading genomic matrix (first 500 SNPs)...")
df = pd.read_csv('data/genotype_matrix.tsv', sep='\t', header=None, nrows=500)

print("Cleaning and Transposing...")
df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
X_all = df.T

X_all = X_all.replace({
    '0|0': 0, '0|1': 1, '1|0': 1, '1|1': 2,
    '0/0': 0, '0/1': 1, '1/0': 1, '1/1': 2
})

X_all = X_all.apply(pd.to_numeric, errors='coerce').fillna(0).values
print(f"Matrix Ready: {X_all.shape[0]} individuals, {X_all.shape[1]} SNPs found.")

signal_snps = X_all[:, 40:60].sum(axis=1)
threshold = np.percentile(signal_snps, 50) 
y = (signal_snps > threshold).astype(int)

results = []
snp_sweep = [5, 10, 20, 50, 100, 200]

plt.figure(figsize=(10, 6))

print("Starting MAF Re-Identification Sweep...")
for k in tqdm(snp_sweep, desc="Processing SNP Features"):
    X = X_all[:, 60:60+k]
    clf = LogisticRegression(max_iter=1000, C=10)
    mean_auc = np.mean(cross_val_score(clf, X, y, cv=5, scoring='roc_auc', n_jobs=-1))
    results.append({"k (SNPs)": k, "Mean AUC": round(mean_auc, 4)})
    clf.fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, probs)
    plt.plot(fpr, tpr, label=f'k={k} (AUC = {mean_auc:.2f})')
    
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Experiment 1: Genomic Re-Identification Attack (AUC vs. SNP Count)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('reid_auc_curves.png')
print("\nPlot saved as 'reid_auc_curves.png'")

print("\n" + "="*35)
print(f"{'SNP Count (k)':<15} | {'Mean AUC':<10}")
print("-" * 35)
for res in results:
    print(f"{res['k (SNPs)']:<15} | {res['Mean AUC']:<10}")
print("="*35)
