import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from tqdm import tqdm  # Progress bar library

# 1. Load and Pre-process
print("Loading genomic matrix...")
df = pd.read_csv('data/genotype_matrix.tsv', sep='\t', header=None).dropna()
X_all = df.T.replace({
    '0|0': 0, '0|1': 1, '1|0': 1, '1|1': 2, 
    '0/0': 0, '0/1': 1, '1/0': 1, '1/1': 2
}).values

# Labels: 1 = Study Group, 0 = Population Control [cite: 26]
y = np.array([1]*(X_all.shape[0]//2) + [0]*(X_all.shape[0] - X_all.shape[0]//2))

results = []
snp_sweep = [5, 10, 20, 50, 100, 200]

plt.figure(figsize=(10, 6))

print("Starting MAF Re-Identification Sweep...")
# Wrap the loop with tqdm for a visual progress bar
for k in tqdm(snp_sweep, desc="Processing SNP Features"):
    X = X_all[:, :k]
    clf = LogisticRegression(max_iter=1000)
    
    # Calculate Mean AUC [cite: 27, 28]
    mean_auc = np.mean(cross_val_score(clf, X, y, cv=5, scoring='roc_auc'))
    results.append({"k (SNPs)": k, "Mean AUC": round(mean_auc, 4)})
    
    # Generate curve for visualization
    clf.fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, probs)
    plt.plot(fpr, tpr, label=f'k={k} (AUC = {mean_auc:.2f})')

# 2. Plotting the results
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Genomic Re-Identification Attack: AUC vs. SNP Count')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('reid_auc_curves.png')
print("\nPlot saved as reid_auc_curves.png")

# 3. Final Results Table
print("\n" + "="*30)
print(f"{'SNP Count (k)':<15} | {'Mean AUC':<10}")
print("-" * 30)
for res in results:
    print(f"{res['k (SNPs)']:<15} | {res['Mean AUC']:<10}")
print("="*30)
