import pandas as pd
import numpy as np


reid_df = pd.read_csv('reidentification_results.csv')
bench_df = pd.read_csv('benchmark_results.csv')
tuned_bench_df = pd.read_csv('tuned_benchmark_results.csv')


orig_time = float(bench_df.loc[bench_df['Metric'] == 'Total FHE Computation (s)', 'Value'].values[0])
tuned_time = float(tuned_bench_df.loc[tuned_bench_df['Metric'] == 'Computation Time (Total) (s)', 'Value'].values[0])
utility_mae = float(bench_df.loc[bench_df['Metric'] == 'Accuracy (MAE)', 'Value'].values[0])
reid_df['Risk_Velocity'] = reid_df['Mean AUC'].diff() / reid_df['k (SNPs)'].diff()
reid_df['Risk_Acceleration'] = reid_df['Risk_Velocity'].diff() / reid_df['k (SNPs)'].diff()

cue_expanded = reid_df.copy()
cue_expanded['Tuned_Latency_Sec'] = tuned_time
cue_expanded['Latency_Per_SNP_ms'] = (tuned_time / cue_expanded['k (SNPs)']) * 1000
cue_expanded['Efficiency_Gain_Pct'] = round(((orig_time - tuned_time) / orig_time) * 100, 4)
cue_expanded['Utility_MAE'] = utility_mae

conditions = [
    (cue_expanded['Mean AUC'] < 0.70),
    (cue_expanded['Mean AUC'] >= 0.70) & (cue_expanded['Mean AUC'] < 0.90),
    (cue_expanded['Mean AUC'] >= 0.90)
]

tiers = ['SECURE', 'VULNERABLE', 'CRITICAL_RISK']
cue_expanded['Security_Tier'] = np.select(conditions, tiers, default='UNKNOWN')
cue_expanded = cue_expanded.fillna(0)

final_cols = [
    'k (SNPs)', 'Mean AUC', 'Security_Tier', 'Risk_Velocity', 
    'Risk_Acceleration', 'Tuned_Latency_Sec', 'Latency_Per_SNP_ms',
    'Efficiency_Gain_Pct', 'Utility_MAE'
]

cue_expanded = cue_expanded[final_cols]
cue_expanded.to_csv('cue_results_expanded.csv', index=False)
print("Expansive analysis complete. File 'cue_results_expanded.csv' generated.")