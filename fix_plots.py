import json

path = 'FraudJobDetection.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']

# ── Cell 9: Employment Type ─────────────────────────────────────────────────
# Original: sns.barplot with unlimited categories crammed into a tiny figure.
# Fix: show top 10 employment types by fraud rate, proper figure size.
CELL9_NEW = '''\
#plot Fraudulent job with respective to employment type
emp_fraud = (
    dataset1[dataset1["employment_type"] != ""]
    .groupby("employment_type")["fraudulent"]
    .agg(fraud_rate="mean", count="count")
    .query("count >= 5")
    .sort_values("fraud_rate", ascending=True)
    .tail(10)
)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(emp_fraud.index, emp_fraud["fraud_rate"], color="steelblue")
ax.set_xlabel("Fraud Rate (0 = Real, 1 = Fraudulent)")
ax.set_ylabel("Employment Type")
ax.set_title("Fraud Rate by Employment Type (Top 10)")
plt.tight_layout()
plt.show()
plt.close()'''

# ── Cell 10: Required Experience ────────────────────────────────────────────
# Original: hundreds of unique free-text strings → completely unreadable.
# Fix: top 15 experience categories by fraud rate, tall figure.
CELL10_NEW = '''\
#plot Fraudulent job with respective to required experience
exp_fraud = (
    dataset1[dataset1["required_experience"] != ""]
    .groupby("required_experience")["fraudulent"]
    .agg(fraud_rate="mean", count="count")
    .query("count >= 5")
    .sort_values("fraud_rate", ascending=True)
    .tail(15)
)
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(exp_fraud.index, exp_fraud["fraud_rate"], color="coral")
ax.set_xlabel("Fraud Rate (0 = Real, 1 = Fraudulent)")
ax.set_ylabel("Required Experience")
ax.set_title("Fraud Rate by Required Experience (Top 15)")
plt.tight_layout()
plt.show()
plt.close()'''

replaced = 0
for cell in code_cells:
    src = ''.join(cell['source'])
    if 'employment_type' in src and 'required_experience' not in src:
        lines = CELL9_NEW.splitlines()
        cell['source'] = [line + '\n' if i < len(lines) - 1 else line
                          for i, line in enumerate(lines)]
        cell['outputs'] = []
        cell['execution_count'] = None
        replaced += 1
        print('Fixed Cell 9 (employment type)')
    elif 'required_experience' in src:
        lines = CELL10_NEW.splitlines()
        cell['source'] = [line + '\n' if i < len(lines) - 1 else line
                          for i, line in enumerate(lines)]
        cell['outputs'] = []
        cell['execution_count'] = None
        replaced += 1
        print('Fixed Cell 10 (required experience)')

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'\nDone. Fixed {replaced} cells.')
