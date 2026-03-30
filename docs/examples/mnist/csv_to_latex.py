import csv
from collections import defaultdict

CSV_PATH = "docs/examples/mnist/results.csv"

# Read CSV
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader if r["method"]]  # skip empty rows

# Collect unique methods and num_compare values (preserve order)
methods = list(dict.fromkeys(r["method"] for r in rows))
num_compares = sorted(set(int(r["num_compare"]) for r in rows))

# Build lookup: (method, num_compare) -> (test_mse, test_spearman)
data = {}
for r in rows:
    key = (r["method"], int(r["num_compare"]))
    data[key] = (float(r["test_mse"]), float(r["test_spearman"]))

# Generate LaTeX
n_cols = len(num_compares)
col_spec = "l" + "c" * n_cols
header_cols = " & ".join(f"n={n}" for n in num_compares)

print(r"\begin{table}[ht]")
print(r"\centering")
print(rf"\begin{{tabular}}{{{col_spec}}}")
print(r"\toprule")
print(rf" & {header_cols} \\")
print(r"\midrule")

for method in methods:
    cells = []
    for nc in num_compares:
        if (method, nc) in data:
            mse, spear = data[(method, nc)]
            cells.append(f"{mse*1e-4:.1f} ({spear*100:.1f})")
        else:
            cells.append("--")
    row_label = method.replace("_", r"\_")
    print(f"{row_label} & {' & '.join(cells)} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\caption{MNIST quantile: test MSE (Spearman correlation [\%] in brackets).}")
print(r"\label{tab:mnist_results}")
print(r"\end{table}")
