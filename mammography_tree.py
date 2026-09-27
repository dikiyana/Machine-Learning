import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import LeaveOneOut, cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
data = [
["P-2001",45,"High","Round","Circumscribed",3,"Benign"],
["P-2002",62,"Low","Irregular","Spiculated",5,"Malignant"],
["P-2003",55,"Medium","Lobular","Ill-defined",4,"Malignant"],
["P-2004",39,"Extreme","Oval","Circumscribed",2,"Benign"],
["P-2005",68,"High","Irregular","Obscured",4,"Malignant"],
["P-2006",51,"Medium","Round","Circumscribed",3,"Benign"],
["P-2007",74,"Low","Irregular","Spiculated",5,"Malignant"],
["P-2008",42,"High","Oval","Microlobulated",4,"Benign"],
["P-2009",58,"Medium","Lobular","Spiculated",5,"Malignant"],
["P-2010",47,"Extreme","Round","Circumscribed",2,"Benign"],
["P-2011",61,"Low","Oval","Obscured",3,"Benign"],
["P-2012",53,"Medium","Irregular","Ill-defined",4,"Malignant"],
["P-2013",49,"High","Lobular","Microlobulated",4,"Benign"],
["P-2014",65,"Low","Irregular","Spiculated",5,"Malignant"],
["P-2015",44,"Extreme","Oval","Circumscribed",3,"Benign"],
]
cols = ["Patient_ID","Age","Breast_Density","Mass_Shape","Mass_Margin","BI_RADS_Category","Diagnosis"]
df = pd.DataFrame(data, columns=cols)
print(df, "\n")

# ---------------------------------------------------------------------------
# 2. Entropy target
# ---------------------------------------------------------------------------
def entropy(counts):
    counts = np.array(counts, dtype=float)
    p = counts / counts.sum()
    p = p[p > 0]
    return -(p * np.log2(p)).sum()

def gini(counts):
    counts = np.array(counts, dtype=float)
    p = counts / counts.sum()
    return 1 - (p ** 2).sum()

vc = df["Diagnosis"].value_counts()
print("Distribusi target:", vc.to_dict())
n_benign, n_malignant = vc["Benign"], vc["Malignant"]
H_root = entropy([n_benign, n_malignant])
G_root = gini([n_benign, n_malignant])
print(f"Entropy(target) = {H_root:.4f}")
print(f"Gini(target)    = {G_root:.4f}\n")

# ---------------------------------------------------------------------------
# 3. Split test: Mass_Shape == 'Irregular'
# ---------------------------------------------------------------------------
mask_irr = df["Mass_Shape"] == "Irregular"
grp_irr = df[mask_irr]["Diagnosis"].value_counts()
grp_not = df[~mask_irr]["Diagnosis"].value_counts()
print("Mass_Shape == 'Irregular':")
print("  Irregular       :", grp_irr.to_dict())
print("  Bukan Irregular :", grp_not.to_dict())

H_irr = entropy([grp_irr.get("Benign",0), grp_irr.get("Malignant",0)])
H_not = entropy([grp_not.get("Benign",0), grp_not.get("Malignant",0)])
n_irr, n_not = mask_irr.sum(), (~mask_irr).sum()
n_total = len(df)
H_weighted = (n_irr/n_total)*H_irr + (n_not/n_total)*H_not
IG = H_root - H_weighted
print(f"  Entropy(Irregular)      = {H_irr:.4f}  (n={n_irr})")
print(f"  Entropy(Bukan Irregular)= {H_not:.4f}  (n={n_not})")
print(f"  Entropy tertimbang anak = {H_weighted:.4f}")
print(f"  Information Gain        = {IG:.4f}\n")

# ---------------------------------------------------------------------------
# 4. Split test: Age <= 50 -> Gini
# ---------------------------------------------------------------------------
mask_age = df["Age"] <= 50
grp_a1 = df[mask_age]["Diagnosis"].value_counts()
grp_a2 = df[~mask_age]["Diagnosis"].value_counts()
print("Age <= 50:")
print("  Age<=50 :", grp_a1.to_dict())
print("  Age>50  :", grp_a2.to_dict())
G_a1 = gini([grp_a1.get("Benign",0), grp_a1.get("Malignant",0)])
G_a2 = gini([grp_a2.get("Benign",0), grp_a2.get("Malignant",0)])
n_a1, n_a2 = mask_age.sum(), (~mask_age).sum()
G_weighted = (n_a1/n_total)*G_a1 + (n_a2/n_total)*G_a2
print(f"  Gini(Age<=50) = {G_a1:.4f}  (n={n_a1})")
print(f"  Gini(Age>50)  = {G_a2:.4f}  (n={n_a2})")
print(f"  Gini tertimbang (split)= {G_weighted:.4f}")
print(f"  Gini gain               = {G_root - G_weighted:.4f}\n")

# ---------------------------------------------------------------------------
# 5. Decision Tree Classifier dengan scikit-learn
# ---------------------------------------------------------------------------
X_cat = df[["Breast_Density","Mass_Shape","Mass_Margin"]]
ohe = OneHotEncoder(sparse_output=False)
X_cat_enc = ohe.fit_transform(X_cat)
cat_feature_names = ohe.get_feature_names_out(["Breast_Density","Mass_Shape","Mass_Margin"])

X = np.hstack([df[["Age","BI_RADS_Category"]].values, X_cat_enc])
feature_names = ["Age","BI_RADS_Category"] + list(cat_feature_names)
y = (df["Diagnosis"] == "Malignant").astype(int).values  # 1=Malignant, 0=Benign

print("Jumlah fitur setelah encoding:", len(feature_names))
print("Fitur:", feature_names, "\n")

# --- 5a. Fit tree tanpa batasan untuk melihat bentuk "alami" ---
clf_full = DecisionTreeClassifier(criterion="entropy", random_state=0)
clf_full.fit(X, y)
print(f"[Tree tanpa batasan] depth={clf_full.get_depth()}, n_leaves={clf_full.get_n_leaves()}")

# --- 5b. Bandingkan beberapa kedalaman/pruning pakai Leave-One-Out CV ---
print("\nPerbandingan performa (Leave-One-Out CV) untuk memilih 'bentuk' tree terbaik:")
loo = LeaveOneOut()
results = []
for max_depth in [1, 2, 3, None]:
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=0)
    y_pred_loo = cross_val_predict(clf, X, y, cv=loo)
    f1 = f1_score(y, y_pred_loo)
    prec = precision_score(y, y_pred_loo)
    rec = recall_score(y, y_pred_loo)
    acc = (y_pred_loo == y).mean()
    results.append((max_depth, acc, prec, rec, f1))
    print(f"  max_depth={str(max_depth):4}  acc={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}")

# --- 5c. Pilih model final: max_depth=3, evaluasi dgn LOOCV, tampilkan classification_report ---
best_depth = 3
clf_best = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth, random_state=0)
y_pred_best = cross_val_predict(clf_best, X, y, cv=loo)
print(f"\n=== Evaluasi model terpilih (max_depth={best_depth}, criterion='entropy'), via LOOCV ===")
print(confusion_matrix(y, y_pred_best))
print(classification_report(y, y_pred_best, target_names=["Benign","Malignant"]))

# fit ulang di seluruh data untuk lihat struktur tree final yang dipakai
clf_best.fit(X, y)
plt.figure(figsize=(18, 8))
plot_tree(clf_best, feature_names=feature_names, class_names=["Benign","Malignant"],
          filled=True, rounded=True, impurity=True, fontsize=9)
plt.title(f"Decision Tree (criterion='entropy', max_depth={best_depth})")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/mammography_decision_tree.png", dpi=150)
print("\nGambar tree tersimpan: mammography_decision_tree.png")

print("\nFeature importances:")
for f, imp in sorted(zip(feature_names, clf_best.feature_importances_), key=lambda x: -x[1]):
    if imp > 0:
        print(f"  {f}: {imp:.4f}")
