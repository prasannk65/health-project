# health_project2.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# --- CONFIG ---
DATA_DIR = "data"
DIAB_PATH = os.path.join(r'C:\Users\prasa\Downloads\archive (2).zip')   # PIMA CSV
HEART_PATH = os.path.join(r'C:\Users\prasa\Downloads\archive (1).zip')     # Heart CSV
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def load_datasets():
    if not os.path.exists(DIAB_PATH) or not os.path.exists(HEART_PATH):
        raise FileNotFoundError("Place diabetes.csv and heart.csv inside the 'data' folder.")
    diabetes = pd.read_csv(DIAB_PATH)
    heart = pd.read_csv(HEART_PATH)
    return diabetes, heart

def explore(df, name):
    print(f"\n=== {name} ===")
    print("shape:", df.shape)
    print(df.info())
    print(df.describe().T)
    print("null counts:\n", df.isnull().sum())
    print("zero counts per col:\n", (df == 0).sum())

def preprocess_diabetes(df):
    df = df.copy()
    zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    for c in zero_cols:
        if c in df.columns:
            df[c].replace(0, np.nan, inplace=True)
            df[c].fillna(df[c].median(), inplace=True)
    return df

def label_encode_df(df):
    df2 = df.copy()
    for col in df2.select_dtypes(include=['object','category']).columns:
        df2[col] = LabelEncoder().fit_transform(df2[col].astype(str))
    return df2

def standardize(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler

def do_pca_and_plot(X_scaled, y, outname):
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X_scaled)
    plt.figure(figsize=(6,5))
    plt.scatter(Xp[:,0], Xp[:,1], c=y, alpha=0.6, s=20)
    plt.title(outname + " PCA (2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    p = os.path.join("outputs", outname + "_pca2d.png")
    plt.savefig(p, dpi=150)
    plt.close()

def train_models_and_select_best(X, y, prefix):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "logistic": LogisticRegression(max_iter=2000),
        "svm": SVC(probability=True),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "naive_bayes": GaussianNB()
    }
    results = {}
    best_auc = -1
    best_model = None
    best_name = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:,1]
        else:
            proba = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {"model": model, "auc": roc_auc, "cm": cm, "report": classification_report(y_test, y_pred, output_dict=True)}
        print(f"{prefix}-{name}: AUC={roc_auc:.3f}")
        # save roc plot per model
        plt.figure()
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
        plt.plot([0,1],[0,1],'--', color='grey')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC - {prefix} - {name}")
        plt.legend(loc='lower right')
        plt.savefig(os.path.join("outputs", f"{prefix}_{name}_roc.png"))
        plt.close()

        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model = model
            best_name = name

    # Save all models for later inspection
    for name, d in results.items():
        pickle.dump(d["model"], open(os.path.join("models", f"{prefix}_{name}.pkl"), "wb"))

    print(f"Best for {prefix}: {best_name} with AUC={best_auc:.3f}")
    # return best model and results dict
    return best_model, results

def feature_importance_plot(dt_model, feature_names, outname):
    if not hasattr(dt_model, "feature_importances_"):
        return
    importances = dt_model.feature_importances_
    idx = np.argsort(importances)[::-1]
    names = np.array(feature_names)[idx]
    plt.figure(figsize=(6,4))
    plt.barh(names, importances[idx])
    plt.title(outname + " Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", outname + "_feature_importance.png"))
    plt.close()

def run_all():
    diabetes, heart = load_datasets()

    # 11. Explore
    explore(diabetes, "Diabetes")
    explore(heart, "Heart")

    # 12. Missing / zero handling
    diab = preprocess_diabetes(diabetes)
    heart_df = heart.copy().fillna(heart.median())

    # 13. Label encode
    diab_enc = label_encode_df(diab)
    heart_enc = label_encode_df(heart_df)

    # identify features & targets
    if 'Outcome' not in diab_enc.columns:
        raise ValueError("Diabetes dataset must have 'Outcome' column")
    X_diab = diab_enc.drop('Outcome', axis=1)
    y_diab = diab_enc['Outcome']

    # heart common target names: try 'target' else last binary column
    target_col = 'target' if 'target' in heart_enc.columns else None
    if target_col is None:
        candidates = [c for c in heart_enc.columns if heart_enc[c].nunique() == 2]
        if candidates:
            target_col = candidates[-1]
        else:
            raise ValueError("No binary target found in heart dataset; please check")
    X_heart = heart_enc.drop(target_col, axis=1)
    y_heart = heart_enc[target_col]

    # 14. StandardScaler
    X_diab_scaled, scaler_d = standardize(X_diab)
    X_heart_scaled, scaler_h = standardize(X_heart)
    pickle.dump(scaler_d, open(os.path.join("models","scaler_diab.pkl"), "wb"))
    pickle.dump(scaler_h, open(os.path.join("models","scaler_heart.pkl"), "wb"))

    # 15. PCA visuals
    do_pca_and_plot(X_diab_scaled, y_diab, "diabetes")
    do_pca_and_plot(X_heart_scaled, y_heart, "heart")

    # 16-17. Train models
    best_diab_model, diab_results = train_models_and_select_best(X_diab_scaled, y_diab, "diab")
    best_heart_model, heart_results = train_models_and_select_best(X_heart_scaled, y_heart, "heart")

    # 18. Save best models (by AUC)
    pickle.dump(best_diab_model, open(os.path.join("models","diab_best_model.pkl"), "wb"))
    pickle.dump(best_heart_model, open(os.path.join("models","heart_best_model.pkl"), "wb"))

    # Save evaluation summary csv
    rows = []
    for k,v in diab_results.items():
        rows.append({"dataset":"diab","model":k,"auc":v["auc"]})
    for k,v in heart_results.items():
        rows.append({"dataset":"heart","model":k,"auc":v["auc"]})
    pd.DataFrame(rows).to_csv(os.path.join("outputs","model_summary.csv"), index=False)

    # 19. Feature importance using decision tree saved earlier
    # if we trained decision_tree and saved it as diab_decision_tree.pkl etc
    try:
        dt_d = pickle.load(open(os.path.join("models","diab_decision_tree.pkl"), "rb"))
        feature_importance_plot(dt_d, X_diab.columns, "diabetes")
    except Exception:
        pass
    try:
        dt_h = pickle.load(open(os.path.join("models","heart_decision_tree.pkl"), "rb"))
        feature_importance_plot(dt_h, X_heart.columns, "heart")
    except Exception:
        pass

    print("\n✅ Training & evaluation complete. Files saved in models/ and outputs/")

# Utility used by Flask when .pkl missing
def get_models_and_scalers_from_run():
    """
    If models/diab_best_model.pkl etc exist they will be loaded.
    Otherwise run training and return models+scalers.
    """
    try:
        diab_model = pickle.load(open(os.path.join("models","diab_best_model.pkl"), "rb"))
        heart_model = pickle.load(open(os.path.join("models","heart_best_model.pkl"), "rb"))
        scaler_diab = pickle.load(open(os.path.join("models","scaler_diab.pkl"), "rb"))
        scaler_heart = pickle.load(open(os.path.join("models","scaler_heart.pkl"), "rb"))
        return diab_model, heart_model, scaler_diab, scaler_heart
    except Exception:
        run_all()
        diab_model = pickle.load(open(os.path.join("models","diab_best_model.pkl"), "rb"))
        heart_model = pickle.load(open(os.path.join("models","heart_best_model.pkl"), "rb"))
        scaler_diab = pickle.load(open(os.path.join("models","scaler_diab.pkl"), "rb"))
        scaler_heart = pickle.load(open(os.path.join("models","scaler_heart.pkl"), "rb"))
        return diab_model, heart_model, scaler_diab, scaler_heart

if __name__ == "__main__":
    # run training if executed directly
    run_all()
