import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadstat
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.stats import spearmanr, norm
from io import BytesIO

st.set_page_config(page_title="ROC AUC & Correlation Heatmap Dashboard", layout="wide")
st.title('🔬 ROC AUC & Correlation Heatmap Dashboard (.csv, .txt, .sav)')

uploaded_file = st.file_uploader("Upload CSV, TXT, or SPSS (.sav)", type=["csv", "txt", "sav"])

if uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1]

    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file, sep=';', encoding='ISO-8859-9')
    elif file_extension == 'txt':
        df = pd.read_csv(uploaded_file, sep=';', encoding='ISO-8859-9')
    elif file_extension == 'sav':
        with open("temp.sav", "wb") as f:
            f.write(uploaded_file.read())
        df, meta = pyreadstat.read_sav("temp.sav")

    st.write('Data Preview:', df.head())

    st.sidebar.header("Global Plot Options")
    palette_choice = st.sidebar.selectbox(
        "Heatmap Color Palette",
        ["coolwarm", "vlag", "rocket", "mako", "icefire"]
    )

    st.sidebar.header("Select Analysis")
    analysis_type = st.sidebar.radio("Choose Analysis",
                                     ["Correlation Heatmap", "Single ROC Curve", "Multiple ROC Curves"])

    # Gerekli değişkenleri global yapalım
    y_true, y_scores = None, None
    best_threshold = None
    best_sensitivity = None
    best_specificity = None
    roc_auc = None
    custom_name = ""

    tab1, tab2, tab3 = st.tabs(["🔬 Analysis", "📋 ROC Table", "🧠 About"])

    with tab1:
        if analysis_type == "Correlation Heatmap":
            correlation_vars = st.sidebar.multiselect(
                "Select variables for Correlation Matrix (numeric)",
                options=df.columns,
                default=df.select_dtypes(include=[np.number]).columns.tolist()
            )

            if len(correlation_vars) < 2:
                st.warning("Select at least 2 numeric variables.")
                st.stop()

            heatmap_title = st.sidebar.text_input("Heatmap Title", value="Spearman Correlation Heatmap")
            custom_names = {}
            for col in correlation_vars:
                new_name = st.sidebar.text_input(f"Rename '{col}'", value=col)
                custom_names[col] = new_name

            footnote = st.text_area("Add footnote below the plot", value="")

            df_corr = df[correlation_vars].apply(pd.to_numeric, errors='coerce').dropna()
            df_corr.rename(columns=custom_names, inplace=True)

            corr, _ = spearmanr(df_corr)
            corr_df = pd.DataFrame(corr, index=df_corr.columns, columns=df_corr.columns)

            mask = np.triu(np.ones_like(corr_df, dtype=bool))

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                corr_df, mask=mask, cmap=palette_choice, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .75}, ax=ax
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_aspect('equal', adjustable='box')
            fig.tight_layout()
            plt.title(heatmap_title)
            st.pyplot(fig)

            if footnote:
                st.markdown(f"**Note:** {footnote}")

        elif analysis_type == "Single ROC Curve":
            outcome_var = st.sidebar.selectbox("Select Outcome Variable (0/1)", options=df.columns)
            predictor_var = st.sidebar.selectbox("Select Predictor Variable (numeric)", options=df.columns)

            plot_title = st.sidebar.text_input("ROC Title", "ROC Curve")
            x_label = st.sidebar.text_input("X-axis Label", "100-Specificity")
            y_label = st.sidebar.text_input("Y-axis Label", "Sensitivity")
            custom_name = st.sidebar.text_input(f"Rename '{predictor_var}'", value=predictor_var)
            show_ci = st.sidebar.checkbox("Show 95% Confidence Interval", value=True)
            footnote = st.text_area("Add footnote below the plot", value="")

            y_true = pd.to_numeric(df[outcome_var], errors='coerce')
            y_scores = pd.to_numeric(df[predictor_var], errors='coerce')
            mask = ~y_true.isna() & ~y_scores.isna()
            y_true = y_true[mask].astype(int)
            y_scores = y_scores[mask].astype(float)
            y_true = y_true.replace({2: 0, 1: 1})

            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            youden_index = tpr - fpr
            best_index = np.argmax(youden_index)
            best_threshold = thresholds[best_index]
            best_sensitivity = tpr[best_index] * 100
            best_specificity = (1 - fpr[best_index]) * 100

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(fpr * 100, tpr * 100, lw=2, label=f'{custom_name} (AUC = {roc_auc:.3f})', color='purple', marker='s', markevery=5)

            if show_ci:
                bootstraps = 1000
                tpr_boots = []
                rng = np.random.default_rng(seed=42)
                for _ in range(bootstraps):
                    indices = rng.choice(len(y_true), size=len(y_true), replace=True)
                    if len(np.unique(y_true[indices])) < 2:
                        continue
                    fpr_b, tpr_b, _ = roc_curve(y_true[indices], y_scores[indices])
                    tpr_interp = np.interp(np.linspace(0, 1, 100), fpr_b, tpr_b)
                    tpr_boots.append(tpr_interp)

                tpr_boots = np.array(tpr_boots)
                tpr_lower = np.percentile(tpr_boots, 2.5, axis=0)
                tpr_upper = np.percentile(tpr_boots, 97.5, axis=0)

                ax.plot(np.linspace(0, 100, 100), tpr_lower * 100, linestyle='--', color='gray')
                ax.plot(np.linspace(0, 100, 100), tpr_upper * 100, linestyle='--', color='gray')

            ax.plot([0, 100], [0, 100], color='black', linestyle='--')
            ax.set_xlim([0.0, 100.0])
            ax.set_ylim([0.0, 100.0])
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(plot_title)
            ax.legend(loc="lower right")

            info_text = f"Sensitivity: {best_sensitivity:.1f}\nSpecificity: {best_specificity:.1f}\nCriterion: <= {best_threshold:.3f}"
            ax.text(60, 15, info_text, fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="navy"))

            st.pyplot(fig)

            if footnote:
                st.markdown(f"**Note:** {footnote}")

        elif analysis_type == "Multiple ROC Curves":
            st.warning("ROC table generation is only supported for Single ROC Curve.")

    with tab2:
        if analysis_type == "Single ROC Curve" and y_true is not None:
            y_pred = (y_scores <= best_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            ppv = 100 * tp / (tp + fp) if (tp + fp) != 0 else 0
            npv = 100 * tn / (tn + fn) if (tn + fn) != 0 else 0

            aucs = []
            for _ in range(1000):
                indices = np.random.choice(len(y_true), len(y_true), replace=True)
                if len(np.unique(y_true[indices])) < 2:
                    continue
                fpr_b, tpr_b, _ = roc_curve(y_true[indices], y_scores[indices])
                aucs.append(auc(fpr_b, tpr_b))
            ci_lower = np.percentile(aucs, 2.5)
            ci_upper = np.percentile(aucs, 97.5)
            auc_mean = np.mean(aucs)
            auc_std = np.std(aucs)
            z = (auc_mean - 0.5) / (auc_std if auc_std > 0 else 1e-6)
            p_value = 2 * (1 - norm.cdf(abs(z)))

            table_df = pd.DataFrame({
                "Marker": [custom_name],
                "Cut-off": [round(best_threshold, 3)],
                "AUC (95% CI)": [f"{roc_auc:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"],
                "p": [f"{p_value:.3f}" + ("*" if p_value < 0.05 else "")],
                "Sensitivity": [round(best_sensitivity, 1)],
                "Specificity": [round(best_specificity, 1)],
                "PPV": [round(ppv, 1)],
                "NPV": [round(npv, 1)]
            })

            st.dataframe(table_df)

    with tab3:
        st.markdown("""
        **ROC AUC & Correlation Heatmap Dashboard**

        - ROC curves with cutoff, CI, and AUC calculations
        - Diagnostic performance table with Sensitivity, Specificity, PPV, NPV
        - Developed interactively with support for CSV, TXT, SAV formats

        **Version**: 1.0
        """)