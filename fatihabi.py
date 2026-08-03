import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadstat
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.stats import spearmanr, pearsonr, kendalltau, norm, mannwhitneyu
from io import BytesIO
import pickle

# Sayfa Ayarları
st.set_page_config(page_title="Dr. Ozdemir Analysis Tool", layout="wide")
st.title('🔬 ROC AUC & Correlation Dashboard (Proje Kayıt Özellikli)')

# --- YARDIMCI FONKSİYONLAR ---

def load_data(uploaded_file):
    """Farklı dosya tiplerini yükler ve session_state'i günceller."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'pkl':
        try:
            saved_state = pickle.load(uploaded_file)
            df = saved_state.pop('data_frame')
            for key, value in saved_state.items():
                st.session_state[key] = value
            st.success("✅ Proje dosyası başarıyla yüklendi! Ayarlarınız geri getirildi.")
            return df
        except Exception as e:
            st.error(f"Proje dosyası açılırken hata oluştu: {e}")
            return None

    elif file_extension == 'csv':
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=';', encoding='ISO-8859-9')
    elif file_extension == 'txt':
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=';', encoding='ISO-8859-9')
    elif file_extension == 'sav':
        with open("temp.sav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        return pyreadstat.read_sav("temp.sav")[0]
    return None

def compute_correlation_matrices(df_data, method="spearman"):
    """Pearson, Spearman veya Kendall korelasyon ve p-değeri matrislerini hesaplar."""
    cols = df_data.columns
    n = len(cols)
    corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    p_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    n_matrix = pd.DataFrame(index=cols, columns=cols, dtype=int)

    for i in range(n):
        for j in range(n):
            col1, col2 = cols[i], cols[j]
            valid_mask = df_data[col1].notna() & df_data[col2].notna()
            sub_df = df_data.loc[valid_mask, [col1, col2]]
            n_samples = len(sub_df)
            n_matrix.iloc[i, j] = n_samples

            if n_samples < 3 or col1 == col2:
                if col1 == col2:
                    corr_matrix.iloc[i, j] = 1.0
                    p_matrix.iloc[i, j] = 0.0
                else:
                    corr_matrix.iloc[i, j] = np.nan
                    p_matrix.iloc[i, j] = np.nan
                continue

            x = sub_df[col1].values
            y = sub_df[col2].values

            if method == "pearson":
                r, p = pearsonr(x, y)
            elif method == "kendall":
                r, p = kendalltau(x, y)
            else: # spearman
                r, p = spearmanr(x, y)

            corr_matrix.iloc[i, j] = r
            p_matrix.iloc[i, j] = p

    return corr_matrix, p_matrix, n_matrix

def generate_spss_corr_table(corr_df, p_df, n_df, custom_names, method_name):
    """SPSS stili katmanlı korelasyon tablosu üretir."""
    rows = []
    cols = corr_df.columns
    
    stat_label = "Pearson r" if method_name == "pearson" else ("Kendall τ" if method_name == "kendall" else "Spearman ρ")

    for col1 in cols:
        name1 = custom_names.get(col1, col1)
        r_row = {"Değişken": name1, "İstatistik": stat_label}
        p_row = {"Değişken": name1, "İstatistik": "Anlamlılık (2-Yönlü p)"}
        n_row = {"Değişken": name1, "İstatistik": "N (Örneklem)"}

        for col2 in cols:
            name2 = custom_names.get(col2, col2)
            r_val = corr_df.loc[col1, col2]
            p_val = p_df.loc[col1, col2]
            n_val = n_df.loc[col1, col2]

            if pd.isna(r_val):
                r_str, p_str = "-", "-"
            else:
                sig_star = ""
                if p_val < 0.001: sig_star = "***"
                elif p_val < 0.01: sig_star = "**"
                elif p_val < 0.05: sig_star = "*"

                r_str = f"{r_val:.3f}{sig_star}"
                p_str = f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001"

            r_row[name2] = r_str
            p_row[name2] = p_str
            n_row[name2] = str(n_val)

        rows.extend([r_row, p_row, n_row])
    
    return pd.DataFrame(rows)

def calculate_auc_ci(auc_val, n_pos, n_neg, alpha=0.05):
    """Hanley & McNeil yöntemi ile AUC için Standart Hata ve %95 Güven Aralığı hesaplar."""
    if auc_val <= 0 or auc_val >= 1 or n_pos == 0 or n_neg == 0:
        return 0.0, auc_val, auc_val
    q1 = auc_val / (2.0 - auc_val)
    q2 = 2.0 * (auc_val ** 2) / (1.0 + auc_val)
    se = np.sqrt((auc_val * (1.0 - auc_val) + (n_pos - 1.0) * (q1 - auc_val**2) + (n_neg - 1.0) * (q2 - auc_val**2)) / (n_pos * n_neg))
    z = norm.ppf(1.0 - alpha / 2.0)
    ci_lower = max(0.0, auc_val - z * se)
    ci_upper = min(1.0, auc_val + z * se)
    return se, ci_lower, ci_upper


# --- ANA KOD ---

st.sidebar.header("📁 Veri Yükleme")
uploaded_file = st.sidebar.file_uploader(
    "Veri Seti (CSV, TXT, SAV) veya Proje Dosyası (.pkl)", 
    type=["csv", "txt", "sav", "pkl"]
)

if uploaded_file:
    df = load_data(uploaded_file)

    if df is not None:
        st.write('### 📊 Veri Önizleme:', df.head())

        # --- GLOBAL DEĞİŞKEN ADI DÜZENLEME ---
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        st.sidebar.markdown("---")
        with st.sidebar.expander("✏️ Değişken Etiketlerini Düzenle", expanded=False):
            st.caption("Grafik ve tablolarda görünecek isimleri değiştirebilirsiniz:")
            custom_names = {}
            for col in df.columns:
                key_name = f"global_rename_{col}"
                default_val = st.session_state.get(key_name, col)
                new_val = st.text_input(f"{col} ->", value=default_val, key=key_name)
                custom_names[col] = new_val

        # --- SIDEBAR GRAFİK AYARLARI ---
        st.sidebar.header("⚙️ Analiz & Grafik Ayarları")
        
        analysis_type = st.sidebar.radio(
            "Analiz Türü Seçin",
            ["Correlation Heatmap", "Single ROC Curve", "Multiple ROC Curves"],
            key="analysis_type"
        )

        palette_options = [
            "coolwarm", "RdBu", "RdBu_r", "vlag", "icefire", "Spectral", "RdYlGn", "RdYlBu", 
            "rocket", "mako", "flare", "crest", "viridis", "plasma", "inferno", "magma", "cividis",
            "Blues", "Reds", "Greens", "Purples", "Greys"
        ]

        roc_palette_options = ["tab10", "Set1", "Set2", "Dark2", "Accent", "viridis", "coolwarm", "rainbow"]

        # Sekmeleri Oluştur
        tab1, tab2, tab3 = st.tabs(["🔬 Analiz & Grafik", "📋 SPSS Benzeri Detaylı Tablolar", "💾 Proje İşlemleri"])

        # --- 1. KORELASYON ANALİZİ ---
        if analysis_type == "Correlation Heatmap":
            st.sidebar.subheader("Korelasyon Parametreleri")
            
            corr_method = st.sidebar.selectbox(
                "Korelasyon Yöntemi",
                ["spearman", "pearson", "kendall"],
                format_func=lambda x: "Spearman (ρ)" if x == "spearman" else ("Pearson (r)" if x == "pearson" else "Kendall (τ)"),
                key="corr_method"
            )

            heatmap_shape = st.sidebar.radio(
                "Matris Görünümü",
                ["Alt Üçgen", "Üst Üçgen", "Tam Matris"],
                key="hm_shape"
            )

            palette_choice = st.sidebar.selectbox("Heatmap Renk Paleti", palette_options, key="palette_choice")

            correlation_vars = st.sidebar.multiselect(
                "Korelasyon Değişkenleri (Numerik)",
                options=num_cols,
                default=num_cols[:5] if len(num_cols) >= 5 else num_cols,
                key="corr_vars"
            )

            if len(correlation_vars) < 2:
                st.warning("Lütfen en az 2 sayısal değişken seçiniz.")
            else:
                heatmap_title = st.sidebar.text_input("Grafik Başlığı", value=f"{corr_method.capitalize()} Correlation Heatmap", key="hm_title")
                show_annot = st.sidebar.checkbox("Değerleri Göster", value=True, key="hm_annot")
                font_scale = st.sidebar.slider("Yazı Boyutu", 0.5, 2.0, 1.0, key="hm_font")

                # Korelasyon Hesabı
                df_corr_sub = df[correlation_vars].apply(pd.to_numeric, errors='coerce')
                corr_df, p_df, n_df = compute_correlation_matrices(df_corr_sub, method=corr_method)
                
                # Etiket İsimlerini Değiştir
                renamed_labels = [custom_names.get(c, c) for c in corr_df.columns]
                corr_df_display = corr_df.copy()
                corr_df_display.columns = renamed_labels
                corr_df_display.index = renamed_labels

                # Matris Üçgen Maskesi Ayarı
                if heatmap_shape == "Alt Üçgen":
                    mask = np.triu(np.ones_like(corr_df, dtype=bool))
                elif heatmap_shape == "Üst Üçgen":
                    mask = np.tril(np.ones_like(corr_df, dtype=bool))
                else:
                    mask = None

                with tab1:
                    calc_size = max(8, len(correlation_vars) * 0.9)
                    fig, ax = plt.subplots(figsize=(calc_size, calc_size * 0.8))
                    
                    sns.set_theme(font_scale=font_scale)
                    sns.heatmap(
                        corr_df_display, mask=mask, cmap=palette_choice, center=0,
                        annot=show_annot, fmt=".2f", square=True, 
                        linewidths=.5, cbar_kws={"shrink": .75}, ax=ax
                    )
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                    plt.title(heatmap_title)
                    
                    st.pyplot(fig, use_container_width=True)
                    
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    st.download_button("💾 Resmi İndir (300 DPI)", buf.getvalue(), "heatmap.png", "image/png")
                    sns.reset_orig()

                # Tablo Sekmesi (SPSS Benzeri Output)
                with tab2:
                    st.write(f"### 📋 SPSS Formatında {corr_method.capitalize()} Korelasyon Tablosu")
                    st.caption("* p < 0.05, ** p < 0.01, *** p < 0.001")
                    spss_corr_table = generate_spss_corr_table(corr_df, p_df, n_df, custom_names, corr_method)
                    st.dataframe(spss_corr_table, use_container_width=True)

                    # Excel / CSV İndirme
                    excel_buf = BytesIO()
                    spss_corr_table.to_excel(excel_buf, index=False)
                    st.download_button("⬇️ SPSS Tablosunu Excel Olarak İndir", excel_buf.getvalue(), "spss_correlation_table.xlsx", "application/vnd.ms-excel")

        # --- 2. SINGLE ROC CURVE ---
        elif analysis_type == "Single ROC Curve":
            outcome_var = st.sidebar.selectbox("Outcome (Hastalık 0/1)", df.columns, key="s_outcome")
            predictor_var = st.sidebar.selectbox("Predictor (Değer)", num_cols, key="s_predictor")
            plot_title = st.sidebar.text_input("Başlık", "Single ROC Curve Analysis", key="s_title")
            line_color = st.sidebar.color_picker("Çizgi Rengi", "#800080", key="s_color")

            # Veri Hazırlığı
            y_true = pd.to_numeric(df[outcome_var], errors='coerce')
            y_scores = pd.to_numeric(df[predictor_var], errors='coerce')
            mask = ~y_true.isna() & ~y_scores.isna()
            y_true, y_scores = y_true[mask].astype(int), y_scores[mask].astype(float)
            if set(y_true.unique()) == {1, 2}: y_true = y_true.replace({2: 0, 1: 1})

            # ROC Hesabı
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            if roc_auc < 0.5:
                y_scores = -y_scores
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                st.info(f"🔄 Bilgi: '{predictor_var}' ters ilişkili olduğu için otomatik çevrildi.")

            best_idx = np.argmax(tpr - fpr)
            best_threshold = thresholds[best_idx]
            sens, spec = tpr[best_idx]*100, (1-fpr[best_idx])*100

            # PPV/NPV
            pred_cls = (y_scores >= best_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred_cls).ravel()
            ppv = 100*tp/(tp+fp) if (tp+fp)>0 else 0
            npv = 100*tn/(tn+fn) if (tn+fn)>0 else 0

            # p-value & CI
            pos, neg = y_scores[y_true==1], y_scores[y_true==0]
            try: _, p_val = mannwhitneyu(pos, neg)
            except: p_val = 1.0
            
            se, ci_low, ci_upp = calculate_auc_ci(roc_auc, len(pos), len(neg))
            disp_predictor_name = custom_names.get(predictor_var, predictor_var)

            with tab1:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.plot(fpr*100, tpr*100, color=line_color, lw=2, label=f'{disp_predictor_name} (AUC={roc_auc:.3f})')
                ax.plot([0, 100], [0, 100], 'k--', lw=1)
                ax.set(xlabel='100 - Specificity (%)', ylabel='Sensitivity (%)', xlim=[0,100], ylim=[0,105], title=plot_title)
                ax.legend(loc='lower right')
                
                info_text = f"Cut-off: {best_threshold:.3f}\nSens: {sens:.1f}%\nSpec: {spec:.1f}%\nAUC %95 CI: [{ci_low:.3f} - {ci_upp:.3f}]"
                ax.text(52, 12, info_text, bbox=dict(boxstyle="round", facecolor="white", edgecolor="navy", alpha=0.8))

                st.pyplot(fig, use_container_width=True)
                
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                st.download_button("💾 Resmi İndir (300 DPI)", buf.getvalue(), "roc_single.png", "image/png")

            with tab2:
                st.write("### 📋 SPSS Formatında Diagnostik Performans Tablosu")
                spss_roc_tbl = pd.DataFrame([{
                    "Marker": disp_predictor_name,
                    "AUC": f"{roc_auc:.3f}",
                    "Std. Error": f"{se:.3f}",
                    "Asymptotic Sig. (p)": f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001",
                    "%95 CI Alt": f"{ci_low:.3f}",
                    "%95 CI Üst": f"{ci_upp:.3f}",
                    "Cut-off": f"{best_threshold:.3f}",
                    "Duyarlılık (Sens)": f"{sens:.1f}%",
                    "Özgüllük (Spec)": f"{spec:.1f}%",
                    "PPV (PGD)": f"{ppv:.1f}%",
                    "NPV (NGD)": f"{npv:.1f}%"
                }])
                st.dataframe(spss_roc_tbl, use_container_width=True)

        # --- 3. MULTIPLE ROC CURVES ---
        elif analysis_type == "Multiple ROC Curves":
            outcome_var = st.sidebar.selectbox("Outcome (Hastalık 0/1)", df.columns, key="m_outcome")
            layout_mode = st.sidebar.radio("Grafik Düzeni", ["Tek Grafik", "2 Panel (Yan Yana)", "4 Panel (2x2 Grid)"], key="m_layout")
            plot_title = st.sidebar.text_input("Ana Başlık", "Combined ROC Analysis", key="m_title")
            roc_palette_choice = st.sidebar.selectbox("ROC Çizgi Paleti", roc_palette_options, key="m_roc_palette")

            n_panels = 1 if layout_mode == "Tek Grafik" else (2 if layout_mode == "2 Panel (Yan Yana)" else 4)
            panel_selections = []

            st.sidebar.markdown("---")
            st.sidebar.write("### 🎛️ Panel İçerikleri")
            for i in range(n_panels):
                selection = st.sidebar.multiselect(
                    f"Panel {i+1} Değişkenleri", 
                    options=num_cols,
                    key=f"m_panel_{i}"
                )
                panel_selections.append(selection)

            if any(panel_selections):
                if n_panels == 1:
                    fig, axes = plt.subplots(1, 1, figsize=(8, 7))
                    axes_flat = [axes]
                elif n_panels == 2:
                    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
                    axes_flat = axes.flatten()
                else:
                    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
                    axes_flat = axes.flatten()

                results_list = []

                for i, ax in enumerate(axes_flat):
                    current_vars = panel_selections[i] if i < len(panel_selections) else []
                    ax.plot([0,100], [0,100], 'k--', lw=1)
                    ax.set(xlim=[0,100], ylim=[0,105], xlabel='100 - Specificity (%)', ylabel='Sensitivity (%)')
                    ax.grid(True, alpha=0.3)

                    if not current_vars:
                        if n_panels > 1: ax.text(50, 50, "Değişken Seçilmedi", ha='center')
                        continue

                    cmap = plt.cm.get_cmap(roc_palette_choice, max(len(current_vars), 2))

                    for j, var in enumerate(current_vars):
                        y_t = pd.to_numeric(df[outcome_var], errors='coerce')
                        y_s = pd.to_numeric(df[var], errors='coerce')
                        mask = ~y_t.isna() & ~y_s.isna()
                        y_t, y_s = y_t[mask].astype(int), y_s[mask].astype(float)
                        if set(y_t.unique()) == {1, 2}: y_t = y_t.replace({2: 0, 1: 1})

                        fpr, tpr, thres = roc_curve(y_t, y_s)
                        auc_val = auc(fpr, tpr)
                        inverted = False
                        if auc_val < 0.5:
                            y_s = -y_s
                            fpr, tpr, thres = roc_curve(y_t, y_s)
                            auc_val = auc(fpr, tpr)
                            inverted = True

                        best_idx = np.argmax(tpr - fpr)
                        sens, spec = tpr[best_idx]*100, (1-fpr[best_idx])*100
                        cutoff = thres[best_idx]

                        pred_cls = (y_s >= cutoff).astype(int)
                        tn, fp, fn, tp = confusion_matrix(y_t, pred_cls).ravel()
                        ppv = 100*tp/(tp+fp) if (tp+fp)>0 else 0
                        npv = 100*tn/(tn+fn) if (tn+fn)>0 else 0

                        pos, neg = y_s[y_t==1], y_s[y_t==0]
                        try: _, p_val = mannwhitneyu(pos, neg)
                        except: p_val = 1.0
                        
                        se, ci_low, ci_upp = calculate_auc_ci(auc_val, len(pos), len(neg))

                        disp_name = custom_names.get(var, var)
                        lbl = disp_name + (" [Inv]" if inverted else "")

                        results_list.append({
                            "Panel": f"Panel {i+1}",
                            "Marker": lbl,
                            "AUC": f"{auc_val:.3f}",
                            "Std. Error": f"{se:.3f}",
                            "p-value": f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001",
                            "%95 CI": f"[{ci_low:.3f} - {ci_upp:.3f}]",
                            "Cut-off": f"{cutoff:.3f}",
                            "Sens (%)": f"{sens:.1f}",
                            "Spec (%)": f"{spec:.1f}",
                            "PPV (%)": f"{ppv:.1f}",
                            "NPV (%)": f"{npv:.1f}"
                        })

                        ax.plot(fpr*100, tpr*100, lw=2, color=cmap(j), label=f'{lbl} (AUC={auc_val:.3f})')

                    ax.legend(loc='lower right', fontsize='small')

                plt.suptitle(plot_title, fontsize=16)
                plt.tight_layout()

                with tab1:
                    st.pyplot(fig, use_container_width=True)
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    st.download_button("💾 Grafiği İndir (300 DPI)", buf.getvalue(), "roc_multi.png", "image/png")

                with tab2:
                    st.write("### 📋 SPSS Formatında Karşılaştırmalı Diagnostik Tablo")
                    res_df = pd.DataFrame(results_list)
                    st.dataframe(res_df, use_container_width=True)

                    excel_buf = BytesIO()
                    res_df.to_excel(excel_buf, index=False)
                    st.download_button("⬇️ SPSS ROC Tablosunu Excel Olarak İndir", excel_buf.getvalue(), "spss_multi_roc.xlsx", "application/vnd.ms-excel")

        # --- 3. PROJE KAYDETME SEKMESİ ---
        with tab3:
            st.header("💾 Projeyi Bilgisayara Kaydet")
            st.info("Bu işlem verinizi, değişken isimlerinizi ve tüm grafik parametrelerinizi kapsayan bir .pkl dosyası indirir.")
            
            if st.button("Proje Dosyasını Oluştur ve İndir"):
                project_state = {
                    "data_frame": df,
                    "palette_choice": st.session_state.get("palette_choice"),
                    "analysis_type": st.session_state.get("analysis_type"),
                    "corr_method": st.session_state.get("corr_method"),
                    "hm_shape": st.session_state.get("hm_shape"),
                    "corr_vars": st.session_state.get("corr_vars"),
                    "hm_title": st.session_state.get("hm_title"),
                    "hm_annot": st.session_state.get("hm_annot"),
                    "hm_font": st.session_state.get("hm_font"),
                    "s_outcome": st.session_state.get("s_outcome"),
                    "s_predictor": st.session_state.get("s_predictor"),
                    "s_title": st.session_state.get("s_title"),
                    "s_color": st.session_state.get("s_color"),
                    "m_outcome": st.session_state.get("m_outcome"),
                    "m_title": st.session_state.get("m_title"),
                    "m_layout": st.session_state.get("m_layout"),
                    "m_roc_palette": st.session_state.get("m_roc_palette")
                }

                # Dinamik Key'leri Ekle
                for key in st.session_state:
                    if key.startswith("m_panel_") or key.startswith("global_rename_"):
                        project_state[key] = st.session_state[key]

                buffer = BytesIO()
                pickle.dump(project_state, buffer)
                buffer.seek(0)
                
                st.download_button(
                    label="⬇️ Proje Dosyasını İndir (.pkl)",
                    data=buffer,
                    file_name="analiz_projesi.pkl",
                    mime="application/octet-stream"
                )
