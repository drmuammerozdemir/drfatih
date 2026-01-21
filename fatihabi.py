import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadstat
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.stats import spearmanr, norm, mannwhitneyu
from io import BytesIO
import pickle

# Sayfa Ayarları
st.set_page_config(page_title="Dr. Ozdemir Analysis Tool", layout="wide")
st.title('🔬 ROC AUC & Correlation Dashboard (Proje Kayıt Özellikli)')

# --- FONKSİYONLAR ---
def load_data(uploaded_file):
    """Farklı dosya tiplerini yükler ve session_state'i günceller."""
    file_extension = uploaded_file.name.split('.')[-1]
    
    if file_extension == 'pkl':
        # Proje dosyasını (Veri + Ayarlar) yükle
        try:
            saved_state = pickle.load(uploaded_file)
            # Veriyi al
            df = saved_state.pop('data_frame')
            # Ayarları session_state'e geri yükle
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
        # SPSS dosyaları için geçici dosya oluşturulmalı
        with open("temp.sav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        return pyreadstat.read_sav("temp.sav")[0]
    return None
    
# --- ANA KOD ---

st.sidebar.header("📁 Veri Yükleme")
uploaded_file = st.sidebar.file_uploader(
    "Veri Seti (CSV, TXT, SAV) veya Proje Dosyası (.pkl)", 
    type=["csv", "txt", "sav", "pkl"]
)

if uploaded_file:
    # Veriyi Yükle
    df = load_data(uploaded_file)

    if df is not None:
        st.write('### 📊 Veri Önizleme:', df.head())

        # --- SIDEBAR AYARLARI ---
        st.sidebar.header("⚙️ Grafik Ayarları")
        
        # Key parametreleri eklendi (Hafıza için şart)
        palette_choice = st.sidebar.selectbox(
            "Heatmap Renk Paleti",
            ["coolwarm", "vlag", "rocket", "mako", "icefire"],
            key="palette_choice"
        )

        analysis_type = st.sidebar.radio(
            "Analiz Türü Seçin",
            ["Correlation Heatmap", "Single ROC Curve", "Multiple ROC Curves"],
            key="analysis_type"
        )

        # Sekmeleri Oluştur
        tab1, tab2, tab3 = st.tabs(["🔬 Analiz & Grafik", "📋 Detaylı Tablo", "💾 Proje İşlemleri"])

        # --- ANALİZ MANTIĞI ---
        with tab1:
            # 1. KORELASYON HEATMAP
            if analysis_type == "Correlation Heatmap":
                correlation_vars = st.sidebar.multiselect(
                    "Korelasyon Değişkenleri (Numerik)",
                    options=df.select_dtypes(include=[np.number]).columns,
                    default=df.select_dtypes(include=[np.number]).columns.tolist()[:5], # İlk 5'i varsayılan
                    key="corr_vars"
                )

                if len(correlation_vars) < 2:
                    st.warning("Lütfen en az 2 sayısal değişken seçiniz.")
                else:
                    heatmap_title = st.sidebar.text_input("Grafik Başlığı", value="Spearman Correlation Heatmap", key="hm_title")
                    
                    st.sidebar.markdown("---")
                    show_annot = st.sidebar.checkbox("Değerleri Göster", value=True, key="hm_annot")
                    font_scale = st.sidebar.slider("Yazı Boyutu", 0.5, 2.0, 1.0, key="hm_font")
                    
                    footnote = st.text_area("Grafik Altı Notu", value="", key="hm_note")

                    # Korelasyon Hesabı
                    df_corr = df[correlation_vars].apply(pd.to_numeric, errors='coerce').dropna()
                    corr, _ = spearmanr(df_corr)
                    corr_df = pd.DataFrame(corr, index=df_corr.columns, columns=df_corr.columns)
                    mask = np.triu(np.ones_like(corr_df, dtype=bool))

                    # Dinamik Boyutlandırma
                    calc_size = max(10, len(correlation_vars) * 0.8)
                    fig, ax = plt.subplots(figsize=(calc_size, calc_size * 0.8))
                    
                    sns.set(font_scale=font_scale)
                    sns.heatmap(
                        corr_df, mask=mask, cmap=palette_choice, center=0,
                        annot=show_annot, fmt=".2f", square=True, 
                        linewidths=.5, cbar_kws={"shrink": .75}, ax=ax
                    )
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                    plt.title(heatmap_title)
                    
                    # Ekrana Basma ve İndirme
                    st.pyplot(fig, use_container_width=True)
                    
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    st.download_button("💾 Resmi İndir (300 DPI)", buf.getvalue(), "heatmap.png", "image/png")
                    sns.reset_orig() # Ayarları sıfırla

            # 2. SINGLE ROC CURVE
            elif analysis_type == "Single ROC Curve":
                outcome_var = st.sidebar.selectbox("Outcome (Hastalık 0/1)", df.columns, key="s_outcome")
                predictor_var = st.sidebar.selectbox("Predictor (Değer)", df.columns, key="s_predictor")
                
                plot_title = st.sidebar.text_input("Başlık", "ROC Curve", key="s_title")
                custom_name = st.sidebar.text_input(f"Etiket Adı ({predictor_var})", value=predictor_var, key="s_label")
                
                # Veri Hazırlığı
                y_true = pd.to_numeric(df[outcome_var], errors='coerce')
                y_scores = pd.to_numeric(df[predictor_var], errors='coerce')
                mask = ~y_true.isna() & ~y_scores.isna()
                y_true, y_scores = y_true[mask].astype(int), y_scores[mask].astype(float)
                if set(y_true.unique()) == {1, 2}: y_true = y_true.replace({2: 0, 1: 1})

                # ROC Hesabı ve Otomatik Düzeltme
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                
                if roc_auc < 0.5:
                    y_scores = -y_scores
                    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                    roc_auc = auc(fpr, tpr)
                    st.info(f"🔄 Bilgi: '{predictor_var}' ters ilişkili olduğu için otomatik çevrildi.")

                # Youden Index
                best_idx = np.argmax(tpr - fpr)
                best_threshold = thresholds[best_idx]
                sens, spec = tpr[best_idx]*100, (1-fpr[best_idx])*100

                # Çizim
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.plot(fpr*100, tpr*100, color='purple', lw=2, label=f'{custom_name} (AUC={roc_auc:.3f})')
                
                # CI Alanı (Basit Bootstrapping Görseli)
                # (Hız için sadece çizgiyi çiziyoruz, CI bandı eklenmedi)
                
                ax.plot([0, 100], [0, 100], 'k--')
                ax.set(xlabel='100-Specificity', ylabel='Sensitivity', xlim=[0,100], ylim=[0,105], title=plot_title)
                ax.legend(loc='lower right')
                
                # Info Box
                ax.text(60, 15, f"Sens: {sens:.1f}\nSpec: {spec:.1f}\nCut: {best_threshold:.3f}", 
                        bbox=dict(boxstyle="round", facecolor="white", edgecolor="navy"))

                st.pyplot(fig, use_container_width=True)
                
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                st.download_button("💾 Resmi İndir (300 DPI)", buf.getvalue(), "roc_single.png", "image/png")

                # TABLO KISMI (Single ROC için)
                with tab2:
                    st.write("### Tanısal Performans Tablosu")
                    # P değeri hesabı
                    pos, neg = y_scores[y_true==1], y_scores[y_true==0]
                    try: _, p_val = mannwhitneyu(pos, neg)
                    except: p_val = 1.0
                    
                    tbl = pd.DataFrame({
                        "Marker": [custom_name],
                        "AUC": [f"{roc_auc:.3f}"],
                        "p-value": [f"{p_val:.3f}" + ("*" if p_val<0.05 else "")],
                        "Cut-off": [f"{best_threshold:.3f}"],
                        "Sensitivity": [f"{sens:.1f}"],
                        "Specificity": [f"{spec:.1f}"]
                    })
                    st.dataframe(tbl, use_container_width=True)

            # 3. MULTIPLE ROC CURVES
            elif analysis_type == "Multiple ROC Curves":
                outcome_var = st.sidebar.selectbox("Outcome (Hastalık 0/1)", df.columns, key="m_outcome")
                predictor_vars = st.sidebar.multiselect("Predictor Değişkenler", df.select_dtypes(include=[np.number]).columns, key="m_predictors")
                plot_title = st.sidebar.text_input("Başlık", "Combined ROC Analysis", key="m_title")

                if predictor_vars:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    colors = plt.cm.get_cmap('tab10', len(predictor_vars))
                    results_list = []

                    for i, var in enumerate(predictor_vars):
                        # Veri Hazırla
                        y_t = pd.to_numeric(df[outcome_var], errors='coerce')
                        y_s = pd.to_numeric(df[var], errors='coerce')
                        mask = ~y_t.isna() & ~y_s.isna()
                        y_t, y_s = y_t[mask].astype(int), y_s[mask].astype(float)
                        if set(y_t.unique()) == {1, 2}: y_t = y_t.replace({2: 0, 1: 1})
                        
                        # Hesapla & Düzelt
                        fpr, tpr, thres = roc_curve(y_t, y_s)
                        auc_val = auc(fpr, tpr)
                        inverted = False
                        if auc_val < 0.5:
                            y_s = -y_s
                            fpr, tpr, thres = roc_curve(y_t, y_s)
                            auc_val = auc(fpr, tpr)
                            inverted = True
                        
                        # İstatistikler
                        best_idx = np.argmax(tpr - fpr)
                        sens, spec = tpr[best_idx]*100, (1-fpr[best_idx])*100
                        cutoff = thres[best_idx]
                        
                        # PPV/NPV
                        pred_cls = (y_s >= cutoff).astype(int)
                        tn, fp, fn, tp = confusion_matrix(y_t, pred_cls).ravel()
                        ppv = 100*tp/(tp+fp) if (tp+fp)>0 else 0
                        npv = 100*tn/(tn+fn) if (tn+fn)>0 else 0
                        
                        # P-value
                        try: _, p_val = mannwhitneyu(y_s[y_t==1], y_s[y_t==0])
                        except: p_val = 1.0
                        p_txt = f"{p_val:.3f}" + ("*" if p_val<0.001 else "")

                        lbl = var + (" [Ters]" if inverted else "")
                        results_list.append({
                            "Variable": lbl, "AUC": f"{auc_val:.3f}", "p": p_txt,
                            "Cut-off": f"{cutoff:.3f}", "Sens": f"{sens:.3f}", 
                            "Spec": f"{spec:.3f}", "PPV": f"{ppv:.3f}", "NPV": f"{npv:.3f}"
                        })

                        ax.plot(fpr*100, tpr*100, lw=2, color=colors(i%10), label=f'{lbl} (AUC={auc_val:.3f})')

                    ax.plot([0,100], [0,100], 'k--', lw=1)
                    ax.set(xlim=[0,100], ylim=[0,105], xlabel='100-Specificity', ylabel='Sensitivity', title=plot_title)
                    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig, use_container_width=True)
                    
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    st.download_button("💾 Grafiği İndir (300 DPI)", buf.getvalue(), "roc_multi.png", "image/png")

                    st.write("### Karşılaştırmalı Tablo")
                    st.dataframe(pd.DataFrame(results_list), use_container_width=True)

        # --- PROJE KAYDETME SEKMESİ ---
        with tab3:
            st.header("💾 Projeyi Bilgisayara Kaydet")
            st.info("""
            Bu özellik, mevcut verinizi ve yaptığınız tüm seçimleri (değişkenler, renkler, başlıklar) 
            bir dosya (.pkl) olarak indirir. Daha sonra bu dosyayı 'Veri Yükleme' kısmından yükleyerek 
            kaldığınız yerden devam edebilirsiniz.
            """)
            
            # Kaydedilecek verileri hazırla
            if st.button("Proje Dosyasını Oluştur ve İndir"):
                project_state = {
                    "data_frame": df, # Verinin kendisi
                    # Widget Key'lerini kaydet
                    "palette_choice": st.session_state.get("palette_choice"),
                    "analysis_type": st.session_state.get("analysis_type"),
                    "corr_vars": st.session_state.get("corr_vars"),
                    "hm_title": st.session_state.get("hm_title"),
                    "hm_annot": st.session_state.get("hm_annot"),
                    "hm_font": st.session_state.get("hm_font"),
                    "hm_note": st.session_state.get("hm_note"),
                    "s_outcome": st.session_state.get("s_outcome"),
                    "s_predictor": st.session_state.get("s_predictor"),
                    "s_title": st.session_state.get("s_title"),
                    "s_label": st.session_state.get("s_label"),
                    "m_outcome": st.session_state.get("m_outcome"),
                    "m_predictors": st.session_state.get("m_predictors"),
                    "m_title": st.session_state.get("m_title"),
                }
                
                # Pickle ile paketle
                buffer = BytesIO()
                pickle.dump(project_state, buffer)
                buffer.seek(0)
                
                st.download_button(
                    label="⬇️ Proje Dosyasını İndir (.pkl)",
                    data=buffer,
                    file_name="analiz_projesi.pkl",
                    mime="application/octet-stream"
                )

