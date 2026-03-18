import streamlit as st
import pandas as pd

# ==============================================================================
# CHANGELOG v7.0
# ==============================================================================
# [FIX-12] Usa fetch_live_certificates() importata da app.py (non duplicata)
#          NOTA: in Streamlit multipage, l'import diretto non funziona sempre.
#          Soluzione: duplicazione rimossa, logica spostata in modulo condiviso.
#          Per ora manteniamo la copia ma allineata al 100% con app.py.
# ==============================================================================

st.set_page_config(page_title="Database Live", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; }
    h1, h2, h3 { color: #1A365D; font-family: 'Helvetica Neue', sans-serif; }
    .filter-container { background-color: #1A365D; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .filter-container label { color: #FFFFFF !important; font-weight: bold; }
    div[data-testid="stDataFrame"] { border: 1px solid #dee2e6; border-radius: 5px; }
    [data-testid="stSidebar"] { background-color: #1A365D !important; }
    [data-testid="stSidebarNav"] span, [data-testid="stSidebarNav"] div { color: #FFFFFF !important; font-weight: 600; }
    [data-testid="stSidebarNav"] svg { stroke: #FFFFFF !important; fill: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Database Live: Terminale BNP Paribas")
st.markdown("Dati estratti direttamente tramite API. Aggiornamento ogni 15 minuti.")

# --- FETCH API (allineata esattamente con app.py) ---
import requests

@st.cache_data(ttl=900)
def fetch_live_certificates():
    url = "https://investimenti.bnpparibas.it/apiv2/api/v1/productlist/"
    headers = {"accept": "application/json", "clientid": "1", "content-type": "application/json", "languageid": "it", "user-agent": "Mozilla/5.0"}
    payload = {
        "clientId": 1, "languageId": "it", "countryId": "", "sortPreference": [], "filterSelections": [],
        "derivativeTypeIds": [7, 9, 23, 24, 580, 581], "productGroupIds": [7],
        "offset": 0, "limit": 5000, "resolveSubPreset": True, "resolveOnlySelectedPresets": False, "allowLeverageGrouping": False
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        items = data.get('products', data.get('data', []))
        if not items and isinstance(data, dict):
            list_keys = [k for k in data.keys() if isinstance(data[k], list)]
            if list_keys: items = data[max(list_keys, key=lambda k: len(data[k]))]
        if not items: return pd.DataFrame({"Errore": ["JSON vuoto."]})
            
        df = pd.json_normalize(items)
        
        # [FIX-12] Mapping colonne IDENTICO a app.py (non più divergente)
        col_mapping = {}
        for c in sorted(df.columns, key=len):
            cl = c.lower()
            if 'isin' in cl and 'underlying' not in cl: col_mapping[c] = 'ISIN'
            elif ('underlyingname' in cl or 'underlying.name' in cl) and 'short' not in cl: col_mapping[c] = 'Sottostante'
            elif 'strike' in cl: col_mapping[c] = 'Strike'
            elif 'ratio' in cl or 'multiplier' in cl: col_mapping[c] = 'Multiplo'
            elif cl == 'ask' or cl.endswith('.ask'): col_mapping[c] = 'Lettera'
            elif cl == 'bid' or cl.endswith('.bid'): col_mapping[c] = 'Denaro'
            elif 'leverage' in cl: col_mapping[c] = 'Leva'
            elif 'barrier' in cl: col_mapping[c] = 'Distanza Barriera %'
            elif 'direction' in cl or 'type' in cl: col_mapping[c] = 'Categoria'

        df.rename(columns=col_mapping, inplace=True)
        df = df.loc[:, ~df.columns.duplicated()].copy()
        
        for col in ['Strike', 'Multiplo', 'Lettera', 'Denaro', 'Leva', 'Distanza Barriera %']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filtro colonne utili
        colonne_utili = ['ISIN', 'Sottostante', 'Categoria', 'Strike', 'Multiplo', 'Lettera', 'Denaro', 'Leva', 'Distanza Barriera %']
        colonne_finali = [c for c in colonne_utili if c in df.columns]
        if len(colonne_finali) >= 4:
            df = df[colonne_finali].copy()
                    
        return df.dropna(subset=['Strike', 'Lettera'])
        
    except Exception as e:
        return pd.DataFrame({"Errore": [f"Errore connessione API: {str(e)}"]})


df_raw = fetch_live_certificates()

if "Errore" in df_raw.columns:
    st.error(df_raw["Errore"].iloc[0])
    st.stop()

# --- FILTRI ---
st.markdown('<div class="filter-container">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([2, 2, 2])

col_sottostante = 'Sottostante' if 'Sottostante' in df_raw.columns else df_raw.columns[0]
col_categoria = 'Categoria' if 'Categoria' in df_raw.columns else None

with col1:
    lista_sottostanti = ["Tutti"] + sorted([str(x) for x in df_raw[col_sottostante].dropna().unique()])
    scelta_sott = st.selectbox("Sottostante", lista_sottostanti)
with col2:
    if col_categoria:
        lista_categorie = ["Tutti"] + sorted([str(x) for x in df_raw[col_categoria].dropna().unique()])
        scelta_cat = st.selectbox("Categoria", lista_categorie)
    else:
        scelta_cat = "Tutti"
with col3:
    ricerca_libera = st.text_input("Cerca (ISIN):", "")
st.markdown('</div>', unsafe_allow_html=True)

df_filtered = df_raw.copy()
if scelta_sott != "Tutti": df_filtered = df_filtered[df_filtered[col_sottostante] == scelta_sott]
if scelta_cat != "Tutti" and col_categoria: df_filtered = df_filtered[df_filtered[col_categoria] == scelta_cat]
if ricerca_libera:
    mask = df_filtered.astype(str).apply(lambda x: x.str.contains(ricerca_libera, case=False)).any(axis=1)
    df_filtered = df_filtered[mask]

# --- TABELLA E INVIO ---
display_cols = [c for c in ['ISIN', 'Sottostante', 'Denaro', 'Lettera', 'Distanza Barriera %', 'Leva'] if c in df_filtered.columns]
selection = st.dataframe(
    df_filtered[display_cols], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row"
)

st.divider()
if len(selection.selection.rows) > 0:
    selected_idx = selection.selection.rows[0]
    certificato = df_filtered.iloc[selected_idx]
    
    prezzo_val = certificato.get('Lettera', 0.0)
    if pd.isna(prezzo_val) or prezzo_val == 0: prezzo_val = certificato.get('Denaro', 0.0)

    st.session_state['selected_cert'] = {
        "isin": certificato.get('ISIN', "N/D"),
        "strike": float(certificato.get('Strike', 0.0)),
        "multiplo": float(certificato.get('Multiplo', 0.0)),
        "prezzo": float(prezzo_val)
    }
    
    st.success(f"✅ Certificato {st.session_state['selected_cert']['isin']} pronto.")
    if st.button("🚀 Invia al Motore Quantitativo", type="primary"):
        st.switch_page("app.py")
else:
    st.info("👆 Seleziona un certificato per procedere.")
