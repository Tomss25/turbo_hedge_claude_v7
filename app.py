import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import yfinance as yf
import plotly.graph_objects as go
from calculator import TurboParameters, DeterministicTurboCalculator
from charts import generate_scenario_data, generate_sensitivity_matrix, plot_payoff_profile, plot_pl_waterfall
from stress_test import run_stress_test
from backtest import run_historical_backtest, generate_pdf_report

# ==============================================================================
# CHANGELOG v7.0 - TUTTI I FIX APPLICATI
# ==============================================================================
# [FIX-1]  Capitale modalità manuale: usa calculator.override_manual_quantity()
# [FIX-2]  Toggle is_real_ratio ora controlla quale hedge ratio mostrare
# [FIX-3]  Colori diagnosi backtest: usa diag['bg_color'] e diag['color'] hex
# [FIX-4]  Matrice sensibilità: usa generate_sensitivity_matrix() dal calculator
# [FIX-12] fetch_live_certificates() centralizzata (non duplicata)
# [FIX-18] Rimosso import matplotlib (non utilizzato)
# [FIX-19] run_stress_test() ora visualizzato nel Tab 1
# [FIX-16] Soglie backtest parametrizzabili dall'utente
# ==============================================================================

# --- STATO DELLA SESSIONE ---
if 'selected_cert' not in st.session_state:
    st.session_state['selected_cert'] = None

st.set_page_config(page_title="Turbo Hedge Quant", layout="wide", page_icon="🏦")

# --- CSS CORPORATE ---
st.markdown("""
<style>
    .stApp { background-color: #F4F7F6; }
    h1, h2, h3 { color: #1A365D; font-family: 'Helvetica Neue', sans-serif; }
    div[data-testid="stForm"] { background-color: #FFFFFF; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: none; }
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; }
    .stTabs [data-baseweb="tab"] { background-color: #E2E8F0; border-radius: 8px 8px 0 0; border: none; }
    .stTabs [aria-selected="true"] { background-color: #1A365D !important; color: white !important; }
    .excel-table { width: 100%; border-collapse: collapse; font-size: 14px; margin-bottom: 20px; }
    .excel-table td { padding: 8px 12px; border: 1px solid #dee2e6; }
    .excel-header { background-color: #2c5282; color: white; font-weight: bold; text-align: center; }
    .excel-label { background-color: #f8f9fa; color: #6c757d; font-weight: 500; width: 60%; }
    .excel-value { text-align: right; font-weight: bold; color: #1A365D; }
    [data-testid="stSidebar"] { background-color: #1A365D !important; }
    [data-testid="stSidebarNav"] span, [data-testid="stSidebarNav"] div { color: #FFFFFF !important; font-weight: 600; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stNumberInput label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: #FFFFFF !important; }
    div[data-testid="stFormSubmitButton"] button { background-color: #800020 !important; color: #FFFFFF !important; border: none !important; font-weight: bold !important; padding: 10px 24px !important; border-radius: 6px !important; }
    div[data-testid="stFormSubmitButton"] button:hover { background-color: #5c0017 !important; color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

# --- [FIX-12] FUNZIONE CENTRALIZZATA FETCH API ---
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
        data = response.json()
        items = data.get('products', data.get('data', []))
        if not items and isinstance(data, dict):
            list_keys = [k for k in data.keys() if isinstance(data[k], list)]
            if list_keys: items = data[max(list_keys, key=lambda k: len(data[k]))]
        if not items: return pd.DataFrame()
        
        df = pd.json_normalize(items)
        col_mapping = {}
        for c in sorted(df.columns, key=len):
            cl = c.lower()
            if 'isin' in cl and 'underlying' not in cl: col_mapping[c] = 'ISIN'
            elif ('underlyingname' in cl or 'underlying.name' in cl) and 'short' not in cl: col_mapping[c] = 'Sottostante'
            elif ('productname' in cl or cl == 'name' or 'product.name' in cl) and 'underlying' not in cl and 'asset' not in cl: col_mapping[c] = 'Nome Certificato'
            elif 'direction' in cl or ('type' in cl and 'derivative' not in cl and 'asset' not in cl and 'product' not in cl and 'id' not in cl): col_mapping[c] = 'Long/Short'
            elif 'strike' in cl: col_mapping[c] = 'Strike'
            elif 'ratio' in cl or 'multiplier' in cl: col_mapping[c] = 'Multiplo'
            elif cl == 'ask' or cl.endswith('.ask'): col_mapping[c] = 'Lettera'
            elif cl == 'bid' or cl.endswith('.bid'): col_mapping[c] = 'Denaro'
            elif 'leverage' in cl: col_mapping[c] = 'Leva'
            elif 'barrier' in cl: col_mapping[c] = 'Distanza Barriera %'
            elif 'assetclassid' in cl or 'assetclass.id' in cl: col_mapping[c] = 'Categoria_ID'

        df = df.rename(columns=col_mapping)
        df = df.loc[:, ~df.columns.duplicated()].copy()
        
        # Ricava Long/Short dal nome del certificato se la colonna dedicata non esiste
        if 'Long/Short' not in df.columns and 'Nome Certificato' in df.columns:
            df['Long/Short'] = df['Nome Certificato'].astype(str).apply(
                lambda x: 'Short' if 'short' in x.lower() else ('Long' if 'long' in x.lower() else 'N/D')
            )
        elif 'Long/Short' not in df.columns:
            # Ultimo tentativo: cerca in tutte le colonne stringa
            tipo_cols = [c for c in df.columns if df[c].astype(str).str.contains('Short|Long', case=False, na=False).any()]
            if tipo_cols:
                src = tipo_cols[0]
                df['Long/Short'] = df[src].astype(str).apply(
                    lambda x: 'Short' if 'short' in x.lower() else ('Long' if 'long' in x.lower() else 'N/D')
                )
            else:
                df['Long/Short'] = 'N/D'
        
        asset_map = {1: 'Azioni', 2: 'Indici', 3: 'Valute', 4: 'Materie prime', 5: 'Tassi di interesse', 11: 'ETF', 14: 'Volatility'}
        if 'Categoria_ID' in df.columns:
            df['Classe'] = pd.to_numeric(df['Categoria_ID'], errors='coerce').map(asset_map).fillna('Altro')
        
        for col in ['Strike', 'Multiplo', 'Lettera', 'Denaro', 'Leva', 'Distanza Barriera %']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna(subset=['Strike', 'Lettera'])
    except Exception:
        return pd.DataFrame()

# --- [LIM-1] FETCH VOLATILITÀ IMPLICITA (VIX o VSTOXX) ---
@st.cache_data(ttl=900)
def fetch_volatility_index(ticker: str = "^VIX"):
    """
    Scarica l'indice di volatilità corrente da Yahoo Finance.
    ^VIX  = CBOE Volatility Index (S&P 500, esposizione US)
    ^V2X  = VSTOXX (Euro Stoxx 50, esposizione EU)
    Restituisce il valore decimale (es. 18.5 → 0.185) o None se non disponibile.
    """
    try:
        data = yf.download(ticker, period="5d", progress=False)['Close']
        if not data.empty:
            val = float(data.iloc[-1])
            if hasattr(val, '__iter__'):
                val = float(data.iloc[-1].iloc[0])
            return val / 100
        return None
    except Exception:
        return None

# --- SIDEBAR ---
st.sidebar.markdown("<h2 style='color: white;'>📉 Attriti di Mercato</h2>", unsafe_allow_html=True)
ui_spread = st.sidebar.number_input("Bid-Ask Spread (%)", value=0.5, step=0.1) / 100
ui_comm = st.sidebar.number_input("Commissioni (%)", value=0.1, step=0.05) / 100
ui_div = st.sidebar.number_input("Dividend Yield (%)", value=1.5, step=0.1) / 100

st.title("🏦 Dashboard Copertura Istituzionale (v7.0)")

# [FIX-2] Toggle ora effettivamente utilizzato
is_real_ratio = st.toggle("🛡️ **Hedge Ratio Netto (Risk Manager)**", value=True, 
                           help="ON = Ratio al netto di spread/commissioni. OFF = Ratio lordo (commerciale).")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Setup & Matrice", "📈 Backtest Storico", "🔍 Database Live", "🤖 Advisor Strategico"])

# ======================================================================
# TAB 1: SETUP & RISULTATI 
# ======================================================================
with tab1:
    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### ⚙️ Caratteristiche Turbo SHORT")
            cert = st.session_state.get('selected_cert')
            if cert: st.info(f"ISIN: {cert['isin']}")
            p_iniziale = st.number_input("Prezzo Lettera (€)", value=cert['prezzo'] if cert else 7.64, step=0.01)
            strike = st.number_input("Strike", value=cert['strike'] if cert else 7505.97, step=0.01)
            cambio = st.number_input("Cambio", value=1.15, step=0.01)
            multiplo = st.number_input("Multiplo", value=cert['multiplo'] if cert else 0.01, format="%.4f")
            euribor = st.number_input("Euribor 12M", value=0.02456, format="%.5f")
        with col2:
            st.markdown("### 📉 Indice da Coprire")
            v_iniziale = st.number_input("Spot", value=6670.75, step=0.01)
            v_ipotetico = st.number_input("Target", value=6000.0, step=0.01)
            giorni = st.number_input("Giorni Hedging", value=60, step=1)
        with col3:
            st.markdown("### 💼 Portafoglio")
            ptf = st.number_input("Capitale Ptf (€)", value=200000.0, step=1000.0)
            beta = st.number_input("Beta", value=1.00, step=0.05)
            esposizione_geo = st.radio("🌍 Esposizione Geografica", ["🇪🇺 Europa (VSTOXX)", "🇺🇸 USA (VIX)"], horizontal=False,
                                        help="Seleziona il mercato del sottostante per calibrare la volatilità implicita.")
        st.divider()
        tipo_c = st.radio("Ottimizzazione", ["Auto", "Manuale"], horizontal=True)
        n_custom = st.number_input("Qtà", value=1000, step=10) if tipo_c == "Manuale" else None
        
        if st.form_submit_button("🔥 Calcola"):
            try:
                # [LIM-1] Fetch volatilità implicita in base all'esposizione geografica
                if "Europa" in esposizione_geo:
                    vol_ticker = "^V2X"
                    vol_label = "VSTOXX Live"
                else:
                    vol_ticker = "^VIX"
                    vol_label = "VIX Live"
                
                vol_sigma = fetch_volatility_index(vol_ticker)
                # Se il fetch del VSTOXX fallisce, prova il VIX come fallback
                if vol_sigma is None and vol_ticker == "^V2X":
                    vol_sigma = fetch_volatility_index("^VIX")
                    if vol_sigma is not None:
                        vol_label = "VIX Live (fallback)"
                
                params = TurboParameters(
                    p_iniziale, strike, cambio, multiplo, euribor, 
                    v_iniziale, v_ipotetico, giorni, ptf, beta, 
                    dividend_yield=ui_div, bid_ask_spread=ui_spread, commissioni_pct=ui_comm,
                    volatilita=vol_sigma  # None = fallback a stima dal premio
                )
                calc = DeterministicTurboCalculator(params)
                
                # [FIX-1] Usa il metodo dedicato per override manuali
                if n_custom:
                    res = calc.override_manual_quantity(int(n_custom))
                else:
                    res = calc.calculate_all()
                
                # [LIM-1] Monte Carlo
                mc_res = calc.run_monte_carlo(n_sim=5000)
                
                st.session_state['res'] = res
                st.session_state['params'] = params
                st.session_state['barriera_calcolata'] = res['barriera']
                st.session_state['mc_res'] = mc_res
                st.session_state['vix_source'] = vol_label if vol_sigma else "Stima dal Premio"
            except ValueError as e:
                # [FIX-14] Mostra errori di validazione
                st.error(f"⚠️ Errore nei parametri:\n{e}")

    if 'res' in st.session_state:
        res, params = st.session_state['res'], st.session_state['params']
        
        # [FIX-2] Seleziona il ratio in base al toggle
        display_ratio = res['hedge_ratio_reale'] if is_real_ratio else res['hedge_ratio_commerciale']
        ratio_label = "Netto" if is_real_ratio else "Lordo"
        
        st.divider()
        st.markdown("<h2>📊 Risultati della Copertura</h2>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1.3])
        
        with c1:
            st.markdown(f"""
            <table class="excel-table">
                <tr><td colspan="2" class="excel-header">CARATTERISTICHE TURBO</td></tr>
                <tr><td class="excel-label">Prezzo Lettera</td><td class="excel-value">{params.prezzo_iniziale:.2f} €</td></tr>
                <tr><td class="excel-label">Fair Value</td><td class="excel-value">{res['fair_value']:.4f} €</td></tr>
                <tr><td class="excel-label">Premio</td><td class="excel-value">{res['premio']:.4f} €</td></tr>
                <tr><td class="excel-label">Strike</td><td class="excel-value">{params.strike:.2f}</td></tr>
                <tr><td class="excel-label">Strike Adj (Div)</td><td class="excel-value">{res['strike_adj']:.2f}</td></tr>
                <tr><td class="excel-label">σ Implicita ({st.session_state.get('vix_source', 'N/D')})</td><td class="excel-value">{res['sigma']*100:.1f}%</td></tr>
            </table>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <table class="excel-table">
                <tr><td colspan="2" class="excel-header">INDICE DA COPRIRE</td></tr>
                <tr><td class="excel-label">Spot</td><td class="excel-value">{params.valore_iniziale:.2f}</td></tr>
                <tr><td class="excel-label">Target</td><td class="excel-value">{params.valore_ipotetico:.2f}</td></tr>
                <tr><td class="excel-label">Prezzo Futuro</td><td class="excel-value">{res['prezzo_futuro']:.4f} €</td></tr>
                <tr><td class="excel-label">Barriera</td><td class="excel-value">{res['barriera']:.2f}</td></tr>
            </table>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"<div style='text-align:right; font-size:22px;'><b>{params.portafoglio:,.2f} €</b></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <table class="excel-table">
                <tr><td class="excel-label">N. Turbo Short</td><td class="excel-value">{res['n_turbo']:,.2f}</td><td rowspan="2" style="background-color:#E3F2FD; font-weight:bold; text-align:center;">COPERTURA<br>{ratio_label.upper()}</td></tr>
                <tr><td class="excel-label">Capitale + Costi</td><td class="excel-value">{res['capitale']:,.2f} €</td></tr>
                <tr><td colspan="2" style="text-align:right; font-weight:bold;">Hedge Ratio ({ratio_label}):</td><td style="background-color:#E3F2FD; text-align:center; font-weight:bold;">{(display_ratio*100):.1f}%</td></tr>
            </table>
            """, unsafe_allow_html=True)
            perf = res['percentuale'] * 100
            st.markdown(f"<div style='background-color:{'#E8F5E9' if perf>=0 else '#FFEBEE'}; text-align:center; padding:15px; border:2px solid {'#2E7D32' if perf>=0 else '#C62828'};'><h3>{perf:+.2f}% Perf. Netta</h3></div>", unsafe_allow_html=True)

        # --- ANALISI ---
        st.markdown("### 📝 Analisi")
        h_ratio = display_ratio * 100
        if h_ratio > 98:
            st.success(f"**Copertura Ottimale ({ratio_label}):** Il sistema ha neutralizzato il {h_ratio:.1f}% del rischio.")
        elif h_ratio > 80:
            st.warning(f"**Sotto-copertura Parziale ({ratio_label}):** Stai coprendo il {h_ratio:.1f}% del drawdown stimato.")
        else:
            st.error(f"**Copertura Insufficiente ({ratio_label}):** L'Hedge Ratio del {h_ratio:.1f}% è troppo basso.")

        if perf < -1:
            st.info(f"**Nota sui Costi:** Il trascinamento negativo del {perf:.2f}% rappresenta il 'premio assicurativo' pagato al mercato.")

        # --- [FIX-19] STRESS TEST (era importato ma non visualizzato) ---
        st.divider()
        st.markdown("### 💥 Stress Test")
        try:
            df_stress = run_stress_test(params)
            st.dataframe(df_stress, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Stress test non disponibile: {e}")

        # --- [FIX-4] MATRICE DI SENSIBILITÀ (usa il calculator, non formula inline) ---
        st.divider()
        st.markdown("### 🌡️ Matrice di Sensibilità")
        df_sens = generate_sensitivity_matrix(params, res)
        st.dataframe(
            df_sens.style.format("{:.3f}€").background_gradient(cmap='RdYlGn', axis=None, vmin=0.0), 
            use_container_width=True
        )
        
        st.divider()
        df_s, b_l = generate_scenario_data(params)
        st.plotly_chart(plot_payoff_profile(df_s, params.valore_iniziale, b_l), use_container_width=True)
        st.plotly_chart(plot_pl_waterfall(res), use_container_width=True)

        # --- [LIM-1] SIMULAZIONE MONTE CARLO ---
        if 'mc_res' in st.session_state:
            mc = st.session_state['mc_res']
            if mc['mc_pl_distribution']:
                st.divider()
                st.markdown("### 🎲 Simulazione Monte Carlo (5.000 percorsi GBM)")
                st.markdown(f"Volatilità utilizzata: **{res['sigma']*100:.1f}%** (fonte: {st.session_state.get('vix_source', 'N/D')})")
                
                # Metriche di rischio
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("VaR 95%", f"€ {mc['mc_var_95']:,.0f}", help="Perdita massima nel 95% dei casi")
                mc2.metric("CVaR 95%", f"€ {mc['mc_cvar_95']:,.0f}", help="Perdita media nel peggior 5% dei casi")
                mc3.metric("Prob. Knock-Out", f"{mc['mc_prob_ko']:.1f}%", help="Probabilità che il sottostante tocchi la barriera")
                mc4.metric("P&L Medio", f"€ {mc['mc_pl_mean']:,.0f}")
                
                # Istogramma distribuzione P&L
                pl_arr = np.array(mc['mc_pl_distribution'])
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Histogram(
                    x=pl_arr, nbinsx=80, name="Distribuzione P&L",
                    marker_color="#2c5282", opacity=0.85
                ))
                # Linee VaR e CVaR
                fig_mc.add_vline(x=mc['mc_var_95'], line_dash="dash", line_color="#C62828", 
                                 annotation_text=f"VaR 95%: € {mc['mc_var_95']:,.0f}", annotation_position="top left")
                fig_mc.add_vline(x=mc['mc_cvar_95'], line_dash="dot", line_color="#E65100",
                                 annotation_text=f"CVaR 95%: € {mc['mc_cvar_95']:,.0f}", annotation_position="top left")
                fig_mc.add_vline(x=0, line_color="black", line_width=1)
                # Linea scenario deterministico
                pl_det = res['pl_portafoglio'] + res['pl_turbo_netto']
                fig_mc.add_vline(x=pl_det, line_dash="dash", line_color="#2E7D32",
                                 annotation_text=f"Deterministico: € {pl_det:,.0f}", annotation_position="top right")
                fig_mc.update_layout(
                    title="Distribuzione P&L Netto (Monte Carlo vs Deterministico)",
                    xaxis_title="P&L Netto (€)", yaxis_title="Frequenza",
                    template='plotly_white', height=400, showlegend=False,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_mc, use_container_width=True)
                
                # Tabella percentili
                st.markdown("##### Distribuzione per Percentili")
                perc = mc['mc_percentiles']
                df_perc = pd.DataFrame({
                    'Percentile': list(perc.keys()),
                    'P&L (€)': [f"€ {v:,.0f}" for v in perc.values()]
                })
                st.dataframe(df_perc, use_container_width=True, hide_index=True)

# ======================================================================
# TAB 2: BACKTEST
# ======================================================================
with tab2:
    st.markdown("### 🕰️ Analisi Storica e Report")
    if 'barriera_calcolata' not in st.session_state: 
        st.warning("⚠️ Esegui prima il calcolo nel Tab 'Setup & Matrice' per generare la barriera.")
    else:
        with st.expander("Parametri Backtest", expanded=True):
            b1, b2, b3 = st.columns(3)
            t_ptf_input = b1.text_input("Ticker Ptf (separati da virgola)", "SPY")
            t_idx = b2.text_input("Ticker Indice", "^GSPC")
            t_fx = b3.text_input("FX (es. EURUSD=X)", "")
        
        # [FIX-16] Soglie parametrizzabili
        with st.expander("⚙️ Soglie Segnali di Copertura"):
            sc1, sc2 = st.columns(2)
            dd_entry = sc1.number_input("Attivazione Hedge (Drawdown %)", value=-5.0, step=1.0, max_value=0.0) / 100
            dd_exit = sc2.number_input("Disattivazione Hedge (Drawdown %)", value=-2.0, step=1.0, max_value=0.0) / 100

        if st.button("🚀 Avvia Backtest"):
            tickers = [t.strip() for t in t_ptf_input.split(",") if t.strip()]
            for current_ticker in tickers:
                st.markdown(f"#### 🔍 Report per: {current_ticker}")
                df_bt, msg, diag = run_historical_backtest(
                    current_ticker, t_idx, t_fx, 
                    datetime.date(2023, 1, 1), datetime.date.today(), 
                    st.session_state['barriera_calcolata'],
                    drawdown_entry=dd_entry,
                    drawdown_exit=dd_exit
                )
                if df_bt is not None:
                    # [FIX-3] Usa i colori hex dal dizionario diagnosi
                    bg = diag.get('bg_color', '#f8f9fa')
                    tc = diag.get('color', '#1A365D')
                    st.markdown(f"""<div style="background-color: {bg}; border-left: 5px solid {tc}; padding: 15px; border-radius: 5px;">
                        <h3 style="color: {tc}; margin-top:0;">{diag['title']}</h3><p>{diag['body']}</p><b>Azione: {diag['action']}</b></div>""", unsafe_allow_html=True)
                    st.line_chart(df_bt.set_index('Date')[['Ptf_Close']])
                    pdf = generate_pdf_report(df_bt, current_ticker, t_idx, t_fx, st.session_state['barriera_calcolata'], diag)
                    st.download_button(f"📄 Scarica Report PDF per {current_ticker}", data=pdf, file_name=f"Quant_Report_{current_ticker}.pdf")
                else: 
                    st.error(f"Errore su {current_ticker}: {msg}")
                st.divider()

# ======================================================================
# TAB 3: DATABASE LIVE
# ======================================================================
with tab3:
    st.markdown("### 🔍 Live Terminal BNP Paribas")
    df_raw = fetch_live_certificates()
    if df_raw.empty: st.error("Nessun dato.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        scelta_s = c1.selectbox("Sottostante", ["Tutti"] + sorted([str(x) for x in df_raw['Sottostante'].unique()]))
        scelta_c = c2.selectbox("Classe", ["Tutte"] + sorted([str(x) for x in df_raw['Classe'].unique()]))
        min_leva = c3.number_input("Leva Minima", value=1.0, step=1.0)
        max_leva = c4.number_input("Leva Massima", value=100.0, step=1.0)
        
        df_f = df_raw.copy()
        if scelta_s != "Tutti": df_f = df_f[df_f['Sottostante'] == scelta_s]
        if scelta_c != "Tutte": df_f = df_f[df_f['Classe'] == scelta_c]
        df_f = df_f[(df_f['Leva'] >= min_leva) & (df_f['Leva'] <= max_leva)]
        
        # Colonne visibili nella tabella (le altre restano nel df per la selezione)
        display_cols = [c for c in ['ISIN', 'Nome Certificato', 'Sottostante', 'Denaro', 'Lettera', 'Leva', 'Long/Short'] if c in df_f.columns]
        sel = st.dataframe(df_f[display_cols], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")
        if len(sel.selection.rows) > 0:
            row = df_f.iloc[sel.selection.rows[0]]
            st.session_state['selected_cert'] = {"isin": row['ISIN'], "strike": row['Strike'], "multiplo": row['Multiplo'], "prezzo": row['Lettera']}
            st.success(f"✅ ISIN {row['ISIN']} caricato."); st.button("Aggiorna ora")

# ======================================================================
# TAB 4: ADVISOR
# ======================================================================
with tab4:
    st.markdown("### 🤖 Advisor Strategico: Match Portafoglio")
    st.markdown("Imposta i vincoli di capitale e di budget per estrarre dal mercato i certificati matematicamente ottimali.")
    
    st.markdown("#### 1️⃣ Definisci i Parametri di Hedging")
    with st.form("adv_form"):
        c1, c2, c3, c4 = st.columns(4)
        v_p = c1.number_input("Valore Portafoglio (€)", value=200000.0, step=1000.0)
        v_b = c2.number_input("Beta", value=1.0, step=0.1)
        v_bud = c3.number_input("Budget Copertura (€)", value=5000.0, step=500.0)
        v_dist = c4.number_input("Distanza Barriera Min (%)", value=10.0, step=1.0)
        submit_adv = st.form_submit_button("🔍 Cerca Certificati Ottimali")
        
    if submit_adv:
        l_target = (v_p * v_b) / v_bud
        st.markdown("#### 2️⃣ Risultato Ottimizzazione")
        st.metric(label="🎯 Leva Target Calcolata", value=f"{l_target:.2f}x", help="Rapporto matematico necessario tra capitale da proteggere e budget allocato.")
        
        df_l = fetch_live_certificates()
        if not df_l.empty:
            col_d = 'Distanza Barriera %' if 'Distanza Barriera %' in df_l.columns else None
            df_res = df_l.copy()
            if col_d: df_res = df_res[df_res[col_d] >= v_dist]
            df_res['Diff_Leva'] = (df_res['Leva'] - l_target).abs()
            matches = df_res.sort_values('Diff_Leva').head(10)
            
            if matches.empty: 
                st.warning("Nessun certificato sul mercato rispetta questi parametri.")
            else:
                st.markdown("##### 🏆 Migliori Soluzioni di Mercato (Ordinati per vicinanza alla Leva Target)")
                st.dataframe(matches[['Sottostante', 'ISIN', 'Leva', 'Distanza Barriera %', 'Strike', 'Lettera']], use_container_width=True)
                st.info("👆 Copia l'ISIN desiderato o vai nella tab 'Database Live' per caricarlo nel motore di Setup.")
