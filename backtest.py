import yfinance as yf
import pandas as pd
import numpy as np

# ==============================================================================
# CHANGELOG v7.0
# ==============================================================================
# [FIX-3]  Diagnosi con colori esadecimali corretti (bg_color + color)
# [FIX-16] Soglie di attivazione parametrizzabili (non più hardcoded)
# [FIX-17] Avviso se ticker indice != sottostante del certificato
# ==============================================================================
# CHANGELOG v7.1 - MIGLIORAMENTI MODELLO AVANZATI
# ==============================================================================
# [LIM-5]  Beta dinamico: il numero di turbo viene ribilanciato giornalmente
#           in base al beta rolling 60gg. N_turbo_t = N_base × Beta_60d_t.
# [LIM-6]  Backtest quantitativo: simulazione P&L giornaliera effettiva
#           della copertura con tracking cumulativo, costi di ribilanciamento
#           e P&L turbo calcolato dal movimento reale dell'indice.
# ==============================================================================

def run_historical_backtest(
    ticker_ptf: str, 
    ticker_idx: str, 
    ticker_fx: str, 
    start: str, 
    end: str, 
    livello_barriera: float,
    # [FIX-16] Soglie parametrizzabili
    drawdown_entry: float = -0.05,
    drawdown_exit: float = -0.02,
    # [LIM-6] Parametri per simulazione P&L
    n_turbo_base: float = 0.0,
    prezzo_turbo_iniziale: float = 0.0,
    strike: float = 0.0,
    multiplo: float = 0.01,
    cambio: float = 1.15,
    bid_ask_spread: float = 0.005,
    commissioni_pct: float = 0.001
):
    try:
        ptf_data = yf.download(ticker_ptf, start=start, end=end, progress=False)['Close']
        idx_data = yf.download(ticker_idx, start=start, end=end, progress=False)[['Close', 'High']]
        
        if ptf_data.empty or idx_data.empty:
            return None, "Dati Portafoglio o Indice non trovati.", None
            
        df = pd.DataFrame({
            'Ptf_Close': ptf_data.squeeze(),
            'Idx_Close': idx_data['Close'].squeeze(),
            'Idx_High': idx_data['High'].squeeze()
        }).dropna()

        # INTEGRAZIONE FX RISK
        if ticker_fx and ticker_fx.strip() != "":
            fx_data = yf.download(ticker_fx, start=start, end=end, progress=False)['Close']
            if not fx_data.empty:
                df = df.join(fx_data.rename('FX_Close'), how='inner')
                df['Ptf_Base_Currency'] = df['Ptf_Close'] / df['FX_Close']
            else:
                df['Ptf_Base_Currency'] = df['Ptf_Close']
        else:
            df['Ptf_Base_Currency'] = df['Ptf_Close']

        # Beta rolling 60 giorni
        df['R_ptf'] = df['Ptf_Close'].pct_change()
        df['R_idx'] = df['Idx_Close'].pct_change()
        cov_60d = df['R_ptf'].rolling(window=60).cov(df['R_idx'])
        var_60d = df['R_idx'].rolling(window=60).var()
        df['Beta_60d'] = (cov_60d / var_60d).fillna(1.0)
        # Clip beta a valori ragionevoli per evitare leve degeneri
        df['Beta_60d'] = df['Beta_60d'].clip(0.1, 3.0)
        
        # Drawdown su valuta base
        df['Peak'] = df['Ptf_Base_Currency'].cummax()
        df['Drawdown'] = (df['Ptf_Base_Currency'] - df['Peak']) / df['Peak']
        
        # Knock-Out su massimi intraday
        df['Knock_Out_Event'] = np.where(df['Idx_High'] >= livello_barriera, 1, 0)
        
        # [FIX-16] Segnali con soglie parametriche
        df['Hedge_Signal'] = np.where(df['Drawdown'] < drawdown_entry, 1, 0)
        df['Hedge_Signal'] = np.where(
            (df['Drawdown'] > drawdown_exit) | (df['Knock_Out_Event'] == 1), 
            0, df['Hedge_Signal']
        )
        df['Hedge_Signal'] = df['Hedge_Signal'].ffill().fillna(0)
        df['Hedge_Signal'] = np.where(df['Knock_Out_Event'] == 1, 0, df['Hedge_Signal'])

        # ==================================================================
        # [LIM-5] BETA DINAMICO: N_turbo ribilanciato con beta rolling
        # [LIM-6] SIMULAZIONE P&L GIORNALIERA EFFETTIVA
        # ==================================================================
        simulate_pl = (n_turbo_base > 0 and strike > 0 and prezzo_turbo_iniziale > 0)
        
        if simulate_pl:
            costo_half_spread = bid_ask_spread / 2
            costo_transazione = costo_half_spread + commissioni_pct
            
            # [LIM-5] Numero turbo pesato per beta rolling
            # Quando beta sale, servono più turbo; quando scende, meno.
            df['N_Turbo_Dynamic'] = n_turbo_base * df['Beta_60d']
            
            # Variazione giornaliera dell'indice
            df['Idx_Return'] = df['Idx_Close'].pct_change().fillna(0)
            
            # Prezzo teorico turbo giornaliero:
            # Per un Turbo Short, il prezzo = max(0, (strike - idx) * multiplo / cambio)
            df['Turbo_Price'] = np.maximum(0, (strike - df['Idx_Close']) * multiplo / cambio)
            
            # P&L giornaliero del turbo: variazione prezzo × N turbo × segnale
            df['Turbo_Price_Change'] = df['Turbo_Price'].diff().fillna(0)
            
            # [LIM-6] P&L giornaliero effettivo della copertura
            # Solo quando Hedge_Signal = 1 e non c'è KO
            df['Daily_PL_Turbo'] = df['Turbo_Price_Change'] * df['N_Turbo_Dynamic'] * df['Hedge_Signal']
            
            # Costi di ribilanciamento: quando N_turbo cambia o il segnale si attiva/disattiva
            df['N_Turbo_Prev'] = (df['N_Turbo_Dynamic'] * df['Hedge_Signal']).shift(1).fillna(0)
            df['N_Turbo_Curr'] = df['N_Turbo_Dynamic'] * df['Hedge_Signal']
            df['Delta_N'] = (df['N_Turbo_Curr'] - df['N_Turbo_Prev']).abs()
            # Costo di ribilanciamento = |delta_N| × prezzo_turbo × costo_transazione
            df['Rebalancing_Cost'] = df['Delta_N'] * df['Turbo_Price'] * costo_transazione
            
            # Se KO: perdita totale del valore dei turbo in quel momento
            # Il giorno del KO, il turbo va a zero
            ko_days = df['Knock_Out_Event'] == 1
            prev_hedge = df['Hedge_Signal'].shift(1).fillna(0)
            # Perdita KO solo se eravamo in hedge il giorno prima
            df['KO_Loss'] = 0.0
            ko_with_hedge = ko_days & (prev_hedge == 1)
            if ko_with_hedge.any():
                turbo_price_prev = df['Turbo_Price'].shift(1).fillna(0)
                n_turbo_prev = df['N_Turbo_Dynamic'].shift(1).fillna(0)
                df.loc[ko_with_hedge, 'KO_Loss'] = -(turbo_price_prev * n_turbo_prev).loc[ko_with_hedge]
            
            # P&L netto giornaliero = P&L turbo - costi ribilanciamento + eventuali perdite KO
            df['Daily_PL_Net'] = df['Daily_PL_Turbo'] - df['Rebalancing_Cost'] + df['KO_Loss']
            
            # Cumulativo
            df['Cumulative_PL_Turbo'] = df['Daily_PL_Net'].cumsum()
            
            # P&L portafoglio giornaliero (per confronto)
            df['Daily_PL_Ptf'] = df['Ptf_Base_Currency'].pct_change().fillna(0) * df['Ptf_Base_Currency'].shift(1).fillna(df['Ptf_Base_Currency'].iloc[0])
            df['Cumulative_PL_Ptf'] = (df['Ptf_Base_Currency'] - df['Ptf_Base_Currency'].iloc[0])
            
            # P&L combinato (portafoglio + copertura)
            df['Cumulative_PL_Combined'] = df['Cumulative_PL_Ptf'] + df['Cumulative_PL_Turbo']
            
            # Costi totali di ribilanciamento
            total_rebalancing_cost = df['Rebalancing_Cost'].sum()
            total_turbo_pl = df['Daily_PL_Net'].sum()
        else:
            # Se non ci sono parametri per la simulazione, colonne a zero
            df['N_Turbo_Dynamic'] = 0.0
            df['Daily_PL_Turbo'] = 0.0
            df['Rebalancing_Cost'] = 0.0
            df['KO_Loss'] = 0.0
            df['Daily_PL_Net'] = 0.0
            df['Cumulative_PL_Turbo'] = 0.0
            df['Cumulative_PL_Ptf'] = 0.0
            df['Cumulative_PL_Combined'] = 0.0
            total_rebalancing_cost = 0.0
            total_turbo_pl = 0.0

        # Metriche
        giorni_totali = len(df)
        giorni_coperti = df['Hedge_Signal'].sum()
        numero_ko = df['Knock_Out_Event'].sum()
        perc_copertura = (giorni_coperti / giorni_totali) * 100 if giorni_totali > 0 else 0
        max_dd = df['Drawdown'].min() * 100
        
        # [FIX-3] Colori esadecimali per il rendering HTML in app.py
        if numero_ko > 0:
            diagnosis = {
                "title": "FALLIMENTO STRUTTURALE (Rischio Rovina)",
                "body": f"La simulazione ha registrato {numero_ko} eventi di Knock-Out sui massimi intraday. Max Drawdown: {max_dd:.1f}%.",
                "action": "AZIONE CORRETTIVA: Allontana lo Strike o controlla il rischio di cambio.",
                "color": "#C62828",
                "bg_color": "#FFEBEE",
                "severity": "error"
            }
        elif perc_copertura > 40:
            diagnosis = {
                "title": "SOTTOEFFICIENZA (Cash Drag)",
                "body": f"Portafoglio coperto per il {perc_copertura:.1f}% del tempo. Eccessiva esposizione ai costi del derivato. Max Drawdown: {max_dd:.1f}%.",
                "action": "AZIONE CORRETTIVA: Rivedi i trigger di ingresso (attualmente attivazione a {:.0f}%).".format(drawdown_entry * 100),
                "color": "#E65100",
                "bg_color": "#FFF3E0",
                "severity": "warning"
            }
        else:
            diagnosis = {
                "title": "ESITO POSITIVO (Copertura Tattica Ottimale)",
                "body": f"Nessun Knock-Out. Permanenza a mercato chirurgica ({perc_copertura:.1f}% del tempo). Max Drawdown: {max_dd:.1f}%.",
                "action": "RISOLUZIONE: I parametri strutturali sono solidi.",
                "color": "#2E7D32",
                "bg_color": "#E8F5E9",
                "severity": "success"
            }
        
        # [LIM-6] Aggiungi metriche quantitative alla diagnosi
        if simulate_pl:
            diagnosis["body"] += f" P&L Copertura Cumulativo: € {total_turbo_pl:,.0f}. Costi Ribilanciamento Totali: € {total_rebalancing_cost:,.0f}."

        return df.reset_index(), "Successo", diagnosis
        
    except Exception as e:
        return None, str(e), None


def generate_pdf_report(
    df: pd.DataFrame, ticker_ptf: str, ticker_idx: str, 
    ticker_fx: str, barriera: float, diagnosis: dict
) -> bytes:
    import tempfile
    import os
    from fpdf import FPDF
    
    giorni_totali = len(df)
    giorni_coperti = df['Hedge_Signal'].sum()
    percentuale_copertura = (giorni_coperti / giorni_totali) * 100
    numero_ko = df['Knock_Out_Event'].sum()
    max_dd = df['Drawdown'].min() * 100
    fx_note = f" (Aggiustato per rischio cambio su {ticker_fx})" if ticker_fx else " (Nessun rischio cambio inserito)"
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 10, "Turbo Hedge Quant - Report di Backtest (v7.1)", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Asset Analizzati: Portafoglio [{ticker_ptf}] vs Indice [{ticker_idx}]", ln=True)
    pdf.cell(0, 8, f"Periodo Analizzato: {df['Date'].dt.date.iloc[0]} -> {df['Date'].dt.date.iloc[-1]}", ln=True)
    pdf.cell(0, 8, f"Livello Barriera Sotto Stress: {barriera:.2f}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 10, "Metriche Storiche", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Max Drawdown Storico{fx_note}: {max_dd:.2f}%", ln=True)
    pdf.cell(0, 8, f"Tempo trascorso in copertura: {percentuale_copertura:.1f}% del periodo", ln=True)
    pdf.cell(0, 8, f"Eventi di Knock-Out Registrati (Massimi Intraday): {numero_ko} eventi", ln=True)
    pdf.cell(0, 8, f"Beta Medio 60gg: {df['Beta_60d'].mean():.2f} (Min: {df['Beta_60d'].min():.2f}, Max: {df['Beta_60d'].max():.2f})", ln=True)
    
    # [LIM-6] Metriche P&L se disponibili
    if 'Cumulative_PL_Turbo' in df.columns and df['Cumulative_PL_Turbo'].abs().sum() > 0:
        pl_finale = df['Cumulative_PL_Turbo'].iloc[-1]
        costi_reb = df['Rebalancing_Cost'].sum()
        pdf.cell(0, 8, f"P&L Cumulativo Copertura: EUR {pl_finale:,.2f}", ln=True)
        pdf.cell(0, 8, f"Costi Totali Ribilanciamento: EUR {costi_reb:,.2f}", ln=True)
    
    pdf.ln(10)
    
    severity = diagnosis.get('severity', 'success')
    if severity == 'error':
        pdf.set_text_color(180, 0, 0)
    elif severity == 'warning':
        pdf.set_text_color(200, 100, 0)
    else:
        pdf.set_text_color(0, 120, 0)
        
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"DIAGNOSI: {diagnosis['title']}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, txt=diagnosis['body'])
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(0, 6, txt=diagnosis['action'])
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
    pdf.output(tmp_path)
    with open(tmp_path, "rb") as f:
        pdf_bytes = f.read()
    os.remove(tmp_path)
    return pdf_bytes
