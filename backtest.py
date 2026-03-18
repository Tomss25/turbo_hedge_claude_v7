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

def run_historical_backtest(
    ticker_ptf: str, 
    ticker_idx: str, 
    ticker_fx: str, 
    start: str, 
    end: str, 
    livello_barriera: float,
    # [FIX-16] Soglie parametrizzabili
    drawdown_entry: float = -0.05,
    drawdown_exit: float = -0.02
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
                "color": "#C62828",       # Rosso scuro per testo
                "bg_color": "#FFEBEE",    # Rosso chiaro per sfondo
                "severity": "error"
            }
        elif perc_copertura > 40:
            diagnosis = {
                "title": "SOTTOEFFICIENZA (Cash Drag)",
                "body": f"Portafoglio coperto per il {perc_copertura:.1f}% del tempo. Eccessiva esposizione ai costi del derivato. Max Drawdown: {max_dd:.1f}%.",
                "action": "AZIONE CORRETTIVA: Rivedi i trigger di ingresso (attualmente attivazione a {:.0f}%).".format(drawdown_entry * 100),
                "color": "#E65100",       # Arancione scuro
                "bg_color": "#FFF3E0",    # Arancione chiaro
                "severity": "warning"
            }
        else:
            diagnosis = {
                "title": "ESITO POSITIVO (Copertura Tattica Ottimale)",
                "body": f"Nessun Knock-Out. Permanenza a mercato chirurgica ({perc_copertura:.1f}% del tempo). Max Drawdown: {max_dd:.1f}%.",
                "action": "RISOLUZIONE: I parametri strutturali sono solidi.",
                "color": "#2E7D32",       # Verde scuro
                "bg_color": "#E8F5E9",    # Verde chiaro
                "severity": "success"
            }

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
    pdf.cell(0, 10, "Turbo Hedge Quant - Report di Backtest (v7.0)", ln=True, align="C")
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
    pdf.ln(10)
    
    # [FIX-3] Usa severity per il colore nel PDF
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
