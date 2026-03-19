import pandas as pd
import numpy as np
import plotly.graph_objects as go
from calculator import TurboParameters, DeterministicTurboCalculator
import copy

# ==============================================================================
# CHANGELOG v7.0
# ==============================================================================
# [FIX-4] generate_scenario_data ora è l'UNICA fonte di verità per scenari
#         (la matrice in app.py dovrà usare questa funzione, non una formula inline)
# [FIX-8] Prezzo futuro coerente con calculator (usa strike_adj)
# ==============================================================================

def generate_scenario_data(base_params: TurboParameters) -> tuple[pd.DataFrame, float]:
    """
    Genera 100 scenari da -30% a +30% sul sottostante.
    Ritorna (DataFrame, livello_barriera).
    """
    variations = np.linspace(-0.30, 0.30, 100)
    data = []
    
    base_calc = DeterministicTurboCalculator(base_params)
    base_res = base_calc.calculate_all()
    barriera = base_res['barriera']
    
    for var in variations:
        scenario_spot = base_params.valore_iniziale * (1 + var)
        
        p_scenario = copy.deepcopy(base_params)
        p_scenario.valore_ipotetico = scenario_spot
        # Disabilita la validazione strike > spot per scenari rialzisti
        # (lo spot potrebbe superare lo strike nello scenario)
        try:
            calc = DeterministicTurboCalculator(p_scenario)
            res = calc.calculate_all()
        except ValueError:
            # Scenario fuori dai limiti di validazione (spot > strike)
            res = base_calc.calculate_all()
            res['valore_copertura_simulata'] = 0.0
            res['valore_ptf_simulato'] = base_params.portafoglio * (1 + var * base_params.beta)
        
        is_ko = scenario_spot >= barriera
        
        if is_ko:
            valore_copertura = 0.0
            totale_simulato = res['valore_ptf_simulato']
        else:
            valore_copertura = res['valore_copertura_simulata']
            totale_simulato = res['valore_ptf_simulato'] + valore_copertura
            
        pl_netto = totale_simulato - res['totale_copertura']
        
        data.append({
            'Variazione Indice': var * 100,
            'Livello Indice': scenario_spot,
            'P&L Netto (€)': pl_netto,
            'Valore Turbo (€)': valore_copertura,
            'Valore Ptf Indifeso (€)': res['valore_ptf_simulato'] - base_params.portafoglio,
            'Knock-Out': is_ko
        })
        
    return pd.DataFrame(data), barriera


def generate_sensitivity_matrix(base_params: TurboParameters, base_res: dict) -> pd.DataFrame:
    """
    [FIX-4] Matrice di sensibilità che usa il calculator completo.
    Elimina la divergenza tra matrice e grafici.
    """
    var_list = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
    t_steps = sorted(list(set([0, max(1, int(base_params.giorni / 2)), base_params.giorni])))
    barriera = base_res['barriera']
    
    matrix = []
    for t in t_steps:
        row = []
        for v in var_list:
            s = base_params.valore_iniziale * (1 + v)
            
            if s >= barriera:
                row.append(0.0)
            else:
                p_scen = copy.deepcopy(base_params)
                p_scen.valore_ipotetico = s
                p_scen.giorni = t
                try:
                    res_scen = DeterministicTurboCalculator(p_scen).calculate_all()
                    row.append(max(0.0, res_scen['prezzo_futuro']))
                except ValueError:
                    row.append(0.0)
            
        matrix.append(row)
    
    df = pd.DataFrame(
        matrix, 
        columns=[f"{v*100:+.0f}%" for v in var_list], 
        index=[f"T+{t}gg" for t in t_steps]
    )
    return df


def plot_payoff_profile(df: pd.DataFrame, current_spot: float, barriera: float) -> go.Figure:
    """
    Grafico intuitivo a doppio livello:
    - SOPRA: Valore del portafoglio (€) con e senza copertura → l'utente vede 
      immediatamente "quanto vale il mio portafoglio" in ogni scenario
    - Area verde = zona dove la copertura ti protegge (guadagno rispetto a nudo)
    - Area rossa = zona dove la copertura ti costa (paghi il premio senza beneficio)
    - Banda rossa = zona knock-out
    """
    from plotly.subplots import make_subplots
    
    # Calcola valore portafoglio (non P&L) per leggibilità immediata
    ptf_base = df['Valore Ptf Indifeso (€)'].iloc[50]  # ~variazione 0% = punto centrale
    # Ricostruisci il valore assoluto del portafoglio
    # P&L nudo = ptf_simulato - portafoglio, quindi ptf_simulato = P&L nudo + portafoglio
    # Ma non abbiamo portafoglio qui — usiamo il fatto che a variazione 0% il P&L è ~0
    # Usiamo direttamente il P&L per confronto, ma con scala "€ dal capitale iniziale"
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
        subplot_titles=["📊 Quanto Ti Protegge la Copertura", "🛡️ Vantaggio Netto della Copertura"]
    )
    
    # === PANNELLO SUPERIORE: Confronto Ptf Nudo vs Coperto ===
    
    # Ptf Non Coperto (linea rossa tratteggiata — il rischio)
    fig.add_trace(go.Scatter(
        x=df['Variazione Indice'], y=df['Valore Ptf Indifeso (€)'],
        name='❌ Senza Copertura',
        line=dict(color='#C62828', width=2, dash='dash'),
        mode='lines',
        hovertemplate='Senza copertura: €%{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    
    # Ptf Coperto (linea blu solida — il risultato reale)
    fig.add_trace(go.Scatter(
        x=df['Variazione Indice'], y=df['P&L Netto (€)'],
        name='✅ Con Copertura (netto costi)',
        line=dict(color='#1A365D', width=3),
        mode='lines',
        hovertemplate='Con copertura: €%{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    
    # Area verde: dove la copertura protegge (coperto > nudo, cioè perdi meno)
    df_protect = df[df['P&L Netto (€)'] > df['Valore Ptf Indifeso (€)']].copy()
    if not df_protect.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([df_protect['Variazione Indice'], df_protect['Variazione Indice'][::-1]]),
            y=pd.concat([df_protect['P&L Netto (€)'], df_protect['Valore Ptf Indifeso (€)'][::-1]]),
            fill='toself', fillcolor='rgba(46,125,50,0.15)',
            line=dict(width=0), showlegend=True, name='🟢 Zona Protezione',
            hoverinfo='skip'
        ), row=1, col=1)
    
    # Area rossa: dove la copertura costa (nudo > coperto)
    df_cost = df[df['Valore Ptf Indifeso (€)'] > df['P&L Netto (€)']].copy()
    if not df_cost.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([df_cost['Variazione Indice'], df_cost['Variazione Indice'][::-1]]),
            y=pd.concat([df_cost['Valore Ptf Indifeso (€)'], df_cost['P&L Netto (€)'][::-1]]),
            fill='toself', fillcolor='rgba(198,40,40,0.10)',
            line=dict(width=0), showlegend=True, name='🔴 Costo Copertura',
            hoverinfo='skip'
        ), row=1, col=1)
    
    # Zona KO
    barriera_var = ((barriera - current_spot) / current_spot) * 100
    fig.add_vrect(
        x0=barriera_var, x1=df['Variazione Indice'].max(),
        fillcolor="red", opacity=0.12, layer="below", line_width=0,
        annotation_text="⚠️ KNOCK-OUT", annotation_position="top left",
        annotation_font_color="#C62828", annotation_font_size=11,
        row=1, col=1
    )
    
    # Linea zero e spot attuale
    fig.add_hline(y=0, line_color="#999999", line_width=1, line_dash="dot", row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="#2E7D32", 
                  annotation_text="Oggi", annotation_position="top right",
                  annotation_font_color="#2E7D32", row=1, col=1)
    
    # === PANNELLO INFERIORE: Vantaggio netto (differenza) ===
    vantaggio = df['P&L Netto (€)'] - df['Valore Ptf Indifeso (€)']
    
    colors = ['#2E7D32' if v >= 0 else '#C62828' for v in vantaggio]
    fig.add_trace(go.Bar(
        x=df['Variazione Indice'], y=vantaggio,
        name='Vantaggio Copertura',
        marker_color=colors, opacity=0.7,
        showlegend=False,
        hovertemplate='Vantaggio: €%{y:,.0f}<extra></extra>'
    ), row=2, col=1)
    
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="#2E7D32", row=2, col=1)
    
    # Zona KO anche nel pannello inferiore
    fig.add_vrect(
        x0=barriera_var, x1=df['Variazione Indice'].max(),
        fillcolor="red", opacity=0.12, layer="below", line_width=0,
        row=2, col=1
    )
    
    # Layout
    fig.update_xaxes(title_text="Variazione Indice (%)", row=2, col=1)
    fig.update_yaxes(title_text="P&L (€)", row=1, col=1)
    fig.update_yaxes(title_text="Vantaggio (€)", row=2, col=1)
    
    fig.update_layout(
        template='plotly_white', height=650,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=80, b=20)
    )

    return fig


def plot_pl_waterfall(res: dict) -> go.Figure:
    pl_ptf = res['pl_portafoglio']
    pl_turbo_lordo = res['pl_turbo_lordo']
    attriti = res['pl_turbo_netto'] - res['pl_turbo_lordo']
    pl_netto = pl_ptf + res['pl_turbo_netto']

    fig = go.Figure(go.Waterfall(
        name="P&L Breakdown", orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["P&L Ptf Nudo", "Gain Turbo (Lordo)", "Attriti (Spread/Fee)", "P&L Netto Finale"],
        textposition="outside",
        text=[f"€ {pl_ptf:,.0f}", f"€ {pl_turbo_lordo:,.0f}", f"€ {attriti:,.0f}", f"€ {pl_netto:,.0f}"],
        y=[pl_ptf, pl_turbo_lordo, attriti, 0],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#C62828"}},
        increasing={"marker": {"color": "#2E7D32"}},
        totals={"marker": {"color": "#1A365D"}}
    ))

    fig.update_layout(
        title="Scomposizione P&L: L'Emorragia dei Costi di Mercato",
        showlegend=False,
        template='plotly_white',
        height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig
