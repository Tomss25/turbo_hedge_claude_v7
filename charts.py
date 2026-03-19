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
        try:
            calc = DeterministicTurboCalculator(p_scenario)
            res = calc.calculate_all()
        except ValueError:
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
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Livello Indice'], y=df['Valore Ptf Indifeso (€)'],
        name='Ptf Non Coperto', line=dict(color='gray', width=2, dash='dot'), mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=df['Livello Indice'], y=df['P&L Netto (€)'],
        name='P&L Netto (Coperto)', line=dict(color='#2c5282', width=3), mode='lines'
    ))

    fig.add_vrect(
        x0=barriera, x1=df['Livello Indice'].max(),
        fillcolor="red", opacity=0.15, layer="below", line_width=0,
        annotation_text="ZONA KNOCK-OUT (Perdita Premio)", 
        annotation_position="top left", annotation_font_color="red"
    )

    fig.add_vline(
        x=current_spot, line_dash="dash", line_color="green",
        annotation_text="Spot Attuale", annotation_position="bottom right"
    )

    fig.update_layout(
        title='Profilo di Rischio e Rendimento (P&L a Scadenza)',
        xaxis_title='Livello Indice', yaxis_title='Profitto / Perdita (€)',
        hovermode='x unified', template='plotly_white', height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.add_hline(y=0, line_color="black", line_width=1)

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
