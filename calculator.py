import math
from dataclasses import dataclass, field
from typing import Dict, Any

# ==============================================================================
# CHANGELOG v7.0 - FIX CRITICI E MIGLIORAMENTI MODELLO
# ==============================================================================
# [FIX-8]  Prezzo futuro: usa strike_adj (non barriera) come riferimento payoff
# [FIX-5]  Aggiunto modello theta-decay convesso (radice quadrata) 
# [FIX-14] Validazione input con eccezioni chiare
# [FIX-9]  Cambio dinamico: supporto cambio_futuro per scenari FX
# [NEW]    Due hedge ratio: "reale" e "commerciale" (per toggle UI)
# [NEW]    Metodo dedicato per override manuali (elimina bug app.py riga 134)
# ==============================================================================

@dataclass
class TurboParameters:
    prezzo_iniziale: float
    strike: float
    cambio: float
    multiplo: float
    euribor: float
    valore_iniziale: float
    valore_ipotetico: float
    giorni: int
    portafoglio: float
    beta: float = 1.0
    spread_emittente: float = 0.0056
    # Attriti di mercato
    dividend_yield: float = 0.015
    bid_ask_spread: float = 0.005
    commissioni_pct: float = 0.001
    stress_slippage: float = 0.0
    # [FIX-9] Cambio futuro per scenari FX (None = usa cambio corrente)
    cambio_futuro: float = None

    def __post_init__(self):
        """[FIX-14] Validazione degli input alla costruzione."""
        errors = []
        if self.cambio <= 0:
            errors.append("Cambio deve essere > 0")
        if self.multiplo <= 0:
            errors.append("Multiplo deve essere > 0")
        if self.prezzo_iniziale <= 0:
            errors.append("Prezzo Lettera deve essere > 0")
        if self.giorni < 0:
            errors.append("Giorni deve essere >= 0")
        if self.portafoglio <= 0:
            errors.append("Portafoglio deve essere > 0")
        if self.valore_iniziale <= 0:
            errors.append("Spot deve essere > 0")
        if self.strike <= self.valore_iniziale:
            errors.append(f"Strike ({self.strike}) deve essere > Spot ({self.valore_iniziale}) per un Turbo SHORT")
        if self.bid_ask_spread < 0 or self.commissioni_pct < 0:
            errors.append("Spread e commissioni non possono essere negativi")
        if errors:
            raise ValueError("Errori di validazione parametri:\n- " + "\n- ".join(errors))
        # Default cambio_futuro
        if self.cambio_futuro is None:
            self.cambio_futuro = self.cambio


class DeterministicTurboCalculator:
    def __init__(self, p: TurboParameters):
        self.p = p

    @staticmethod
    def safe_divide(numerator: float, denominator: float) -> float:
        return 0.0 if denominator == 0 else numerator / denominator

    def calculate_all(self) -> Dict[str, Any]:
        p = self.p
        T = self.safe_divide(p.giorni, 360)
        
        # --- Strike aggiustato per dividendi continui ---
        strike_adj = p.strike * math.exp(-p.dividend_yield * T)
        
        # --- Fair Value (valore intrinseco attuale) ---
        fair_value = max(0.0, (strike_adj - p.valore_iniziale) * p.multiplo / p.cambio)
        premio = max(0.0, p.prezzo_iniziale - fair_value)
        
        # --- Leva implicita ---
        denominatore_leva = (p.prezzo_iniziale * p.cambio) / p.multiplo if p.multiplo > 0 else 0
        leva = self.safe_divide(p.valore_iniziale, denominatore_leva)

        # --- Barriera (Knock-Out) ---
        tasso_netto = 1 - p.euribor + p.spread_emittente
        barriera = p.strike * math.pow(max(tasso_netto, 0.001), T)
        
        # --- [FIX-8] Prezzo futuro: usa strike_adj come riferimento, NON barriera ---
        # La barriera è il livello di knock-out, non il livello di payoff.
        # Il payoff di un Turbo Short è (strike - spot_target), non (barriera - spot_target).
        valore_intrinseco_futuro = (strike_adj - p.valore_ipotetico) * p.multiplo / p.cambio_futuro
        valore_intrinseco_futuro = max(0.0, valore_intrinseco_futuro)
        
        # [FIX-5] Premio residuo con decay convesso (radice quadrata del tempo)
        # Theta decay reale: il premio si erode più rapidamente vicino a scadenza.
        # sqrt(tempo_residuo/tempo_totale) approssima il comportamento convesso.
        if p.giorni > 0:
            premio_residuo = premio * math.sqrt(max(0, 1.0 - T))
        else:
            premio_residuo = 0.0
        
        prezzo_futuro = valore_intrinseco_futuro + premio_residuo

        # --- Dimensionamento ---
        esposizione_pesata = p.portafoglio * p.beta
        n_turbo = self.safe_divide(esposizione_pesata, leva * p.prezzo_iniziale) if leva > 0 and p.prezzo_iniziale > 0 else 0
        
        # --- Costi di transazione ---
        costo_ingresso_pct = (p.bid_ask_spread / 2) + p.commissioni_pct
        costo_uscita_pct = (p.bid_ask_spread / 2) + p.commissioni_pct + p.stress_slippage
        
        investimento_lordo = n_turbo * p.prezzo_iniziale
        capitale = investimento_lordo * (1 + costo_ingresso_pct)
        totale_copertura = p.portafoglio + capitale
        
        # --- Simulazione Portafoglio ---
        var_indice = self.safe_divide(p.valore_ipotetico - p.valore_iniziale, p.valore_iniziale)
        var_ptf = var_indice * p.beta
        valore_ptf_simulato = p.portafoglio * (1 + var_ptf)
        
        # --- Valore copertura netto ---
        valore_copertura_lorda = prezzo_futuro * n_turbo
        valore_copertura_netta = valore_copertura_lorda * (1 - costo_uscita_pct)
        
        totale_simulato = valore_ptf_simulato + valore_copertura_netta
        percentuale = self.safe_divide(totale_simulato - totale_copertura, totale_copertura)

        # --- P&L e Hedge Ratios ---
        pl_portafoglio = valore_ptf_simulato - p.portafoglio
        pl_turbo_lordo = valore_copertura_lorda - investimento_lordo
        pl_turbo_netto = valore_copertura_netta - capitale
        
        hedge_ratio_reale = self.safe_divide(pl_turbo_netto, abs(pl_portafoglio)) if pl_portafoglio < 0 else 0.0
        hedge_ratio_commerciale = self.safe_divide(pl_turbo_lordo, abs(pl_portafoglio)) if pl_portafoglio < 0 else 0.0

        return {
            "fair_value": fair_value,
            "premio": premio,
            "premio_residuo": premio_residuo,
            "leva": leva,
            "barriera": barriera,
            "strike_adj": strike_adj,
            "prezzo_futuro": prezzo_futuro,
            "n_turbo": n_turbo,
            "investimento_lordo": investimento_lordo,
            "capitale": capitale,
            "totale_copertura": totale_copertura,
            "valore_ptf_simulato": valore_ptf_simulato,
            "valore_copertura_simulata": valore_copertura_netta,
            "valore_copertura_lorda": valore_copertura_lorda,
            "totale_simulato": totale_simulato,
            "percentuale": percentuale,
            "pl_portafoglio": pl_portafoglio,
            "pl_turbo_netto": pl_turbo_netto,
            "pl_turbo_lordo": pl_turbo_lordo,
            "hedge_ratio_reale": hedge_ratio_reale,
            "hedge_ratio_commerciale": hedge_ratio_commerciale
        }

    def override_manual_quantity(self, n_custom: int) -> Dict[str, Any]:
        """
        [FIX-1] Ricalcola con quantità manuale SENZA il bug del capitale.
        In modalità manuale il capitale è SOLO il costo dei turbo,
        non portafoglio + costo turbo.
        """
        res = self.calculate_all()
        p = self.p
        
        costo_ingresso_pct = (p.bid_ask_spread / 2) + p.commissioni_pct
        costo_uscita_pct = (p.bid_ask_spread / 2) + p.commissioni_pct + p.stress_slippage
        
        res['n_turbo'] = float(n_custom)
        investimento_lordo = res['n_turbo'] * p.prezzo_iniziale
        res['investimento_lordo'] = investimento_lordo
        # [FIX-1] Capitale = solo costo turbo (non ptf + costo turbo)
        res['capitale'] = investimento_lordo * (1 + costo_ingresso_pct)
        res['totale_copertura'] = p.portafoglio + res['capitale']
        
        res['valore_copertura_lorda'] = res['n_turbo'] * res['prezzo_futuro']
        res['valore_copertura_simulata'] = res['valore_copertura_lorda'] * (1 - costo_uscita_pct)
        res['totale_simulato'] = res['valore_ptf_simulato'] + res['valore_copertura_simulata']
        res['percentuale'] = self.safe_divide(
            res['totale_simulato'] - res['totale_copertura'], 
            res['totale_copertura']
        )
        
        res['pl_turbo_lordo'] = res['valore_copertura_lorda'] - investimento_lordo
        res['pl_turbo_netto'] = res['valore_copertura_simulata'] - res['capitale']
        
        pl_ptf = res['pl_portafoglio']
        res['hedge_ratio_reale'] = self.safe_divide(res['pl_turbo_netto'], abs(pl_ptf)) if pl_ptf < 0 else 0.0
        res['hedge_ratio_commerciale'] = self.safe_divide(res['pl_turbo_lordo'], abs(pl_ptf)) if pl_ptf < 0 else 0.0
        
        return res
