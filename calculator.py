import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List

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
# CHANGELOG v7.1 - MIGLIORAMENTI MODELLO AVANZATI
# ==============================================================================
# [LIM-1]  Monte Carlo: simulazione N percorsi GBM per distribuzione P&L,
#           VaR e CVaR. La volatilità viene stimata dal premio di mercato.
# [LIM-2]  Theta decay strutturale: il decay dipende dalla distanza dalla
#           barriera e dalla volatilità implicita, non solo dal tempo.
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
    # [LIM-1] Volatilità annualizzata (None = stima implicita dal premio)
    volatilita: float = None

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

    def _estimate_implied_volatility(self, premio: float, T: float) -> float:
        """
        [LIM-1] Stima la volatilità implicita dal premio di mercato.
        Il premio approssima il costo del gap risk, proporzionale a
        sigma * sqrt(T) * spot * multiplo / cambio.
        Inversione: sigma = premio / (sqrt(T) * spot * multiplo / cambio).
        """
        p = self.p
        if T <= 0 or premio <= 0:
            return 0.20
        denominatore = math.sqrt(T) * p.valore_iniziale * p.multiplo / p.cambio
        if denominatore <= 0:
            return 0.20
        sigma = premio / denominatore
        return max(0.05, min(1.50, sigma))

    def _theta_decay_structural(self, premio: float, T: float, sigma: float) -> float:
        """
        [LIM-2] Theta decay strutturale dipendente da:
        - Tempo residuo (T): decay accelera vicino a scadenza
        - Distanza dalla barriera: più vicino = decay più rapido (rischio KO)
        - Volatilità: alta vol = premio più resiliente (più valore temporale)

        Modello: premio_residuo = premio × sqrt(1-T) × exp(-lambda × T)
        dove lambda = k / (distanza_relativa × sigma)
        """
        p = self.p
        if p.giorni <= 0 or T >= 1.0:
            return 0.0

        tasso_netto = 1 - p.euribor + p.spread_emittente
        barriera = p.strike * math.pow(max(tasso_netto, 0.001), T)

        distanza_rel = (barriera - p.valore_iniziale) / p.valore_iniziale
        distanza_rel = max(distanza_rel, 0.005)

        sigma_safe = max(sigma, 0.05)
        lambda_decay = 0.5 / (distanza_rel * sigma_safe)
        lambda_decay = min(lambda_decay, 10.0)

        sqrt_component = math.sqrt(max(0, 1.0 - T))
        exp_component = math.exp(-lambda_decay * T)

        return premio * sqrt_component * exp_component

    def calculate_all(self) -> Dict[str, Any]:
        p = self.p
        T = self.safe_divide(p.giorni, 360)
        
        # --- Strike aggiustato per dividendi continui ---
        strike_adj = p.strike * math.exp(-p.dividend_yield * T)
        
        # --- Fair Value (valore intrinseco attuale) ---
        fair_value = max(0.0, (strike_adj - p.valore_iniziale) * p.multiplo / p.cambio)
        premio = max(0.0, p.prezzo_iniziale - fair_value)
        
        # --- [LIM-1] Volatilità: usa parametro utente o stima dal premio ---
        if p.volatilita is not None and p.volatilita > 0:
            sigma = p.volatilita
        else:
            sigma = self._estimate_implied_volatility(premio, T)
        
        # --- Leva implicita ---
        denominatore_leva = (p.prezzo_iniziale * p.cambio) / p.multiplo if p.multiplo > 0 else 0
        leva = self.safe_divide(p.valore_iniziale, denominatore_leva)

        # --- Barriera (Knock-Out) ---
        tasso_netto = 1 - p.euribor + p.spread_emittente
        barriera = p.strike * math.pow(max(tasso_netto, 0.001), T)
        
        # --- [FIX-8] Prezzo futuro: usa strike_adj come riferimento ---
        valore_intrinseco_futuro = (strike_adj - p.valore_ipotetico) * p.multiplo / p.cambio_futuro
        valore_intrinseco_futuro = max(0.0, valore_intrinseco_futuro)
        
        # --- [LIM-2] Premio residuo con decay strutturale ---
        premio_residuo = self._theta_decay_structural(premio, T, sigma)
        
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
            "sigma": sigma,
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
        """
        res = self.calculate_all()
        p = self.p
        
        costo_ingresso_pct = (p.bid_ask_spread / 2) + p.commissioni_pct
        costo_uscita_pct = (p.bid_ask_spread / 2) + p.commissioni_pct + p.stress_slippage
        
        res['n_turbo'] = float(n_custom)
        investimento_lordo = res['n_turbo'] * p.prezzo_iniziale
        res['investimento_lordo'] = investimento_lordo
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

    def run_monte_carlo(self, n_sim: int = 5000, seed: int = 42) -> Dict[str, Any]:
        """
        [LIM-1] Simulazione Monte Carlo con Geometric Brownian Motion.
        
        Genera n_sim percorsi del sottostante su T giorni, per ciascuno calcola:
        - Se il percorso tocca la barriera (knock-out path-dependent)
        - Il P&L netto finale del portafoglio coperto
        
        Restituisce distribuzione P&L, VaR, CVaR e probabilità di KO.
        """
        p = self.p
        res = self.calculate_all()
        sigma = res['sigma']
        T = self.safe_divide(p.giorni, 360)
        
        if T <= 0 or p.giorni <= 0:
            return {
                "mc_pl_distribution": [],
                "mc_var_95": 0.0, "mc_cvar_95": 0.0,
                "mc_prob_ko": 0.0, "mc_pl_mean": 0.0,
                "mc_pl_median": 0.0, "mc_pl_std": 0.0,
                "mc_percentiles": {}
            }
        
        rng = np.random.default_rng(seed)
        dt = T / p.giorni
        n_turbo = res['n_turbo']
        barriera = res['barriera']
        strike_adj = res['strike_adj']
        
        costo_uscita_pct = (p.bid_ask_spread / 2) + p.commissioni_pct + p.stress_slippage
        capitale_investito = res['capitale']
        
        mu = p.euribor - p.dividend_yield
        
        # Genera percorsi GBM (n_sim × giorni)
        z = rng.standard_normal((n_sim, p.giorni))
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
        spot_paths = p.valore_iniziale * np.exp(np.cumsum(log_returns, axis=1))
        
        spot_final = spot_paths[:, -1]
        spot_max = np.max(spot_paths, axis=1)
        ko_mask = spot_max >= barriera
        
        # P&L portafoglio
        var_idx = (spot_final - p.valore_iniziale) / p.valore_iniziale
        pl_ptf = p.portafoglio * var_idx * p.beta
        
        # P&L turbo
        valore_intrinseco = np.maximum(0, (strike_adj - spot_final) * p.multiplo / p.cambio_futuro)
        prezzo_finale = valore_intrinseco + res['premio_residuo']
        valore_cop_netta = prezzo_finale * n_turbo * (1 - costo_uscita_pct)
        pl_turbo = valore_cop_netta - capitale_investito
        pl_turbo = np.where(ko_mask, -capitale_investito, pl_turbo)
        
        pl_netto = pl_ptf + pl_turbo
        
        var_95 = float(np.percentile(pl_netto, 5))
        pl_sorted = np.sort(pl_netto)
        cvar_95 = float(np.mean(pl_sorted[pl_sorted <= var_95])) if np.any(pl_sorted <= var_95) else var_95
        
        return {
            "mc_pl_distribution": pl_netto.tolist(),
            "mc_var_95": var_95,
            "mc_cvar_95": cvar_95,
            "mc_prob_ko": float(np.mean(ko_mask)) * 100,
            "mc_pl_mean": float(np.mean(pl_netto)),
            "mc_pl_median": float(np.median(pl_netto)),
            "mc_pl_std": float(np.std(pl_netto)),
            "mc_percentiles": {
                "1%": float(np.percentile(pl_netto, 1)),
                "5%": float(np.percentile(pl_netto, 5)),
                "25%": float(np.percentile(pl_netto, 25)),
                "50%": float(np.percentile(pl_netto, 50)),
                "75%": float(np.percentile(pl_netto, 75)),
                "95%": float(np.percentile(pl_netto, 95)),
                "99%": float(np.percentile(pl_netto, 99)),
            }
        }
