"""
simulate.py
===========
THANOS Network Simulation Engine

Monte Carlo simulation for supply chain network planning under uncertainty.

Core idea: supply chain planners need to evaluate how proposed routing
strategies perform across the *distribution* of possible futures —
not just the expected case. A plan that looks optimal at the mean
may have catastrophic tail risk under peak demand or capacity disruption.

This module:
  1. Models stochastic demand and capacity per lane
  2. Runs N simulation trials per scenario
  3. Computes outcome distributions (cost, volume routed, SLA compliance, spillover)
  4. Compares strategies: CONSTRAINED vs UNCONSTRAINED capacity scenarios
  5. Identifies lanes at risk: P95 demand > capacity

Usage
-----
    python simulate.py                          # full simulation, all scenarios
    python simulate.py --trials 2000            # more trials for tighter CIs
    python simulate.py --scenario constrained   # single scenario
    python simulate.py --peak                   # apply peak demand multiplier
"""

import argparse
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_TRIALS    = 1000     # simulation runs per scenario
PEAK_MULTIPLIER   = 1.8      # demand scaling for peak season scenario
SLA_TRANSIT_MAX   = 3.0      # days — shipments beyond this are SLA failures
SPILLOVER_COST    = 25.0     # cost per unit of unmet demand (expediting premium)
CONSTRAINED_CAP   = 1.0      # use stated capacity as hard ceiling
UNCONSTRAINED_CAP = 1.5      # allow 50% surge capacity (flex staffing, etc.)

# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    scenario:          str
    trial:             int
    lane_id:           str
    origin:            str
    destination:       str
    simulated_demand:  float
    routed_volume:     float
    spillover:         float
    lane_cost:         float
    transit_days:      float
    sla_breach:        int       # 1 if transit > SLA threshold
    capacity_utilized: float     # routed / capacity

@dataclass
class ScenarioSummary:
    scenario:            str
    total_cost_mean:     float
    total_cost_p25:      float
    total_cost_p75:      float
    total_cost_p95:      float
    volume_routed_mean:  float
    spillover_mean:      float
    spillover_p95:       float
    sla_breach_rate:     float
    lanes_at_risk:       int     # lanes where P95 demand > capacity


# ── Demand Sampling ───────────────────────────────────────────────────────────

def sample_demand(base: float, cv: float, n: int,
                  peak: bool = False) -> np.ndarray:
    """
    Sample demand from a log-normal distribution.

    Log-normal is appropriate for demand:
      - Right-skewed (occasional extreme peaks)
      - Strictly positive
      - CV parameterised directly

    peak=True applies the PEAK_MULTIPLIER to the base before sampling —
    simulating festive season / Prime Day demand surge.
    """
    mu    = base * (PEAK_MULTIPLIER if peak else 1.0)
    sigma = cv * mu
    # Log-normal parameters from mean and std
    log_var   = np.log(1 + (sigma / mu) ** 2)
    log_mean  = np.log(mu) - log_var / 2
    return np.random.lognormal(log_mean, np.sqrt(log_var), n)


def sample_transit(mean: float, std: float, n: int) -> np.ndarray:
    """
    Sample transit times from a truncated normal (transit ≥ 0.1 days).
    Disruption events occasionally spike transit — modelled via contamination:
    5% of trials draw from a heavier-tailed distribution.
    """
    base   = np.random.normal(mean, std, n)
    # Disruption contamination: 5% of shipments experience delays
    disrupt_mask = np.random.uniform(0, 1, n) < 0.05
    base[disrupt_mask] += np.random.exponential(1.0, disrupt_mask.sum())
    return np.maximum(base, 0.1)


# ── Single Trial ──────────────────────────────────────────────────────────────

def run_trial(lanes_df: pd.DataFrame, scenario: str,
              peak: bool, trial: int) -> List[SimulationResult]:
    """
    One Monte Carlo trial: sample demand and transit for every lane,
    apply capacity constraint, compute costs.
    """
    cap_factor = CONSTRAINED_CAP if scenario == "constrained" else UNCONSTRAINED_CAP
    results    = []

    for _, lane in lanes_df.iterrows():
        demand   = float(sample_demand(lane.base_demand, lane.demand_cv, 1, peak)[0])
        transit  = float(sample_transit(lane.transit_mean, lane.transit_std, 1)[0])

        capacity      = lane.capacity * cap_factor
        routed        = min(demand, capacity)
        spillover     = max(0.0, demand - capacity)

        # Cost = routing cost + spillover penalty
        routing_cost  = routed * lane.cost_per_unit
        spill_cost    = spillover * SPILLOVER_COST
        total_cost    = routing_cost + spill_cost

        cap_util      = routed / lane.capacity  # utilisation vs stated capacity

        results.append(SimulationResult(
            scenario          = scenario,
            trial             = trial,
            lane_id           = lane.lane_id,
            origin            = lane.origin,
            destination       = lane.destination,
            simulated_demand  = round(demand, 1),
            routed_volume     = round(routed, 1),
            spillover         = round(spillover, 1),
            lane_cost         = round(total_cost, 2),
            transit_days      = round(transit, 2),
            sla_breach        = int(transit > SLA_TRANSIT_MAX),
            capacity_utilized = round(cap_util, 3),
        ))

    return results


# ── Full Monte Carlo ──────────────────────────────────────────────────────────

def run_monte_carlo(lanes_df: pd.DataFrame, scenario: str,
                    n_trials: int, peak: bool) -> pd.DataFrame:
    """
    Run n_trials independent simulations for a given scenario.
    Returns a long-format DataFrame of all trial results.
    """
    all_results = []
    for t in range(n_trials):
        trial_results = run_trial(lanes_df, scenario, peak, t)
        all_results.extend(trial_results)

    rows = [vars(r) for r in all_results]
    return pd.DataFrame(rows)


# ── Scenario Summarisation ────────────────────────────────────────────────────

def summarise_scenario(sim_df: pd.DataFrame, lanes_df: pd.DataFrame,
                       scenario: str) -> ScenarioSummary:
    """
    Aggregate trial-level results into network-level outcome distribution.
    Key outputs: cost distribution (mean, P25, P75, P95), spillover, SLA rate.
    """
    # Sum across lanes per trial to get network-level outcomes
    trial_agg = sim_df.groupby("trial").agg(
        total_cost    = ("lane_cost",    "sum"),
        total_volume  = ("routed_volume","sum"),
        total_spill   = ("spillover",    "sum"),
        total_breach  = ("sla_breach",   "sum"),
        total_lanes   = ("lane_id",      "count"),
    ).reset_index()

    trial_agg["sla_breach_rate"] = trial_agg["total_breach"] / trial_agg["total_lanes"]

    # Identify lanes at structural risk: P95 simulated demand > capacity
    lane_p95 = sim_df.groupby("lane_id")["simulated_demand"].quantile(0.95).reset_index()
    lane_p95.columns = ["lane_id", "p95_demand"]
    lane_p95 = lane_p95.merge(lanes_df[["lane_id", "capacity"]], on="lane_id")
    at_risk  = (lane_p95["p95_demand"] > lane_p95["capacity"]).sum()

    return ScenarioSummary(
        scenario            = scenario,
        total_cost_mean     = round(trial_agg["total_cost"].mean(), 0),
        total_cost_p25      = round(trial_agg["total_cost"].quantile(0.25), 0),
        total_cost_p75      = round(trial_agg["total_cost"].quantile(0.75), 0),
        total_cost_p95      = round(trial_agg["total_cost"].quantile(0.95), 0),
        volume_routed_mean  = round(trial_agg["total_volume"].mean(), 0),
        spillover_mean      = round(trial_agg["total_spill"].mean(), 0),
        spillover_p95       = round(trial_agg["total_spill"].quantile(0.95), 0),
        sla_breach_rate     = round(trial_agg["sla_breach_rate"].mean(), 4),
        lanes_at_risk       = int(at_risk),
    )


# ── Lane Risk Report ──────────────────────────────────────────────────────────

def lane_risk_report(sim_df: pd.DataFrame, lanes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-lane risk summary:
      - Mean / P50 / P95 demand
      - Capacity utilisation mean and P95
      - Spillover rate (% of trials with any spillover)
      - SLA breach rate
      - Risk flag: HIGH / MEDIUM / LOW
    """
    lane_stats = sim_df.groupby("lane_id").agg(
        demand_mean       = ("simulated_demand",  "mean"),
        demand_p50        = ("simulated_demand",  lambda x: x.quantile(0.50)),
        demand_p95        = ("simulated_demand",  lambda x: x.quantile(0.95)),
        util_mean         = ("capacity_utilized", "mean"),
        util_p95          = ("capacity_utilized", lambda x: x.quantile(0.95)),
        spillover_rate    = ("spillover",         lambda x: (x > 0).mean()),
        spillover_p95     = ("spillover",         lambda x: x.quantile(0.95)),
        sla_breach_rate   = ("sla_breach",        "mean"),
        cost_mean         = ("lane_cost",         "mean"),
    ).reset_index().round(3)

    lane_stats = lane_stats.merge(
        lanes_df[["lane_id", "origin", "destination", "capacity",
                  "cost_per_unit", "transit_mean"]], on="lane_id"
    )

    def risk_flag(row):
        if row["util_p95"] > 0.95 or row["spillover_rate"] > 0.20:
            return "HIGH"
        elif row["util_p95"] > 0.80 or row["spillover_rate"] > 0.10:
            return "MEDIUM"
        return "LOW"

    lane_stats["risk_flag"] = lane_stats.apply(risk_flag, axis=1)
    return lane_stats.sort_values("spillover_rate", ascending=False)


# ── Scenario Comparison ───────────────────────────────────────────────────────

def compare_scenarios(summaries: List[ScenarioSummary]) -> pd.DataFrame:
    """
    Side-by-side scenario comparison table.
    This is the output planners use to choose between routing strategies.
    """
    rows = []
    for s in summaries:
        rows.append({
            "Scenario":              s.scenario,
            "Cost Mean (₹)":         f"{s.total_cost_mean:,.0f}",
            "Cost P25 (₹)":          f"{s.total_cost_p25:,.0f}",
            "Cost P75 (₹)":          f"{s.total_cost_p75:,.0f}",
            "Cost P95 (₹)":          f"{s.total_cost_p95:,.0f}",
            "Cost P75-P25 Range":    f"{s.total_cost_p75 - s.total_cost_p25:,.0f}",
            "Volume Routed (mean)":  f"{s.volume_routed_mean:,.0f}",
            "Spillover Mean":        f"{s.spillover_mean:,.0f}",
            "Spillover P95":         f"{s.spillover_p95:,.0f}",
            "SLA Breach Rate":       f"{s.sla_breach_rate:.1%}",
            "Lanes at Risk":         s.lanes_at_risk,
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_simulation(n_trials: int = DEFAULT_TRIALS,
                   scenario_filter: str = "all",
                   peak: bool = False):

    from generate_network import build_lanes_df
    lanes_df = build_lanes_df()

    mode_tag  = "PEAK" if peak else "NORMAL"
    scenarios = ["constrained", "unconstrained"] \
                if scenario_filter == "all" else [scenario_filter]

    print(f"\n{'='*65}")
    print(f"THANOS — Monte Carlo Network Simulation")
    print(f"Demand mode: {mode_tag}   |   Trials: {n_trials:,}   |   "
          f"Scenarios: {', '.join(scenarios)}")
    print(f"{'='*65}")

    all_sims      = {}
    all_summaries = []

    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario.upper()} ({n_trials:,} trials)...", end=" ")
        sim_df = run_monte_carlo(lanes_df, scenario, n_trials, peak)
        print("done.")

        summary = summarise_scenario(sim_df, lanes_df, scenario)
        all_summaries.append(summary)
        all_sims[scenario] = sim_df

        # Save raw simulation output
        fname = f"sim_results_{scenario}{'_peak' if peak else ''}.csv"
        sim_df.to_csv(fname, index=False)

    # ── Scenario Comparison ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SCENARIO COMPARISON — Network-Level Outcome Distributions")
    print(f"{'='*65}")
    comp_df = compare_scenarios(all_summaries)
    print(comp_df.T.to_string())
    comp_df.to_csv("scenario_comparison.csv", index=False)

    # ── Lane Risk Report (constrained scenario) ───────────────────────────
    ref_scenario = "constrained" if "constrained" in all_sims else scenarios[0]
    risk_df = lane_risk_report(all_sims[ref_scenario], lanes_df)

    print(f"\n{'='*65}")
    print(f"LANE RISK REPORT — {ref_scenario.upper()} scenario")
    print(f"{'='*65}")
    display_cols = ["lane_id", "origin", "destination",
                    "demand_p95", "capacity", "util_p95",
                    "spillover_rate", "sla_breach_rate", "risk_flag"]
    print(risk_df[display_cols].to_string(index=False))
    risk_df.to_csv("lane_risk_report.csv", index=False)

    # ── Key Insight ───────────────────────────────────────────────────────
    if len(all_summaries) == 2:
        c = next(s for s in all_summaries if s.scenario == "constrained")
        u = next(s for s in all_summaries if s.scenario == "unconstrained")
        cost_reduction = (c.total_cost_p95 - u.total_cost_p95) / c.total_cost_p95
        spill_reduction = (c.spillover_p95 - u.spillover_p95) / max(c.spillover_p95, 1)

        print(f"\n{'='*65}")
        print("KEY FINDINGS")
        print(f"{'='*65}")
        print(f"  P95 cost reduction (unconstrained vs constrained): "
              f"{cost_reduction:.1%}")
        print(f"  P95 spillover reduction:  {spill_reduction:.1%}")
        print(f"  Lanes at structural risk (constrained): {c.lanes_at_risk}")
        print(f"  Lanes at structural risk (unconstrained): {u.lanes_at_risk}")

        high_risk = risk_df[risk_df["risk_flag"] == "HIGH"]
        if not high_risk.empty:
            print(f"\n  HIGH RISK lanes (P95 util > 95% or spillover > 20%):")
            for _, row in high_risk.iterrows():
                print(f"    {row['lane_id']}  {row['origin']} → {row['destination']}"
                      f"  spillover_rate={row['spillover_rate']:.1%}"
                      f"  util_p95={row['util_p95']:.1%}")

    print(f"\nOutputs saved: sim_results_*.csv, scenario_comparison.csv, "
          f"lane_risk_report.csv")


def parse_args():
    p = argparse.ArgumentParser(description="THANOS — Network Simulation Engine")
    p.add_argument("--trials",   type=int, default=DEFAULT_TRIALS,
                   help=f"Number of Monte Carlo trials (default: {DEFAULT_TRIALS})")
    p.add_argument("--scenario", default="all",
                   choices=["all", "constrained", "unconstrained"],
                   help="Capacity scenario to simulate (default: all)")
    p.add_argument("--peak",     action="store_true",
                   help=f"Apply peak demand multiplier ({PEAK_MULTIPLIER}x)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(
        n_trials        = args.trials,
        scenario_filter = args.scenario,
        peak            = args.peak,
    )
