# THANOS — Network Simulation Engine

**Tactical Hub for Automated Network Optimisation & Simulation**

Monte Carlo simulation engine for supply chain network planning under uncertainty.

The core idea: supply chain planners making FC expansion and volume routing decisions need to know how their plans perform across the *distribution* of possible futures — not just the expected case. A routing strategy that minimises cost at the mean may have catastrophic tail risk under peak demand or capacity disruption. THANOS answers the question that deterministic optimisers can't: **"How does this plan perform across thousands of possible futures?"**

---

## What This Repo Does

| Layer | What's implemented |
|---|---|
| **Network** | 12-node supply chain (6 FCs, 6 Sort Centers), 12 lanes, realistic topology |
| **Demand** | Log-normal stochastic demand per lane (captures right-skewed peak surge) |
| **Transit** | Stochastic transit times with 5% disruption contamination |
| **Simulation** | Monte Carlo: N independent trials, each sampling full network demand |
| **Scenarios** | Constrained (hard capacity) vs Unconstrained (flex surge capacity) |
| **Peak mode** | 1.8× demand multiplier to stress-test peak season plans |
| **Outputs** | Cost distributions (mean/P25/P75/P95), spillover, SLA breach rate, lane risk flags |

---

## Why Monte Carlo, Not Deterministic Optimisation

| Approach | What it answers | What it misses |
|---|---|---|
| LP / MIP | "What is the optimal plan?" | Assumes inputs are known; ignores uncertainty |
| Scenario trees | "What if demand is high/low?" | Combinatorially intractable at network scale |
| **Monte Carlo** | "How does this plan perform across all futures?" | No single optimal plan; requires interpretation |

Deterministic optimisation gives you the best plan for a model. Monte Carlo tells you whether that plan is *fragile* — whether a modest demand spike causes disproportionate cost or service failure. For strategic decisions (FC expansion, routing contracts), understanding the risk distribution matters more than finding the point optimum.

---

## Simulation Results (1,000 trials, Normal Demand)

### Scenario Comparison — Network-Level Outcomes

| Metric | Constrained | Unconstrained |
|---|---|---|
| Cost Mean (₹) | 456,028 | 447,989 |
| Cost P25 (₹) | 430,540 | 427,706 |
| Cost P75 (₹) | 479,256 | 467,053 |
| **Cost P95 (₹)** | **520,781** | **501,178** |
| Cost P75–P25 Range | 48,716 | 39,347 |
| Volume Routed (mean) | 44,731 | 45,279 |
| **Spillover Mean** | **576** | **9** |
| **Spillover P95** | **2,216** | **0** |
| SLA Breach Rate | 2.8% | 2.8% |

**Key finding: Unconstrained capacity reduces P95 cost by 3.8% and eliminates 100% of P95 spillover.** The mean difference looks modest (₹8K); the tail difference is what drives the decision.

### Lane Risk Report (Constrained Scenario)

| Lane | Route | P95 Demand | Capacity | Util P95 | Spillover Rate | Risk |
|---|---|---|---|---|---|---|
| L09 | SC-DEL → SC-KOL | 5,463 | 4,500 | 100% | 16.2% | 🔴 HIGH |
| L11 | SC-MUM → SC-CHE | 3,895 | 3,500 | 100% | 10.9% | 🔴 HIGH |
| L12 | SC-HYD → SC-CHE | 3,310 | 3,000 | 100% | 10.7% | 🔴 HIGH |
| L07 | SC-BLR → SC-CHE | 5,526 | 5,000 | 100% | 10.5% | 🔴 HIGH |
| L10 | SC-DEL → SC-MUM | 5,780 | 5,500 | 100% | 8.2% | 🔴 HIGH |

SC-DEL → SC-KOL has the highest risk: P95 demand exceeds capacity by 21%, causing spillover in 1 in 6 simulations.

---

## Quickstart

```bash
pip install -r requirements.txt

# Generate network topology
python generate_network.py

# Run full simulation — both scenarios, 1,000 trials
python simulate.py

# Peak season stress test (1.8× demand)
python simulate.py --peak

# Single scenario, more trials for tighter confidence intervals
python simulate.py --scenario constrained --trials 5000

# Unconstrained only
python simulate.py --scenario unconstrained --trials 2000
```

---

## Project Structure

```
thanos-network-optimization-demo/
├── generate_network.py       # Network topology: nodes, lanes, capacity, cost
├── simulate.py               # Monte Carlo simulation engine
├── network_lanes.csv         # Input: 12 lanes with demand params
├── network_nodes.csv         # Input: 12 nodes (FCs, Sort Centers)
├── scenario_comparison.csv   # Output: scenario-level cost/spillover distributions
├── lane_risk_report.csv      # Output: per-lane risk flags
├── sim_results_*.csv         # Output: full trial-level simulation data
├── requirements.txt
└── README.md
```

---

## Design Decisions

**Log-normal demand distribution**
Demand is right-skewed — normal events cluster near the mean, but extreme peaks
(flash sales, supply disruptions) create heavy right tails. Log-normal captures
this; Gaussian underestimates tail risk.

**Disruption contamination in transit times**
Transit is modelled as Normal + 5% Exponential contamination. Pure Normal
underestimates the frequency of severe delays (port congestion, vehicle
breakdown). The contamination adds a realistic heavy tail without requiring a
full discrete-event simulation.

**Spillover cost at ₹25/unit**
Represents the expediting premium for overflow demand — courier upgrade,
emergency vehicle hire, customer compensation. Calibrate this parameter to
your network's actual costs.

**Constrained vs Unconstrained**
- Constrained: hard capacity ceiling (represents current infrastructure)
- Unconstrained: 1.5× surge capacity (flex staffing, temporary vehicle hire,
  partner network overflow)

The gap between them quantifies the value of flex capacity — a key input
to FC expansion and SLA contract decisions.

---

## Extending This Simulator

Adjust parameters in `generate_network.py` to model your network:

| Parameter | Meaning |
|---|---|
| `base_demand` | Expected daily volume per lane |
| `demand_cv` | Demand volatility (coefficient of variation) |
| `cost_per_unit` | Routing cost per shipment unit |
| `transit_mean/std` | Transit time distribution |
| `capacity` | Hard capacity ceiling per lane |

And in `simulate.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `DEFAULT_TRIALS` | 1,000 | Simulation runs (↑ for tighter CI) |
| `PEAK_MULTIPLIER` | 1.8 | Peak demand scaling |
| `SLA_TRANSIT_MAX` | 3.0 | Days beyond which a shipment is an SLA breach |
| `SPILLOVER_COST` | 25.0 | Cost per unit of unmet demand |
| `UNCONSTRAINED_CAP` | 1.5 | Surge capacity multiplier |

---

## Related Projects

- [Middle-Mile Forecasting Simulator](https://github.com/SouRitra01/middle-mile-forecasting-simulator) — XGBoost/LightGBM demand forecasting (the input to this simulation)
- [Email Task Assignment](https://github.com/SouRitra01/email-task-assignment) — BERT-based NLP pipeline for operational task routing

---

## Requirements

```
numpy>=1.24
pandas>=2.0
scipy>=1.11
```

---

*THANOS does not tell you what is optimal. It tells you which plans are fragile.*
