"""
generate_network.py
===================
Generates synthetic supply chain network data for THANOS simulation.

Network topology: FC → Sort Center → Delivery Station
Each lane has stochastic demand, capacity constraints, cost, and transit time.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ── Network Nodes ─────────────────────────────────────────────────────────────

NODES = {
    # Fulfillment Centers (supply)
    "FC-BLR-1": {"type": "FC",   "region": "South",  "capacity": 12000},
    "FC-BLR-2": {"type": "FC",   "region": "South",  "capacity": 9000},
    "FC-DEL-1": {"type": "FC",   "region": "North",  "capacity": 15000},
    "FC-DEL-2": {"type": "FC",   "region": "North",  "capacity": 11000},
    "FC-MUM-1": {"type": "FC",   "region": "West",   "capacity": 13000},
    "FC-HYD-1": {"type": "FC",   "region": "South",  "capacity": 8000},

    # Sort Centers (transit hubs)
    "SC-BLR":   {"type": "SC",   "region": "South",  "capacity": 25000},
    "SC-DEL":   {"type": "SC",   "region": "North",  "capacity": 30000},
    "SC-MUM":   {"type": "SC",   "region": "West",   "capacity": 28000},
    "SC-HYD":   {"type": "SC",   "region": "South",  "capacity": 18000},
    "SC-CHE":   {"type": "SC",   "region": "South",  "capacity": 15000},
    "SC-KOL":   {"type": "SC",   "region": "East",   "capacity": 14000},
}

# ── Lane Definitions ──────────────────────────────────────────────────────────
# Each lane: origin → destination, base demand distribution, cost, transit time

LANES = [
    # FC → Sort Center lanes
    {"origin": "FC-BLR-1", "dest": "SC-BLR",  "base_demand": 4500, "demand_cv": 0.20,
     "cost_per_unit": 8.5,  "transit_mean": 0.5, "transit_std": 0.1, "capacity": 6000},
    {"origin": "FC-BLR-2", "dest": "SC-BLR",  "base_demand": 3200, "demand_cv": 0.22,
     "cost_per_unit": 9.0,  "transit_mean": 0.5, "transit_std": 0.1, "capacity": 4500},
    {"origin": "FC-DEL-1", "dest": "SC-DEL",  "base_demand": 6000, "demand_cv": 0.18,
     "cost_per_unit": 7.5,  "transit_mean": 0.5, "transit_std": 0.1, "capacity": 8000},
    {"origin": "FC-DEL-2", "dest": "SC-DEL",  "base_demand": 4200, "demand_cv": 0.21,
     "cost_per_unit": 8.0,  "transit_mean": 0.5, "transit_std": 0.1, "capacity": 5500},
    {"origin": "FC-MUM-1", "dest": "SC-MUM",  "base_demand": 5200, "demand_cv": 0.19,
     "cost_per_unit": 8.2,  "transit_mean": 0.5, "transit_std": 0.1, "capacity": 7000},
    {"origin": "FC-HYD-1", "dest": "SC-HYD",  "base_demand": 3000, "demand_cv": 0.24,
     "cost_per_unit": 9.5,  "transit_mean": 0.5, "transit_std": 0.1, "capacity": 4000},

    # Sort Center → Sort Center (inter-regional lanes)
    {"origin": "SC-BLR",   "dest": "SC-CHE",  "base_demand": 3800, "demand_cv": 0.25,
     "cost_per_unit": 12.0, "transit_mean": 1.8, "transit_std": 0.4, "capacity": 5000},
    {"origin": "SC-BLR",   "dest": "SC-HYD",  "base_demand": 2900, "demand_cv": 0.22,
     "cost_per_unit": 10.5, "transit_mean": 1.5, "transit_std": 0.3, "capacity": 4000},
    {"origin": "SC-DEL",   "dest": "SC-KOL",  "base_demand": 3500, "demand_cv": 0.28,
     "cost_per_unit": 14.0, "transit_mean": 2.5, "transit_std": 0.5, "capacity": 4500},
    {"origin": "SC-DEL",   "dest": "SC-MUM",  "base_demand": 4100, "demand_cv": 0.23,
     "cost_per_unit": 11.5, "transit_mean": 2.0, "transit_std": 0.4, "capacity": 5500},
    {"origin": "SC-MUM",   "dest": "SC-CHE",  "base_demand": 2600, "demand_cv": 0.26,
     "cost_per_unit": 13.0, "transit_mean": 2.2, "transit_std": 0.5, "capacity": 3500},
    {"origin": "SC-HYD",   "dest": "SC-CHE",  "base_demand": 2200, "demand_cv": 0.27,
     "cost_per_unit": 11.0, "transit_mean": 1.6, "transit_std": 0.3, "capacity": 3000},
]


def build_lanes_df() -> pd.DataFrame:
    rows = []
    for i, lane in enumerate(LANES):
        rows.append({
            "lane_id":       f"L{i+1:02d}",
            "origin":        lane["origin"],
            "destination":   lane["dest"],
            "origin_type":   NODES[lane["origin"]]["type"],
            "dest_type":     NODES[lane["dest"]]["type"],
            "region":        NODES[lane["origin"]]["region"],
            "base_demand":   lane["base_demand"],
            "demand_cv":     lane["demand_cv"],         # coefficient of variation
            "cost_per_unit": lane["cost_per_unit"],
            "transit_mean":  lane["transit_mean"],
            "transit_std":   lane["transit_std"],
            "capacity":      lane["capacity"],
        })
    return pd.DataFrame(rows)


def build_nodes_df() -> pd.DataFrame:
    rows = [{"node_id": k, **v} for k, v in NODES.items()]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    lanes_df = build_lanes_df()
    nodes_df = build_nodes_df()
    lanes_df.to_csv("network_lanes.csv", index=False)
    nodes_df.to_csv("network_nodes.csv", index=False)
    print(f"Generated {len(lanes_df)} lanes, {len(nodes_df)} nodes.")
    print(lanes_df.to_string(index=False))
