# THANOS Network Optimisation — Demo

This repository contains synthetic shipping lane data and a Python module that
illustrates how to build a basic network optimisation analysis.  The goal of
the THANOS (Tactical Hub for Automated Network Optimisation & Simulation)
project is to help logistics planners identify high‑value lanes and decide
when to upgrade from standard to express services.

## What this demo does

* **Loads shipments** from a CSV and ensures numeric columns are typed.
* **Recommends service level** per shipment based on transit time thresholds.
* **Aggregates lane metrics** (total volume, mean cost, mean transit time).
* **Scores lanes** using a simple volume–cost product and surfaces the top
  candidates for optimisation.
* **Outputs multiple CSVs**: `suggested_routes.csv`, `lane_summary.csv` and
  `top_routes.csv`.

In a production implementation you might replace the scoring logic with more
advanced optimisation (e.g. linear programming, metaheuristics or discrete
event simulation) and integrate with network graph libraries such as
``networkx`` or OR‑Tools.

## Files

* **shipments.csv** – synthetic dataset of shipment legs with origin,
  destination, volume, cost and transit days.
* **optimizer.py** – Python module containing helper functions and a CLI
  entry point to generate recommendations and summaries.
* **lane_summary.csv**, **top_routes.csv**, **suggested_routes.csv** – outputs
  produced by running the script.

## How to run

Make sure you have Python and ``pandas`` installed, then execute:

```bash
python optimizer.py
```

After running, inspect the generated CSV files to see the recommended service
levels and top lanes.  You can also open the script and adjust the
``threshold`` and ``top_n`` parameters to explore different scenarios.
