"""
optimizer.py
--------------

This module implements a few helper functions to analyse shipping lane data and
produce recommendations for service levels and route prioritisation.  The
original version of this demo simply labelled lanes as 'Express' when transit
times were above a hard‑coded threshold.  This re‑write introduces more
structure and realistic logic:

* Loads a synthetic shipments dataset from ``shipments.csv`` into a DataFrame
  with typed columns.
* Calculates service recommendations per shipment using a configurable
  transit‑day threshold.
* Produces an aggregated lane summary (by origin/destination) with total
  volume, average cost and mean transit time.
* Scores lanes based on a volume–cost product and extracts the top
  performers for review.
* Writes multiple CSV outputs and prints a summary to the console.

These routines are deliberately written to be easy to extend.  In a real
production environment you might replace the scoring function with a more
sophisticated optimisation model (e.g. linear programming) or integrate with
network simulation libraries such as ``networkx`` or OR‑Tools.
"""

import pandas as pd
from typing import Tuple


def load_shipments(path: str) -> pd.DataFrame:
    """Load shipment data from a CSV into a DataFrame with explicit dtypes."""
    df = pd.read_csv(path)
    # normalise column names: use 'destination' consistently
    if 'dest' in df.columns and 'destination' not in df.columns:
        df = df.rename(columns={'dest': 'destination'})
    # ensure numeric columns have the right types
    df['volume'] = df['volume'].astype(float)
    df['cost'] = df['cost'].astype(float)
    df['transit_days'] = df['transit_days'].astype(float)
    return df


def suggest_service(row: pd.Series, threshold: float = 3.0) -> str:
    """
    Suggest a service level for a shipment.  Lanes exceeding the threshold
    transit time are tagged as 'Express' otherwise 'Standard'.

    Parameters
    ----------
    row : pd.Series
        The shipment row to evaluate.
    threshold : float, optional
        Transit day threshold for switching service tiers.

    Returns
    -------
    str
        The service recommendation.
    """
    return 'Express' if row['transit_days'] > threshold else 'Standard'


def summarise_lanes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate shipments by (origin, destination) and compute total volume,
    mean cost and mean transit days for each lane.

    Returns a DataFrame sorted by total volume descending.
    """
    summary = (
        df.groupby(['origin', 'destination'])
        .agg(
            total_volume=pd.NamedAgg(column='volume', aggfunc='sum'),
            mean_cost=pd.NamedAgg(column='cost', aggfunc='mean'),
            mean_transit=pd.NamedAgg(column='transit_days', aggfunc='mean'),
        )
        .reset_index()
    )
    return summary.sort_values('total_volume', ascending=False)


def score_lanes(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Create a simple score for each shipment based on the product of volume and
    cost.  Higher scores indicate lanes that may deserve optimisation or
    investment.  Returns the top ``top_n`` records.
    """
    df = df.copy()
    df['route_score'] = df['volume'] * df['cost']
    return df.sort_values('route_score', ascending=False).head(top_n)[
        ['origin', 'destination', 'route_score']
    ]


def main() -> None:
    # load shipments
    shipments = load_shipments('shipments.csv')

    # determine service recommendation for each shipment
    shipments['suggested_service'] = shipments.apply(suggest_service, axis=1)

    # summarise lanes and rank them by total volume
    lane_summary = summarise_lanes(shipments)
    lane_summary.to_csv('lane_summary.csv', index=False)

    # compute top routes by a simple volume*cost score
    top_routes = score_lanes(shipments, top_n=5)
    top_routes.to_csv('top_routes.csv', index=False)

    # write the enriched shipment data
    shipments.to_csv('suggested_routes.csv', index=False)

    # output a summary to the console
    print('Top recommended routes based on volume*cost score:')
    print(top_routes)


if __name__ == '__main__':
    main()
