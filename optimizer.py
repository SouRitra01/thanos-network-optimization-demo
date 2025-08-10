import pandas as pd

df=pd.read_csv('shipments.csv')
# simple suggestion: express if >3 days
df['suggested_service']=df['transit_days'].apply(lambda d:'Express' if d>3 else 'Standard')
df.to_csv('suggested_routes.csv', index=False)
