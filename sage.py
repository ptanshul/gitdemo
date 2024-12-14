import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
data = {
    "TimeOnWebsite": np.random.uniform(1, 50, n_samples),  # Minutes spent on the website
    "PagesVisited": np.random.randint(1, 20, n_samples),   # Number of pages visited
    "Age": np.random.randint(18, 65, n_samples),           # Age of customer
    "ClickedAd": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # Target: Clicked or not
}

# Create DataFrame
df = pd.DataFrame(data)

# Save as CSV
df.to_csv("customer_behavior.csv", index=False)
print("CSV file 'customer_behavior.csv' created!")
