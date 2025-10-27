import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Data from your table
data = {
    "Date": ["13-4-2023", "12-6-2025", "12-6-2025", "12-8-2025", "15-8-2025"],
    "CEO_RF_Beat_power_dBm": [-50, -31, -11, -23, -24],
    "Software_RF_attenuation_dB": [30, 0, 5, 3, 0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert date strings to datetime objects
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

# Calculate beat signal before attenuation
df["Beat_signal_before_dBm"] = df["CEO_RF_Beat_power_dBm"] + df["Software_RF_attenuation_dB"]

# Plot
plt.figure(figsize=(8, 5))
plt.semilogy(df["Date"], 10**(df["Beat_signal_before_dBm"]/10), marker='o')  # convert dBm to mW for log scale

plt.xlabel("Date")
plt.ylabel("Beat Signal Power Before Attenuation [mW] (log scale)")
plt.title("Beat Signal Power Before Attenuation vs Date")
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()
