
# SpaceX Launch Data Analysis and Visualization


# Install required packages (JupyterLite environment only)
import piplite
await piplite.install(['numpy'])
await piplite.install(['pandas'])
await piplite.install(['seaborn'])

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from js import fetch
import io


#  Load the dataset

DATA_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
)

resp = await fetch(DATA_URL)
dataset_csv = io.BytesIO((await resp.arrayBuffer()).to_py())
df = pd.read_csv(dataset_csv)

# Preview the dataset
print("First 5 rows of the dataset:")
print(df.head())


#  Visualizations


# Payload Mass vs Flight Number by Class
sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect=5)
plt.xlabel("Flight Number", fontsize=20)
plt.ylabel("Payload Mass (kg)", fontsize=20)
plt.title("Payload Mass vs Flight Number by Class", fontsize=16)
plt.show()

# Flight Number vs Launch Site by Class
sns.catplot(x="FlightNumber", y="LaunchSite", hue="Class", data=df, aspect=2.5, height=6)
plt.xlabel("Flight Number", fontsize=14)
plt.ylabel("Launch Site", fontsize=14)
plt.title("Flight Number vs Launch Site by Class", fontsize=16)
plt.show()

# Payload Mass vs Launch Site by Class
sns.scatterplot(data=df, x="PayloadMass", y="LaunchSite", hue="Class")
plt.xlabel("Payload Mass (kg)", fontsize=14)
plt.ylabel("Launch Site", fontsize=14)
plt.title("Payload Mass vs Launch Site by Class", fontsize=16)
plt.show()

# Flight Number vs Orbit by Class
sns.scatterplot(data=df, x="FlightNumber", y="Orbit", hue="Class")
plt.xlabel("Flight Number", fontsize=14)
plt.ylabel("Orbit Type", fontsize=14)
plt.title("Flight Number vs Orbit Type by Launch Success", fontsize=16)
plt.show()

# Payload Mass vs Orbit by Class
sns.scatterplot(data=df, x="PayloadMass", y="Orbit", hue="Class")
plt.xlabel("Payload Mass (kg)", fontsize=14)
plt.ylabel("Orbit Type", fontsize=14)
plt.title("Payload Mass vs Orbit Type by Launch Success", fontsize=16)
plt.show()


#  Extract Year and Plot Success Trend


def extract_year():
    """Extracts year from the 'Date' column."""
    return [date.split("-")[0] for date in df["Date"]]

df["Date"] = extract_year()
df["Date"] = df["Date"].astype(int)

# Group by year and calculate average success rate
yearly_success = df.groupby("Date")["Class"].mean().reset_index()

# Plot the trend
plt.figure(figsize=(10, 6))
sns.lineplot(x="Date", y="Class", data=yearly_success, marker="o")
plt.title("Yearly Launch Success Trend", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average Success Rate", fontsize=14)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# Feature Engineering


# Select input features
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights',
               'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block',
               'ReusedCount', 'Serial']]

# One-hot encode categorical variables
categorical_cols = ['Orbit', 'LaunchSite', 'LandingPad', 'Serial']
features_encoded = pd.get_dummies(features, columns=categorical_cols)

# Convert all columns to float64
features_encoded = features_encoded.astype('float64')

# Preview encoded features
print("Encoded Features:")
print(features_encoded.head())

