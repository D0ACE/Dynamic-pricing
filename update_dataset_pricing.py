import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('processed_airline_data_sample.csv')

print("Original dataset shape:", df.shape)
print("Sample Ticket Prices (before):", df['Ticket_Price'].head(5).tolist())

# Calculate Days_Until_Flight
df['Days_Until_Flight'] = (pd.to_datetime(df['Flight_Date']) - pd.to_datetime(df['Booking_Date'])).dt.days

# Apply the new surge pricing logic
def calculate_surge(days_until):
    """Calculate surge factor based on days until flight"""
    if days_until > 90:
        return 1.0  # 90+ days = normal price, no surge
    elif days_until > 50:
        return 1.0  # 51-90 days = normal price
    elif days_until == 0:
        return 4.5  # SAME DAY booking = 4.5x surge! EXTREME premium
    elif days_until <= 1:
        return 3.8  # Next day = 3.8x surge!
    elif days_until <= 3:
        return 3.2  # 2-3 days = 3.2x surge
    elif days_until <= 7:
        return 2.8  # 4-7 days = 2.8x surge
    elif days_until <= 14:
        return 2.2  # 8-14 days = 2.2x surge
    elif days_until <= 30:
        return 1.5  # 15-30 days = 1.5x surge
    elif days_until <= 50:
        return 1.2  # 31-50 days = 1.2x surge
    else:
        return 1.0  # Should not reach here

# Apply surge factor to the dataset
df['Surge_Factor'] = df['Days_Until_Flight'].apply(calculate_surge)

# Update ticket prices based on surge factor
# We need to estimate what the base price would be without surge
# For simplicity, we'll divide the current price by whatever surge was applied originally
# and then multiply by the new surge

# Since we don't know the original surge logic, we'll use a simple approach:
# 1. Assume the current prices have some correlation with our new surge logic
# 2. Normalize the prices based on the new surge factors

# For this update, we'll simply apply the new surge factors to adjust prices
# This is a simplification but will demonstrate the concept

# Create a copy of the original prices for reference
df['Original_Ticket_Price'] = df['Ticket_Price']

# Apply new pricing logic (this is a simplified approach for demonstration)
# In a real scenario, you would retrain the model with the correct pricing logic
df['Ticket_Price'] = df['Ticket_Price'] * df['Surge_Factor'] / 2.0  # Simplified adjustment

print("Sample Ticket Prices (after):", df['Ticket_Price'].head(5).tolist())
print("Sample Days Until Flight:", df['Days_Until_Flight'].head(5).tolist())
print("Sample Surge Factors:", df['Surge_Factor'].head(5).tolist())

# Save the updated dataset
df.to_csv('updated_processed_airline_data.csv', index=False)
print("Dataset updated and saved as 'updated_processed_airline_data.csv'")