import os
from pathlib import Path
import numpy as np
import pandas as pd

def generate_house_prices_dataset(num_samples=2400, seed=42):
    """
    Generates a realistic housing dataset reflecting real-world market dynamics:
    - Non-linear returns to square footage with submarket tier pricing
    - Exponential age depreciation with renovation / condition interaction
    - Bath-to-bedroom coherence penalties and bonuses
    - Urban/Downtown parking premium
    - Heteroscedastic, market-realistic noise (wider variance on luxury properties)
    """
    np.random.seed(seed)
    
    # 1. Feature distributions
    # Area: log-normal skew typical of residential properties (700 to 5,500 sqft)
    raw_area = np.random.lognormal(mean=7.55, sigma=0.45, size=num_samples)
    area = np.clip(np.round(raw_area, -1), 650, 5200).astype(int)

    # Rooms / Bedrooms correlated with area
    expected_rooms = np.clip(np.round(area / 650), 1, 6).astype(int)
    room_noise = np.random.choice([-1, 0, 1], size=num_samples, p=[0.2, 0.65, 0.15])
    rooms = np.clip(expected_rooms + room_noise, 1, 6)

    # Bathrooms correlated with bedrooms
    bath_diff = np.random.choice([0, 1, 2], size=num_samples, p=[0.35, 0.55, 0.10])
    bathrooms = np.clip(rooms - bath_diff, 1, 5)

    # Floors: 1 to 3, single floor more common for small, multi-floor for large
    floors_large = np.random.choice([1, 2, 3], size=num_samples, p=[0.15, 0.60, 0.25])
    floors_small = np.random.choice([1, 2, 3], size=num_samples, p=[0.55, 0.40, 0.05])
    floors = np.where(area > 2400, floors_large, floors_small)

    # Locations
    locations = np.random.choice(
        ['Rural', 'Suburban', 'Urban', 'Downtown'],
        size=num_samples,
        p=[0.18, 0.42, 0.25, 0.15]
    )

    # Year Built: 1960 to 2024
    year_built = np.random.choice(
        np.arange(1960, 2025),
        size=num_samples,
        p=np.linspace(0.5, 1.5, 65) / np.sum(np.linspace(0.5, 1.5, 65))
    )

    # Parking spaces: 0 to 3
    parking = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.12, 0.45, 0.35, 0.08])

    # Condition
    conditions = np.random.choice(['Fair', 'Good', 'Excellent'], size=num_samples, p=[0.22, 0.58, 0.20])

    # 2. Market pricing model with realistic non-linear interactions
    # Base rate per sqft by location
    base_sqft_rates = {
        'Rural': 135.0,
        'Suburban': 210.0,
        'Urban': 295.0,
        'Downtown': 410.0
    }

    # Condition multipliers
    cond_multipliers = {
        'Fair': 0.84,
        'Good': 1.00,
        'Excellent': 1.22
    }

    current_year = 2026
    prices = []

    for i in range(num_samples):
        loc = locations[i]
        sqft = area[i]
        r = rooms[i]
        b = bathrooms[i]
        fl = floors[i]
        pk = parking[i]
        yb = year_built[i]
        cond = conditions[i]

        # Area base with diminishing returns on extreme square footage
        rate = base_sqft_rates[loc]
        if sqft <= 3000:
            area_value = sqft * rate
        else:
            # Diminishing marginal value past 3000 sqft
            area_value = (3000 * rate) + ((sqft - 3000) * rate * 0.78)

        # Room & Bathroom value
        room_value = r * 14000
        bath_value = b * 22000

        # Bath-to-room ratio penalty or bonus
        ratio = b / max(r, 1)
        if ratio < 0.5:
            ratio_adj = -18000  # Severe under-provision of bathrooms
        elif ratio >= 1.0:
            ratio_adj = 15000   # Luxury ensuite provision
        else:
            ratio_adj = 0

        # Parking value is heavily dependent on neighborhood density
        if loc == 'Downtown':
            parking_val = pk * 28000
        elif loc == 'Urban':
            parking_val = pk * 19000
        else:
            parking_val = pk * 10000

        # Floor adjustment
        floor_val = (fl - 1) * 9000

        # Age depreciation curve: older homes lose value, but renovated/excellent condition reverses it
        age = current_year - yb
        depreciation_rate = 0.0038  # ~0.38% per year
        age_factor = max(0.68, 1.0 - (age * depreciation_rate))
        if cond == 'Excellent' and age > 25:
            # Renovated character home premium
            age_factor += 0.12

        # Subtotal before condition and noise
        subtotal = (area_value + room_value + bath_value + ratio_adj + parking_val + floor_val) * age_factor
        conditioned_price = subtotal * cond_multipliers[cond]

        # Realistic heteroscedastic noise (standard error ~ 4-6% of property valuation)
        noise_std = max(18000, conditioned_price * 0.052)
        noise = np.random.normal(0, noise_std)
        final_price = max(65000, round((conditioned_price + noise) / 500) * 500)

        prices.append(float(final_price))

    df = pd.DataFrame({
        'Area(sqft)': area,
        'Rooms': rooms,
        'Bathrooms': bathrooms,
        'Floors': floors,
        'Location': locations,
        'YearBuilt': year_built,
        'Parking': parking,
        'Condition': conditions,
        'Price': prices
    })

    return df

if __name__ == '__main__':
    # Determine directory dynamically
    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "house_prices.csv"

    print("Generating enhanced realistic housing dataset...")
    df = generate_house_prices_dataset(num_samples=2400)
    df.to_csv(csv_path, index=False)
    print(f"Dataset successfully created at: {csv_path}")
    print(f"Total records: {len(df):,}")
    print("\nSummary Statistics:")
    print(df.describe().round(2))
    print("\nLocation counts:")
    print(df['Location'].value_counts())
    print("\nCondition counts:")
    print(df['Condition'].value_counts())
