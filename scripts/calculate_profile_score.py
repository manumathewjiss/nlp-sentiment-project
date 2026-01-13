"""
Quick helper to calculate Profile Completeness Score (0-5)
Run this after collecting data to fill in scores automatically
"""

import pandas as pd

def calculate_completeness_score(row):
    """Calculate profile completeness score (0-5)"""
    score = 0
    
    # +1 if Has_Custom_URL = Yes
    if str(row.get('Has_Custom_URL', '')).lower() in ['yes', 'true', '1']:
        score += 1
    
    # +1 if Video_Count > 0
    try:
        if pd.notna(row.get('Video_Count')) and float(row.get('Video_Count', 0)) > 0:
            score += 1
    except:
        pass
    
    # +1 if Subscriber_Count > 10
    try:
        if pd.notna(row.get('Subscriber_Count')) and float(row.get('Subscriber_Count', 0)) > 10:
            score += 1
    except:
        pass
    
    # +1 if Account_Age_Days > 30
    try:
        if pd.notna(row.get('Account_Age_Days')) and float(row.get('Account_Age_Days', 0)) > 30:
            score += 1
    except:
        pass
    
    # +1 if View_Count > 100
    try:
        if pd.notna(row.get('View_Count')) and float(row.get('View_Count', 0)) > 100:
            score += 1
    except:
        pass
    
    return score

# Load your collected data
input_file = "task1_collection_template.csv"
print(f"Loading: {input_file}")

df = pd.read_csv(input_file)

# Calculate scores
df['Profile_Completeness_Score'] = df.apply(calculate_completeness_score, axis=1)

# Save updated file
output_file = "task1_collected_profiles.csv"
df.to_csv(output_file, index=False)

print(f"\n✅ Calculated completeness scores!")
print(f"✅ Saved to: {output_file}")
print(f"\nScore distribution:")
print(df['Profile_Completeness_Score'].value_counts().sort_index())

