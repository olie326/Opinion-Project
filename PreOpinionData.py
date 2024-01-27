import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# Load the CSV file into a pandas DataFrame

file_path = './31120637.csv'

df = pd.read_csv(file_path, encoding='ISO 8859-1')

print(df)
# Define a mapping for your Likert scale
climate_mapping = {
    'Skipped': 0,
    'Not important at all': 1,
    'Not so important': 2,
    'Somewhat important': 3,
    'Very important': 4,
}

# Replace keywords with Likert scale values
df.replace({'Q3_6': climate_mapping}, inplace=True)

# Extract the relevant columns as a NumPy array
columns_to_extract = ['Q3_6']  # Replace with your column names
data = df[columns_to_extract].to_numpy()

# Print the updated DataFrame and the NumPy array
print("Updated DataFrame:")

mu, std = norm.fit(data)

# Plot the histogram.
plt.hist(data, bins=25, density=True, alpha=0.6, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)

p_norm = (p - min(p)) / (max(p) - min(p))
print(p_norm)

plt.plot(x, p, 'k', linewidth=2)
plt.show()


