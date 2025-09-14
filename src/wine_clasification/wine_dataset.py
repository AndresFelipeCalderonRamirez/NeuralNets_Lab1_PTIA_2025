from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch Wine dataset (ID 109) from UCI repository
wine = fetch_ucirepo(id=109)

# Data (features and targets) as pandas dataframes
X = wine.data.features # Features (chemical properties of wine)
y = wine.data.targets # Target variable (wine classes)

# Display basic information about the features (e.g., data types, non-null values)
print(X.info())

# Display the distribution of the wine classes (target variable)
print(y.value_counts()) # Show the count of each class
y.value_counts().plot(kind="bar", title="Distribución de clases") # Plot bar chart for class distribution
plt.show() # Display the plot

# Check for missing values in the features and target variables
print(X.isnull().values.any()) # Check if there are missing values in the features
print(y.isnull().values.any()) # Check if there are missing values in the target variable

# Compute and display the correlation matrix of the feature data
corr_matrix = X.corr() # Calculate correlation matrix between features
plt.figure(figsize=(12,10)) # Set plot size
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True) # Create heatmap with annotations
plt.title("Matriz de correlación de variables") # Set title for the plot
plt.show() # Display the heatmap