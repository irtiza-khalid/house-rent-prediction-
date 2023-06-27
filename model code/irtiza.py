import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve


df = pd.read_csv(r"C:\Users\omer0\OneDrive\Desktop\House_Rent_Dataset.csv")

df.pop("Posted On")
df.pop("Floor")
df.pop("Area Locality")
df.pop("Point of Contact")
df.pop("Area Type")
df.pop("Tenant Preferred")

df['City'] = df['City'].replace(["Mumbai", "Bangalore", "Hyderabad", "Delhi", "Chennai", "Kolkata"], [5, 4, 3, 2, 1, 0])
df['Furnishing Status'] = df['Furnishing Status'].replace(["Furnished", "Semi-Furnished", "Unfurnished"], [2, 1, 0])

target = df.pop("Rent")

x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(x_train, y_train)

train_accuracy = model.score(x_train, y_train)
test_accuracy = model.score(x_test, y_test)

# Create a directory to save the graphs
os.makedirs('static/plots', exist_ok=True)

# Histograms or box plots
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.savefig('static/plots/histograms.png')
plt.close()

# Correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.savefig('static/plots/correlation_matrix.png')
plt.close()

# Scatter plot matrix
sns.pairplot(df)
plt.savefig('static/plots/scatter_plot_matrix.png')
plt.close()

# Confusion matrix


# Generate confusion matrix


# Generate predictions
y_pred = model.predict(x_test)

# Plot predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.savefig('static/plots/predicted_vs_actual.png')
plt.close()


# Learning curves

train_sizes, train_scores, test_scores = learning_curve(model, x_train, y_train, cv=5, scoring='r2')

plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.legend()
plt.savefig('static/plots/learning_curves.png')
plt.close()






print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Save the model
pickle.dump(model, open('model.pkl', 'wb'))
