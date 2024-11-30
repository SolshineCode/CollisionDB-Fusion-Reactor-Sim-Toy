import tensorflow as tf
from pycollisiondb import PyCollision
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the query for retrieving datasets from CollisionDB
# This example focuses on a hypothetical tokamak reactor
# and will retrieve electron-impact ionization cross sections for deuterium
query = {
   'reactants': ['e-', '(2H)'],
    'process_types': ['EIN'],
    'data_type': 'cross section'
}

# Instantiate the PyCollision object and retrieve datasets
pycoll = PyCollision.get_datasets(query=query)

# Extract relevant data for training
energies = []
cross_sections = []
for dataset in pycoll.datasets.values():
    energies.extend(dataset.x)
    cross_sections.extend(dataset.y)

# Convert data to NumPy arrays for preprocessing
energies = np.array(energies).reshape(-1, 1)  # Reshape the array
cross_sections = np.array(cross_sections)

# Data Preprocessing
# Scale the input features (energies)
scaler = StandardScaler()
scaled_energies = scaler.fit_transform(energies)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    scaled_energies, cross_sections, test_size=0.2, random_state=42
)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Mean Absolute Error on Test Data: {mae}")

# Make predictions
# For example, predict the cross section at a specific energy
new_energy = np.array([])  # Example energy value (in the original units)
scaled_new_energy = scaler.transform(new_energy)
predicted_cross_section = model.predict(scaled_new_energy)

print(f"Predicted Cross Section: {predicted_cross_section}")

# Save the trained model
model.save("trained_collisiondb_toy_model.h5")
