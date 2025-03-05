import numpy as np
from sklearn.preprocessing import StandardScaler

class HousePriceModel:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
    
    def generate_sample_data(self):
        """Generate synthetic data for training"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        bedrooms = np.random.randint(1, 6, n_samples)
        bathrooms = np.random.randint(1, 4, n_samples)
        sqft = np.random.uniform(500, 5000, n_samples)
        age = np.random.uniform(0, 100, n_samples)
        garage = np.random.randint(0, 3, n_samples)
        
        # Create feature matrix
        X = np.column_stack([bedrooms, bathrooms, sqft, age, garage])
        
        # Generate target prices with some noise
        y = (150000 + 
             bedrooms * 50000 + 
             bathrooms * 40000 + 
             sqft * 200 + 
             -age * 1000 + 
             garage * 25000 + 
             np.random.normal(0, 50000, n_samples))
        
        return X, y
    
    def train(self):
        """Train the linear regression model"""
        # Generate synthetic data
        X, y = self.generate_sample_data()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Add bias term
        X_scaled = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
        
        # Train using normal equation
        self.weights = np.linalg.inv(X_scaled.T @ X_scaled) @ X_scaled.T @ y
        
        # Extract bias and weights
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
        
        # Calculate feature importance
        feature_names = ['bedrooms', 'bathrooms', 'sqft', 'age', 'garage']
        importance = np.abs(self.weights)
        total_importance = np.sum(importance)
        self.feature_importance = {
            name: (imp / total_importance) * 100 
            for name, imp in zip(feature_names, importance)
        }
    
    def predict(self, features):
        """Make prediction for new data"""
        # Convert features to array
        X = np.array([[
            features['bedrooms'],
            features['bathrooms'],
            features['sqft'],
            features['age'],
            features['garage']
        ]])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = np.dot(X_scaled, self.weights) + self.bias
        
        return max(0, prediction[0])  # Ensure non-negative prediction
    
    def get_feature_importance(self):
        """Return feature importance percentages"""
        return self.feature_importance
