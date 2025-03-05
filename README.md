# House Price Prediction Web Application

A Flask-based web application that uses machine learning to predict house prices based on input features. The application implements a custom linear regression model and provides real-time predictions along with feature importance visualization.

## Features

- Interactive web interface for entering house details
- Real-time price predictions
- Feature importance visualization using Chart.js
- Input validation and error handling
- Responsive design using Bootstrap
- Dark mode theme

## Screenshots

### Main Application Interface
![Main Interface](docs/main-interface.png)

### Price Prediction Example
![Price Prediction](docs/price-prediction.png)

## Technical Implementation

### Technology Stack
- Backend: Python Flask
- ML Model: Custom Linear Regression implementation
- Frontend: Bootstrap, Chart.js, Feather Icons
- Data Visualization: Chart.js

### Model Details
The application uses a custom-implemented linear regression model that:
- Generates synthetic training data
- Normalizes input features using StandardScaler
- Calculates feature importance
- Supports real-time predictions

### Key Features
1. **Input Features**:
   - Number of bedrooms (1-10)
   - Number of bathrooms (1-10)
   - Square footage (100-10000)
   - House age (0-200 years)
   - Garage spaces (0-4)

2. **Model Training**:
   - Uses synthetic data generation for training
   - Implements normal equation for linear regression
   - Features standardization for better predictions

3. **Validation**:
   - Input range validation
   - Error handling for invalid inputs
   - Non-negative price predictions

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd house-price-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

The application will be available at `http://localhost:5000`.

## Usage Guide

1. Open the application in your web browser
2. Enter house details in the form:
   - Number of bedrooms
   - Number of bathrooms
   - Square footage
   - House age
   - Garage spaces
3. Click "Predict Price" to get the estimated price
4. View the feature importance chart to understand which factors most influence the prediction

## Project Structure

```
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── predict.js
├── templates/
│   ├── base.html
│   └── index.html
├── app.py
├── main.py
├── model.py
└── README.md
```

## Development

The project follows a modular structure:
- `app.py`: Flask application and routes
- `model.py`: Machine learning model implementation
- `templates/`: HTML templates
- `static/`: CSS, JavaScript, and other static assets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
