import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class GDPPredictor:
    def __init__(self, sequence_length=3):
        self.sequence_length = sequence_length
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_columns = None
        
    def prepare_data(self, df):
        """
        Prepare the data for LSTM training
        """
        # Create a copy of the dataframe
        data = df.copy()
        
        # Remove any completely empty rows
        data = data.dropna(how='all')
        
        # Handle missing values for numeric columns only
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
        
        # Encode categorical variables
        if 'Status' in data.columns:
            data['Status_encoded'] = self.label_encoder.fit_transform(data['Status'].fillna('Unknown'))
        

        # Only include numeric columns as features
        potential_features = [col for col in data.columns]
        self.feature_columns = [col for col in potential_features if data[col].dtype in ['int64', 'float64']]
        
        print(f"Using features: {self.feature_columns}")
        
        # Ensure we have the target column
        if 'GDP_Percent' not in data.columns:
            raise ValueError("Target column 'GDP_Percent' not found in data")
        
        # Remove any rows where GDP_Percent is NaN
        data = data.dropna(subset=['GDP_Percent'])
        
        # Sort by country and year to ensure proper time series order
        data = data.sort_values(['Country', 'Year'])
        
        return data
    
    def create_sequences(self, data):
        """
        Create sequences for LSTM training
        """
        X, y = [], []
        countries = data['Country'].unique()
        
        # First, collect all features and targets to fit scalers globally
        all_features = []
        all_targets = []
        
        for country in countries:
            country_data = data[data['Country'] == country].sort_values('Year')
            
            if len(country_data) >= self.sequence_length + 1:
                features = country_data[self.feature_columns].values
                target = country_data['GDP_Percent'].values
                
                all_features.append(features)
                all_targets.extend(target)
        
        if len(all_features) == 0:
            raise ValueError("No country has enough data points for the specified sequence length")
        
        # Fit scalers on all data
        all_features_combined = np.vstack(all_features)
        all_targets_array = np.array(all_targets).reshape(-1, 1)
        
        self.scaler_features.fit(all_features_combined)
        self.scaler_target.fit(all_targets_array)
        
        # Now create sequences with the fitted scalers
        for country in countries:
            country_data = data[data['Country'] == country].sort_values('Year')
            
            if len(country_data) >= self.sequence_length + 1:
                features = country_data[self.feature_columns].values
                target = country_data['GDP_Percent'].values
                
                # Transform features and target using fitted scalers
                features_scaled = self.scaler_features.transform(features)
                target_scaled = self.scaler_target.transform(target.reshape(-1, 1))
                
                # Create sequences
                for i in range(len(features_scaled) - self.sequence_length):
                    X.append(features_scaled[i:(i + self.sequence_length)])
                    y.append(target_scaled[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """
        Build LSTM model
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, df, test_size=0.2, epochs=100, batch_size=32):
        """
        Train the LSTM model
        """
        # Prepare data
        print("Preparing data...")
        data = self.prepare_data(df)
        
        # Create sequences
        print("Creating sequences...")
        X, y = self.create_sequences(data)
        
        print(f"Created {len(X)} sequences with shape {X.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Build model
        print("Building model...")
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=10, 
            min_lr=0.0001
        )
        
        # Train model
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[reduce_lr],
            verbose=1
        )
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Inverse transform predictions
        train_pred_inv = self.scaler_target.inverse_transform(train_pred)
        test_pred_inv = self.scaler_target.inverse_transform(test_pred)
        y_train_inv = self.scaler_target.inverse_transform(y_train)
        y_test_inv = self.scaler_target.inverse_transform(y_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train_inv, train_pred_inv)
        test_mse = mean_squared_error(y_test_inv, test_pred_inv)
        train_mae = mean_absolute_error(y_train_inv, train_pred_inv)
        test_mae = mean_absolute_error(y_test_inv, test_pred_inv)
        train_r2 = r2_score(y_train_inv, train_pred_inv)
        test_r2 = r2_score(y_test_inv, test_pred_inv)
        
        print(f"\nModel Performance:")
        print(f"Train MSE: {train_mse:.6f}")
        print(f"Test MSE: {test_mse:.6f}")
        print(f"Train MAE: {train_mae:.6f}")
        print(f"Test MAE: {test_mae:.6f}")
        print(f"Train R²: {train_r2:.6f}")
        print(f"Test R²: {test_r2:.6f}")
        
        return history, (X_train, X_test, y_train, y_test)
    
    def predict(self, sequence):
        """
        Make prediction on new sequence
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Ensure sequence has the right shape
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, *sequence.shape)
        
        # Scale the sequence
        sequence_scaled = self.scaler_features.transform(
            sequence.reshape(-1, sequence.shape[-1])
        ).reshape(sequence.shape)
        
        # Make prediction
        pred_scaled = self.model.predict(sequence_scaled)
        
        # Inverse transform
        pred = self.scaler_target.inverse_transform(pred_scaled)
        
        return pred[0][0]
    
    def predict_future(self, country, df, future_years=5, scenario_data=None):
            """
            Predict GDP_Percent for future years for a specific country
            
            Parameters:
            - country: Country name to predict for
            - df: Original dataframe with historical data
            - future_years: Number of years to predict into the future
            - scenario_data: Dictionary with future values for features (optional)
                            If None, will use trend-based extrapolation
            """
            if self.model is None:
                raise ValueError("Model not trained yet!")
            
            # Prepare the data
            data = self.prepare_data(df)
            
            # Get the country's historical data
            country_data = data[data['Country'] == country].sort_values('Year')
            
            if len(country_data) < self.sequence_length:
                raise ValueError(f"Not enough historical data for {country}. Need at least {self.sequence_length} years.")
            
            # Get the last sequence_length years as starting point
            last_sequence = country_data[self.feature_columns].tail(self.sequence_length).values
            last_year = country_data['Year'].max()
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for year_ahead in range(1, future_years + 1):
                future_year = last_year + year_ahead
                
                # Generate future feature values
                if scenario_data and future_year in scenario_data:
                    # Use provided scenario data
                    future_features = np.array([scenario_data[future_year][col] for col in self.feature_columns])
                else:
                    # Use trend-based extrapolation
                    future_features = self._extrapolate_features(country_data, self.feature_columns, year_ahead)
                
                # Create the input sequence (last sequence_length-1 + new features)
                input_sequence = np.vstack([current_sequence[1:], future_features.reshape(1, -1)])
                
                # Scale the sequence
                input_scaled = self.scaler_features.transform(input_sequence).reshape(1, self.sequence_length, -1)
                
                # Make prediction
                pred_scaled = self.model.predict(input_scaled)
                pred = self.scaler_target.inverse_transform(pred_scaled)[0][0]
                
                predictions.append({
                    'Country': country,
                    'Year': future_year,
                    'Predicted_GDP_Percent': pred
                })
                
                # Update current sequence for next iteration
                current_sequence = input_sequence
            
            return pd.DataFrame(predictions)
    def save(self, model_path='gdp_model.keras'):
        if self.model:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")

    def load(self, model_path='gdp_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")

    def _extrapolate_features(self, country_data, feature_columns, years_ahead):
        """
        Extrapolate feature values based on historical trends
        """
        future_features = []
        
        for col in feature_columns:
            values = country_data[col].values
            years = country_data['Year'].values
            
            # Simple linear trend extrapolation
            if len(values) >= 2:
                # Calculate trend (slope)
                trend = np.polyfit(years, values, 1)[0]
                last_value = values[-1]
                
                # Extrapolate
                future_value = last_value + (trend * years_ahead)
                
                # Handle special cases
                if col in ['Polio', 'Diphtheria']:  # Vaccination rates should stay between 0-100
                    future_value = np.clip(future_value, 0, 100)
                elif col in ['HIV_AIDS']:  # HIV/AIDS rate should be positive
                    future_value = max(0, future_value)
                elif 'deaths' in col.lower():  # Death counts should be non-negative
                    future_value = max(0, future_value)
                
            else:
                # If not enough data, use the last available value
                future_value = values[-1]
            
            future_features.append(future_value)
        
        return np.array(future_features)
    
    def predict_multiple_countries(self, countries, df, future_years=5, scenario_data=None):
        """
        Predict future GDP_Percent for multiple countries
        
        Parameters:
        - countries: List of country names
        - df: Original dataframe
        - future_years: Number of years to predict
        - scenario_data: Dictionary of scenarios by country and year
                        Format: {country: {year: {feature: value}}}
        """
        all_predictions = []
        
        for country in countries:
            try:
                country_scenario = scenario_data.get(country, {}) if scenario_data else None
                pred_df = self.predict_future(country, df, future_years, country_scenario)
                all_predictions.append(pred_df)
                print(f"✓ Predictions completed for {country}")
            except Exception as e:
                print(f"✗ Error predicting for {country}: {str(e)}")
        
        if all_predictions:
            return pd.concat(all_predictions, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def create_scenario_template(self, countries, df, future_years=5):
        """
        Create a template for scenario planning
        """
        data = self.prepare_data(df)
        
        scenario_template = {}
        
        for country in countries:
            country_data = data[data['Country'] == country]
            if len(country_data) == 0:
                continue
                
            last_year = country_data['Year'].max()
            scenario_template[country] = {}
            
            for year_ahead in range(1, future_years + 1):
                future_year = last_year + year_ahead
                scenario_template[country][future_year] = {}
                
                # Add template values (you can modify these)
                for col in self.feature_columns:
                    last_value = country_data[col].iloc[-1]
                    scenario_template[country][future_year][col] = last_value
        
        return scenario_template

    def plot_training_history(self, history):
        """
        Plot training history
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Load your data
    # df = pd.read_csv('your_data.csv')
    
    # For demonstration, let's create a sample dataframe structure
    # Replace this with your actual data loading
    sample_data = df_merged
    
    df = pd.DataFrame(sample_data)
    
    # Initialize and train the model
    predictor = GDPPredictor(sequence_length=3)
    
    # Train the model
    history, (X_train, X_test, y_train, y_test) = predictor.train(df, epochs=100)
    
      # 2. Multiple countries predictions
    print("\n2. Predicting multiple countries:")
    countries_to_predict = ['Albania', 'Algeria']
    multi_predictions = predictor.predict_multiple_countries(countries_to_predict, df, future_years=3)
    print(multi_predictions)
    
    # 3. Scenario-based predictions
    print("\n3. Creating scenario template:")
    scenario_template = predictor.create_scenario_template(['Albania'], df, future_years=2)
    print("Scenario template structure:", list(scenario_template.keys()))
    
    # 4. Custom scenario prediction
    print("\n4. Custom scenario prediction:")
    # Example: What if Albania improves healthcare and education?
    custom_scenario = {
        'Albania': {
            2016: {
                'Infant_deaths': 0,
                'Percentage_expenditure': 500.0,  # Increase health expenditure
                'Hepatitis_B_men': 35,
                'Hepatitis_B_women': 40,
                'Measles': 1,
                'Under_five_deaths': 0,
                'Polio': 99.0,
                'Diphtheria': 99.0,
                'HIV_AIDS': 0.1,
                'GDP': 5000.0,  # Economic growth
                'Schooling': 15.0,  # Improved education
                'Avg_Life_Expectancy': 70.0,  # Better health outcomes
                'Avg_Adult_Mortality': 75.0,
                'Status_encoded': 0  # Keep encoded status same
            }
        }
    }
    
    scenario_predictions = predictor.predict_future('Albania', df, future_years=1, scenario_data=custom_scenario)
    print(scenario_predictions)


    # Plot training history
    predictor.plot_training_history(history)
    
    # Make a prediction on new data
    # Example: predict GDP_Percent for the next time step
    if len(X_test) > 0:
        sample_sequence = X_test[0]  # Take first test sequence
        prediction = predictor.predict(sample_sequence)
        print(f"\nSample Prediction: {prediction:.6f}")
        print(f"Actual Value: {predictor.scaler_target.inverse_transform(y_test[0].reshape(-1, 1))[0][0]:.6f}")