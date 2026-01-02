# stock_prediction.py - FINAL VERSION with correct column names

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_nse_data():
    """Load and combine all NSE CSV files"""
    print("Loading NSE data files...")
    
    data_folder = "Assignment1/data/"
    years = list(range(2007, 2013))
    data_frames = []
    
    for year in years:
        file_path = os.path.join(data_folder, f"NSE_data_all_stocks_{year}.csv")
        if os.path.exists(file_path):
            df_year = pd.read_csv(file_path)
            df_year['Year'] = year
            data_frames.append(df_year)
            print(f"✓ {year}: {len(df_year)} records")
        else:
            print(f"✗ {year}: File not found")
    
    if not data_frames:
        raise FileNotFoundError("No data files found!")
    
    nse_data = pd.concat(data_frames, ignore_index=True)
    print(f"\n✅ Combined dataset: {nse_data.shape[0]} rows, {nse_data.shape[1]} columns")
    
    # Show unique stock codes
    print(f"\nTop 20 unique stock codes:")
    print(nse_data['CODE'].unique()[:20])
    
    return nse_data

def clean_numeric_column(df, column_name):
    """Clean numeric columns with commas and convert to float"""
    if column_name in df.columns:
        # Remove commas and convert to numeric
        df[column_name] = df[column_name].astype(str).str.replace(',', '')
        # Convert to numeric, errors become NaN
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df

def analyze_stock(stock_code='SCOM'):
    """Complete analysis for one stock"""
    print(f"\n{'='*60}")
    print(f"ANALYZING STOCK: {stock_code}")
    print(f"{'='*60}")
    
    # 1. Load data
    nse_data = load_nse_data()
    
    # 2. Filter for specific stock
    stock_data = nse_data[nse_data['CODE'] == stock_code].copy()
    
    if len(stock_data) == 0:
        print(f"❌ Stock '{stock_code}' not found!")
        print("Available stocks:", nse_data['CODE'].unique()[:20])
        return None
    
    print(f"Found {len(stock_data)} trading days for {stock_code}")
    
    # 3. Clean and preprocess data
    # Convert DATE to datetime
    stock_data['DATE'] = pd.to_datetime(stock_data['DATE'])
    stock_data = stock_data.sort_values('DATE').reset_index(drop=True)
    
    print(f"Date range: {stock_data['DATE'].min().date()} to {stock_data['DATE'].max().date()}")
    
    # Clean numeric columns
    stock_data = clean_numeric_column(stock_data, 'Volume')
    
    # Convert 'Change' column (has '-' for no change)
    stock_data['Change'] = stock_data['Change'].replace('-', '0')
    stock_data['Change'] = pd.to_numeric(stock_data['Change'], errors='coerce')
    
    # Convert 'Change%' column (remove % sign)
    stock_data['Change%'] = stock_data['Change%'].replace('-', '0%')
    stock_data['Change%'] = stock_data['Change%'].astype(str).str.replace('%', '')
    stock_data['Change%'] = pd.to_numeric(stock_data['Change%'], errors='coerce')
    
    # 4. Set target variable (Day Price)
    target_col = 'Day Price'
    print(f"\nTarget variable: '{target_col}'")
    
    # 5. Create features for prediction
    # Lagged prices (previous 5 days)
    num_lags = 5
    for lag in range(1, num_lags + 1):
        stock_data[f'Price_lag_{lag}'] = stock_data[target_col].shift(lag)
    
    # Moving averages
    stock_data['Price_MA_5'] = stock_data[target_col].rolling(window=5).mean().shift(1)
    stock_data['Price_MA_10'] = stock_data[target_col].rolling(window=10).mean().shift(1)
    stock_data['Price_MA_20'] = stock_data[target_col].rolling(window=20).mean().shift(1)
    
    # Daily price range (volatility)
    stock_data['Daily_Range'] = (stock_data['Day High'] - stock_data['Day Low']) / stock_data[target_col]
    stock_data['Daily_Range_lag1'] = stock_data['Daily_Range'].shift(1)
    
    # Volume features
    stock_data['Volume_lag1'] = stock_data['Volume'].shift(1)
    stock_data['Volume_MA_5'] = stock_data['Volume'].rolling(window=5).mean().shift(1)
    
    # Change features
    stock_data['Change_lag1'] = stock_data['Change'].shift(1)
    stock_data['Change%_lag1'] = stock_data['Change%'].shift(1)
    
    # Previous day features
    stock_data['Previous_lag1'] = stock_data['Previous'].shift(1)
    
    # Drop rows with missing values
    stock_data_clean = stock_data.dropna().copy()
    
    # Define features
    feature_cols = [f'Price_lag_{i}' for i in range(1, num_lags+1)] + \
                   ['Price_MA_5', 'Price_MA_10', 'Price_MA_20',
                    'Daily_Range_lag1', 'Volume_lag1', 'Volume_MA_5',
                    'Change_lag1', 'Change%_lag1', 'Previous_lag1']
    
    X = stock_data_clean[feature_cols]
    y = stock_data_clean[target_col]
    
    print(f"\nFeatures created: {len(feature_cols)}")
    print(f"Clean dataset: {X.shape}")
    
    # 6. Split data chronologically (80% train, 20% test)
    split_index = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"\nData split:")
    print(f"  Training: {X_train.shape} (First 80%)")
    print(f"  Testing:  {X_test.shape} (Last 20%)")
    
    # 7. Train multiple regression models
    print("\nTraining models...")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Random Forest (50 trees)': RandomForestRegressor(n_estimators=50, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        
        # Store results
        results[name] = {
            'model': model,
            'mse_train': mse_train,
            'mse_test': mse_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mae_test': mae_test,
            'rmse_test': rmse_test,
            'y_pred_test': y_pred_test
        }
        
        print(f"    Test R²:  {r2_test:.4f}, RMSE: {rmse_test:.4f}")
    
    # 8. Display model comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Test R²': [results[m]['r2_test'] for m in results.keys()],
        'Test MSE': [results[m]['mse_test'] for m in results.keys()],
        'Test RMSE': [results[m]['rmse_test'] for m in results.keys()],
        'Test MAE': [results[m]['mae_test'] for m in results.keys()]
    }).sort_values('Test R²', ascending=False)
    
    print(comparison_df.to_string(index=False))
    
    # 9. Create evaluation plots
    create_evaluation_plots(results, X_test, y_test, stock_data_clean, split_index, stock_code)
    
    # 10. Show feature importance for Random Forest
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (Random Forest)")
    print(f"{'='*60}")
    
    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'][:15], feature_importance['Importance'][:15])
    plt.xlabel('Importance')
    plt.title(f'Top 15 Feature Importance for {stock_code} (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"Assignment1/{stock_code}_feature_importance.png", dpi=150)
    plt.show()
    
    return results, comparison_df

def create_evaluation_plots(results, X_test, y_test, stock_data, split_index, stock_code):
    """Create all required evaluation plots"""
    
    # Select best model based on R²
    best_model_name = max(results, key=lambda k: results[k]['r2_test'])
    best_result = results[best_model_name]
    y_pred_test = best_result['y_pred_test']
    residuals = y_test - y_pred_test
    
    # Create 2x2 plot grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'KBS Assignment 1: NSE Stock Prediction ({stock_code})\nBest Model: {best_model_name} (R²={best_result["r2_test"]:.3f})', 
                 fontsize=16, fontweight='bold')
    
    # ===== PLOT 1: Actual vs Predicted (with R²) =====
    axes[0, 0].scatter(y_test, y_pred_test, alpha=0.6, s=50, edgecolor='k', linewidth=0.5)
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Price (KES)', fontsize=12)
    axes[0, 0].set_ylabel('Predicted Price (KES)', fontsize=12)
    axes[0, 0].set_title(f'Actual vs Predicted Values\nR² = {best_result["r2_test"]:.3f}', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # ===== PLOT 2: Residuals Plot =====
    axes[0, 1].scatter(y_pred_test, residuals, alpha=0.6, s=50)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Price (KES)', fontsize=12)
    axes[0, 1].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    axes[0, 1].set_title('Residuals Plot', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add residual statistics
    residual_mean = residuals.mean()
    residual_std = residuals.std()
    axes[0, 1].text(0.05, 0.95, f'Mean: {residual_mean:.2f}\nStd: {residual_std:.2f}', 
                    transform=axes[0, 1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ===== PLOT 3: Error Distribution =====
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Prediction Error', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title(f'Error Distribution\nMSE = {best_result["mse_test"]:.2f}, RMSE = {best_result["rmse_test"]:.2f}', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # ===== PLOT 4: Time Series Prediction =====
    test_dates = stock_data['DATE'].iloc[split_index:].reset_index(drop=True)
    axes[1, 1].plot(test_dates, y_test.values, label='Actual Price', color='blue', alpha=0.8, linewidth=2)
    axes[1, 1].plot(test_dates, y_pred_test, label='Predicted Price', color='red', alpha=0.8, linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Date', fontsize=12)
    axes[1, 1].set_ylabel('Stock Price (KES)', fontsize=12)
    axes[1, 1].set_title('Actual vs Predicted Over Time', fontsize=14)
    axes[1, 1].legend(loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"Assignment1/{stock_code}_evaluation_plots.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"\n✅ Evaluation plots saved as: {plot_filename}")
    
    plt.show()
    
    # Additional: Prediction Error Over Time
    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, np.abs(residuals), alpha=0.7, linewidth=1)
    plt.fill_between(test_dates, 0, np.abs(residuals), alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Absolute Prediction Error')
    plt.title(f'Absolute Prediction Error Over Time for {stock_code}')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"Assignment1/{stock_code}_prediction_error_over_time.png", dpi=150)
    plt.show()

def explore_stock_data(stock_code='SCOM'):
    """Explore data for a specific stock"""
    nse_data = load_nse_data()
    stock_data = nse_data[nse_data['CODE'] == stock_code].copy()
    
    if len(stock_data) == 0:
        print(f"Stock '{stock_code}' not found!")
        return
    
    # Convert date and sort
    stock_data['DATE'] = pd.to_datetime(stock_data['DATE'])
    stock_data = stock_data.sort_values('DATE')
    
    print(f"\nData for {stock_code}:")
    print(f"Number of trading days: {len(stock_data)}")
    print(f"Date range: {stock_data['DATE'].min().date()} to {stock_data['DATE'].max().date()}")
    
    # Summary statistics
    print(f"\nPrice Statistics for {stock_code}:")
    price_stats = stock_data['Day Price'].describe()
    print(price_stats)
    
    # Plot price history
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['DATE'], stock_data['Day Price'], linewidth=1.5)
    plt.xlabel('Date')
    plt.ylabel('Price (KES)')
    plt.title(f'{stock_code} Stock Price History (2007-2012)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"Assignment1/{stock_code}_price_history.png", dpi=150)
    plt.show()
    
    return stock_data

def main():
    """Main execution function"""
    print("="*70)
    print("KBS ASSIGNMENT 1: NAIROBI SECURITIES EXCHANGE STOCK PRICE PREDICTION")
    print("="*70)
    print("Dataset: NSE All Stocks Prices 2007-2012")
    print("Task: Build regression model to predict stock prices")
    print("="*70)
    
    # First, explore available stocks
    nse_data = load_nse_data()
    
    # Show most common stocks
    print(f"\nMost traded stocks (by record count):")
    stock_counts = nse_data['CODE'].value_counts().head(10)
    print(stock_counts)
    
    # Try analysis for different stocks
    # Common NSE stocks: SCOM, KCB, EQTY, COOP, EABL, BAT, SCBK
    
    stocks_to_analyze = ['SCOM', 'KCB', 'EQTY', 'COOP', 'EABL']
    
    all_results = {}
    
    for stock in stocks_to_analyze:
        try:
            print(f"\n{'='*70}")
            print(f"STARTING ANALYSIS FOR: {stock}")
            print(f"{'='*70}")
            
            # First explore the data
            explore_stock_data(stock)
            
            # Then run the full analysis
            results, comparison = analyze_stock(stock)
            all_results[stock] = {
                'results': results,
                'comparison': comparison
            }
            
            print(f"\n✓ Analysis complete for {stock}")
            
        except Exception as e:
            print(f"\n❌ Error analyzing {stock}: {str(e)}")
    
    # Summary of all analyses
    print(f"\n{'='*70}")
    print("ASSIGNMENT 1 - SUMMARY OF ALL ANALYSES")
    print(f"{'='*70}")
    
    if all_results:
        summary_data = []
        for stock, data in all_results.items():
            best_model = data['comparison'].iloc[0]['Model']
            best_r2 = data['comparison'].iloc[0]['Test R²']
            best_rmse = data['comparison'].iloc[0]['Test RMSE']
            summary_data.append([stock, best_model, best_r2, best_rmse])
        
        summary_df = pd.DataFrame(summary_data, 
                                  columns=['Stock', 'Best Model', 'R² Score', 'RMSE'])
        print("\n" + summary_df.to_string(index=False))
        
        # Save summary to CSV
        summary_df.to_csv("Assignment1/summary_results.csv", index=False)
        print(f"\n✅ Summary saved to: Assignment1/summary_results.csv")
    
    print(f"\n{'='*70}")
    print("ASSIGNMENT 1 COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    print("\nGenerated files:")
    print("1. Evaluation plots for each stock (4 plots each)")
    print("2. Feature importance plots")
    print("3. Price history charts")
    print("4. Summary results CSV file")
    print("\nNext steps for improvement:")
    print("1. Try more complex models (XGBoost, Neural Networks)")
    print("2. Add more features (technical indicators)")
    print("3. Try predicting different targets (e.g., next day return %)")
    print("4. Use multiple stocks in one model")

# Run the script
if __name__ == "__main__":
    main()