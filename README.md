# Inventory Demand Forecasting

## Overview
This project focuses on forecasting inventory demand using a variety of machine learning models and techniques. It preprocesses historical sales data, engineers useful features, and evaluates multiple regression models to predict future sales with accuracy.

## Features
- **Data Preprocessing**:
  - Feature engineering from date values, such as year, month, day, weekday, weekend, and holiday indicators.
  - Application of trigonometric transformations for cyclical features like months.
  - Removal of outliers and normalization of features for stable model training.
- **Exploratory Data Analysis**:
  - Visualization of sales trends over time.
  - Analysis of sales distribution using histograms and boxplots.
  - Correlation heatmaps to identify highly correlated features.
- **Machine Learning Models**:
  - Implementation of linear regression, lasso regression, ridge regression, and XGBoost regression.
  - Training and validation of models to evaluate performance using Mean Absolute Error (MAE).

## Installation
To set up the project locally, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/your-repository.git

# Navigate to the project directory
cd your-repository

# Install dependencies
pip install -r requirements.txt
```

## Dependencies
The project uses the following Python libraries:
- `numpy`: For numerical operations.
- `pandas`: For data manipulation and preprocessing.
- `matplotlib` and `seaborn`: For data visualization.
- `scikit-learn`: For machine learning algorithms and preprocessing.
- `xgboost`: For advanced gradient boosting regression models.
- `holidays`: For identifying public holidays.

## Usage
1. **Prepare the Data**:
   Place your dataset in the specified path and ensure the column names match the ones used in the script.

   ```python
   df = pd.read_csv(r'C:\path\to\your\data.csv')
   ```

2. **Run the Script**:
   Execute the script to preprocess the data, analyze it, and evaluate machine learning models:

   ```bash
   python inventory_demand_forecasting.py
   ```

3. **View Results**:
   - Model training and validation errors are printed to the console.
   - Visualizations are displayed, providing insights into sales trends and feature relationships.

## Code Explanation
### 1. Feature Engineering
The script extracts useful features from date values and incorporates domain-specific transformations:
```python
parts = df["date"].str.split("-", n=3, expand=True)
df["year"] = parts[0].astype('int')
df["month"] = parts[1].astype('int')
df["day"] = parts[2].astype('int')
```
Additional features like `weekend`, `holidays`, and trigonometric month representations (`m1` and `m2`) are created.

### 2. Exploratory Data Analysis
Visualization techniques highlight key patterns in the data:
```python
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    df.groupby(col).mean()['sales'].plot.bar()
plt.show()
```

### 3. Machine Learning Models
Four regression models are trained and evaluated:
```python
models = [LinearRegression(), XGBRegressor(), Lasso(), Ridge()]

for model in models:
    model.fit(X_train, Y_train)
    print(f'{model} : ')
    print('Training Error : ', mae(Y_train, model.predict(X_train)))
    print('Validation Error : ', mae(Y_val, model.predict(X_val)))
```

## Results
- The script outputs the training and validation errors for each model.
- Insights from visualizations and statistical analyses can guide inventory planning and decision-making.

## Example Output
| Model               | Training Error | Validation Error |
|---------------------|----------------|------------------|
| Linear Regression   | X.XX           | Y.YY             |
| XGBoost Regressor   | X.XX           | Y.YY             |
| Lasso Regression    | X.XX           | Y.YY             |
| Ridge Regression    | X.XX           | Y.YY             |

## Contributing
Contributions are welcome! If you have suggestions for improvement, feel free to submit an issue or a pull request.


