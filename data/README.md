# Data Directory

Place your sales data CSV files here.

## Expected Data Format

Your CSV file should have the following columns:
- `Date`: Date of the sales record (format: YYYY-MM-DD)
- `Weekly_Sales`: Target variable (sales amount)
- Additional features like Store, Dept, Temperature, Fuel_Price, CPI, Unemployment, etc.

## Sample Data

If you don't have your own data, the project includes a sample data generator. You can use it by running:

```python
from src.data_preprocessing import create_sample_data
df = create_sample_data(num_rows=1000, start_date='2010-01-01')
df.to_csv('data/sample_sales_data.csv', index=False)
```

Or simply run the Jupyter notebook which will automatically generate sample data for demonstration.
