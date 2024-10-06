import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from investment_data_utils import gaussian_process_model, plot_horizontal_lines, remove_plot_spines
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression  # Import LinearRegression
from matplotlib import pyplot as plt

# Example linear regression prediction function
def linear_regression_prediction(data_frame, time_col, value_col, start_year, end_year):
    # Filter the data between the start year and two years before the max year
    df_filtered = data_frame[(data_frame[time_col] >= start_year) & (data_frame[time_col] <= end_year - 2)]
    
    # Check if the filtered DataFrame is empty
    if df_filtered.empty:
        raise ValueError(f"No data available after filtering between {start_year} and {end_year - 2}")

    # Print the filtered data for debugging
    print(f"Filtered Data:\n{df_filtered}")
    
    # Extract years (time column) and values (value column)
    x_values = np.array(df_filtered[time_col]).reshape(-1, 1)
    y_values = df_filtered[value_col]

    # Train the regression model
    model = linear_model.LinearRegression()
    
    if len(x_values) == 0 or len(y_values) == 0:
        raise ValueError("No valid data available for regression.")

    # Fit the linear model
    model.fit(x_values, y_values)

    # Predict values for the last two years
    predicted_values = {}
    prediction_years = [end_year, end_year - 1]
    for year in prediction_years:
        prediction_array = np.array([year]).reshape(-1, 1)
        predicted_values[str(year)] = list(model.predict(prediction_array))[0]

    # Add predicted values for missing years
    for year in range(start_year, end_year + 1):
        if year not in data_frame[time_col].values:
            predicted_value = model.predict(np.array([year]).reshape(-1, 1))
            data_frame = data_frame.append(pd.DataFrame({time_col: year, value_col: predicted_value}))

    # Replace negative predictions with zero
    data_frame[value_col] = data_frame[value_col].apply(lambda x: max(x, 0))

    return data_frame

# Generate example investment data

# Example years (last 10 years)
years = np.arange(datetime.now().year - 10, datetime.now().year + 1)

# Example investment data: Simulated amount of investment (in EUR) per year
investment_data = pd.DataFrame({
    'year': years,
    'amount_EUR': np.random.randint(500000, 5000000, size=len(years))  # Investment amount in EUR
})

# Constants
investment_column = 'amount_EUR'
investment_year_column = 'year'
confidence_interval_investment = 1.64
curve_fit_investment = 5000
min_year = datetime.now().year - 10
max_year = datetime.now().year

# Perform regression to predict missing years
investment_data_pred = linear_regression_prediction(investment_data, investment_year_column, investment_column, min_year, max_year)

# Check if the data is empty
if investment_data_pred.empty:
    raise ValueError("Investment data is empty after processing.")

# Gaussian Process Regression
np.random.seed(1)
x_values_investment = np.atleast_2d(investment_data_pred[investment_year_column]).T
x_mesh = np.atleast_2d(np.linspace(min_year, max_year + 1, 1000)).T

noise_investment = 1.75 + 1.0 * np.random.random(investment_data_pred[investment_column].shape)

predictions_investment, uncertainty_investment = gaussian_process_model(curve_fit_investment, noise_investment, x_values_investment, investment_data_pred[investment_column], x_mesh, scaling_factor=2.5)

# Plot results
fig, ax = plt.subplots(figsize=(26, 20))

scaled_uncertainty_investment = uncertainty_investment * 100000  # Multiply by 10 for visibility

# Investment data plot (green)
plt.fill(np.concatenate([x_mesh, x_mesh[::-1]]),
         np.concatenate([predictions_investment - confidence_interval_investment * uncertainty_investment,
                         (predictions_investment + confidence_interval_investment * uncertainty_investment)[::-1]]),
         alpha=.8, fc='b', ec='None', label='Investment rounds', color='#008000')  # Changed to green

# Update the legend and text color
legend = plt.legend(bbox_to_anchor=(.5, 1), loc="upper center", frameon=False, ncol=2, markerscale=0.008, handletextpad=0.5, prop={'size': 48}, bbox_transform=plt.gcf().transFigure)
plt.setp(legend.get_texts(), color='#0000FF')  # Changed to blue for the text

# Formatting plot
max_y_value = max(predictions_investment + uncertainty_investment)
ymax = round_up(max_y_value)
#remove_plot_spines(ax)
plot_horizontal_lines(plt, ymax)

plt.tick_params(axis='both', which='major', labelsize=50, color='#FF0000', length=0, pad=15)  # Changed to red for tick color
ax.margins(x=0)
plt.ylim(ymin=0)

plt.show()

