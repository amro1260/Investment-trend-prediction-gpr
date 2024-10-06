#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import math
import re
from sklearn import linear_model
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Constant
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm

# Configurations

scale_units = {1: '', 1000: 'k', 1000000: 'm', 1000000000: "bn", 'log': '(log scale)'}

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


def adjust_predicted_values(data_frame, predicted_values, time_col, value_col, max_year):
    """
    Adjusts predicted values for the last two years by combining actual and predicted values with weighted importance.
    """
    current_year_value = data_frame.loc[data_frame[time_col] == max_year, value_col]
    if predicted_values.get(str(max_year)) > list(current_year_value)[0]:
        weighted_prediction = 0.2 * current_year_value + 0.8 * predicted_values.get(str(max_year))
        data_frame.loc[data_frame[time_col] == max_year, value_col] = weighted_prediction

    previous_year_value = data_frame.loc[data_frame[time_col] == max_year - 1, value_col]
    if predicted_values.get(str(max_year - 1)) > list(previous_year_value)[0]:
        weighted_prediction = 0.6 * previous_year_value + 0.4 * predicted_values.get(str(max_year - 1))
        data_frame.loc[data_frame[time_col] == max_year - 1, value_col] = weighted_prediction

def gaussian_process_model(curve_fit_val, noise_factor, x_values, y_values, mesh_values, scaling_factor=3):
    """
    Applies Gaussian Process regression for curve fitting and returns predictions with uncertainty estimation.
    """
    kernel = Constant(10, (100, int(curve_fit_val))) * RBF(1, (1, 100))
    gp_model = GaussianProcessRegressor(kernel=kernel, alpha=noise_factor ** scaling_factor, n_restarts_optimizer=9)
    
    gp_model.fit(x_values, y_values)
    y_predictions, uncertainty = gp_model.predict(mesh_values, return_std=True)
    uncertainty = uncertainty * np.geomspace(0.8, 1.5, len(uncertainty))
    
    return y_predictions, uncertainty

def wrap_long_labels(labels, max_length=30):
    """
    Wraps category labels that exceed a certain length to ensure readability.
    """
    for idx, label in enumerate(labels):
        if len(label) > max_length:
            space_indices = np.array([m.start() for m in re.finditer(' ', label)])
            if space_indices.any():
                middle_index = int(len(label) / 2)
                split_index = space_indices[np.argmin(abs(space_indices - middle_index))]
                labels[idx] = label[:split_index] + '\n' + label[split_index+1:]
    return labels

def round_up(value, buffer=True):
    """
    Rounds up a value to a safe interval for better plotting and readability.
    """
    if value > 500:
        return ((value // 100) + 1) * 100
    elif value > 200:
        return ((value // 50) + 1) * 50
    elif value > 100:
        return ((value // 20) + 1) * 20
    elif value > 20:
        return ((value // 10) + 1) * 10
    elif value > 1:
        return _round_small_values(value, buffer)
    elif value > .25:
        return ((value // .25) + 1) * .25
    elif value > .05:
        return ((value // .05) + 1) * .05
    else:
        return ((value // .01) + 1) * .01

def _round_small_values(value, buffer):
    """
    Helper function for rounding small values based on custom conditions.
    """
    if buffer:
        if value > 5:
            return ((value // 2) + 1) * 2
        elif value > 1.5:
            return ((value // 1) + 1) * 1
        elif value > 1:
            return ((value // 1) + 1) * 1
    else:
        return round(value)

def plot_horizontal_lines(plot, ymax, zero_shift=0):
    """
    Adds horizontal reference lines at ymax and ymax/2 for visual aid in plots.
    """
    y_ticks = [0, ymax / 2, ymax]
    y_ticks[0] += zero_shift
    plot.yticks(y_ticks, [str(int(y)) for y in y_ticks])
    plot.axhline(ymax / 2, linestyle='-', linewidth=.8)
    plot.axhline(ymax, linestyle='-', linewidth=.8)

def remove_plot_spines(axis):
    """
    Removes unnecessary plot spines (top, right) to clean up plot appearance.
    """
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(True)
    axis.spines['left'].set_visible(False)

def add_bottom_text(plot, n_obs=True, font_size=40):
    """
    Adds a text label and optionally the number of observations at the bottom of the plot.
    """
    plot.figtext(0.5, 0.01, BOTTOM_TEXT_LABEL, horizontalalignment='center', fontsize=font_size, color=dark_gray, family='Muli')
    if n_obs:
        plot.gcf().text(0.85, 0.01, 'N = {}'.format(n_obs), fontsize=font_size, color=dark_gray, family='Muli')


# In[ ]:


# Plot results with visible uncertainty (confidence intervals)
fig, ax = plt.subplots(figsize=(26, 20))

# Investment data plot (green line for mean prediction)
plt.fill(np.concatenate([x_mesh, x_mesh[::-1]]),
         np.concatenate([predictions_investment - confidence_interval_investment * scaled_uncertainty_investment,
                         (predictions_investment + confidence_interval_investment * scaled_uncertainty_investment)[::-1]]),
         alpha=.4, fc='lightgreen', ec='None', label='Confidence Interval')  # Adjusted opacity and color for visibility

# Mean prediction (green line)
plt.plot(x_mesh, predictions_investment, label='Investment', color='#007041', lw=2)

# Update the legend and text color
legend = plt.legend(bbox_to_anchor=(.5, 1), loc="upper center", frameon=False, ncol=2, markerscale=0.008, handletextpad=0.5, prop={'size': 48}, bbox_transform=plt.gcf().transFigure)
plt.setp(legend.get_texts(), color='#0000FF')  # Changed to blue for the text

# Formatting plot
max_y_value = max(predictions_investment + scaled_uncertainty_investment)
ymax = round_up(max_y_value)
remove_plot_spines(ax)
plot_horizontal_lines(plt, ymax)

plt.tick_params(axis='both', which='major', labelsize=50, color='#FF0000', length=0, pad=15)  # Changed to red for tick color
#add_bottom_text(plt, n_obs=False)
ax.margins(x=0)
plt.ylim(ymin=0)

plt.show()

