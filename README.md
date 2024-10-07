# Investment Trend Prediction with Uncertainty Estimation

This project focused on predicting investment trends over time using a combination of Linear Regression and Gaussian Process Regression (GPR). The GPR model allows for the estimation of uncertainty, providing confidence intervals around the predictions to better understand the model's confidence in its forecasts.

Benefits of Using Both Linear regression and GPR:
Improved Accuracy: Linear regression gets you a simple, initial model, but GPR takes this further by improving accuracy and handling non-linearity in the data.
Robustness: Combining both models makes your approach more robust. Linear regression provides a fast baseline, while GPR handles the complexity, allowing for better predictions in uncertain or fluctuating environments.
Uncertainty Management: GPR’s ability to generate uncertainty estimates makes the predictions more informative and useful for decision-making. In areas like investment, knowing the confidence in the prediction helps in risk management and strategic planning.

Features
Data Processing: Handles investment data, performs linear regression to estimate trends, and applies GPR for more accurate predictions.
Uncertainty Estimation: Visualizes uncertainty (confidence intervals) around the predictions, offering insight into the model's confidence levels.
Visualization: Generates clear visualizations of the investment trends and confidence intervals using Matplotlib.
Key Concepts
Linear Regression: Provides a quick and simple trend estimate.
Gaussian Process Regression (GPR): Captures non-linear patterns and provides uncertainty estimates for predictions.
Confidence Intervals: Shaded regions around the prediction line, representing the uncertainty of the predictions.

![example ](https://github.com/user-attachments/assets/ed410bc1-c905-4e4b-ad96-b30b57fd11fc)


