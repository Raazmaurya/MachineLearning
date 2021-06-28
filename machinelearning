{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "0976e3b9",
   "metadata": {},
   "outputs": [],
   "source": [
    "## wine quality prediction\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d41980b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd \n",
    "from sklearn.model_selection import train_test_split \n",
    "from sklearn.linear_model import LinearRegression \n",
    "from sklearn import metrics \n",
    "import matplotlib.pyplot as plt \n",
    "import numpy as np \n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bc6f55cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('winequality-red.csv',sep=\";\")\n",
    "df.head()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0def013f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# there are no categorical variables. each feature is a number. Regression problem. \n",
    "# Given the set of values for features, we have to predict the quality of wine. finding correlation of each feature with our target variable - quality\n",
    "correlations = df.corr()['quality'].drop('quality')\n",
    "print(correlations)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b57d2868",
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.heatmap(df.corr())\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "77533d11",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_features(correlation_threshold):\n",
    "    abs_corrs = correlations.abs()\n",
    "    high_correlations = abs_corrs\n",
    "    [abs_corrs > correlation_threshold].index.values.tolist()\n",
    "    return high_correlations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "accd5336",
   "metadata": {},
   "outputs": [],
   "source": [
    "# taking features with correlation more than 0.05 as input x and quality as target variable y \n",
    "features = get_features(0.05) \n",
    "print(features) \n",
    "x = df[features] \n",
    "y = df['quality']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ad4e0ff4",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "97455f40",
   "metadata": {},
   "outputs": [],
   "source": [
    "# fitting linear regression to training data\n",
    "regressor = LinearRegression()\n",
    "regressor.fit(x_train,y_train)\n",
    "  \n",
    "# this gives the coefficients of the 10 features selected above.  print(regressor.coef_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a6247a6c",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_pred = regressor.predict(x_train)\n",
    "print(train_pred)\n",
    "test_pred = regressor.predict(x_test) \n",
    "print(test_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9e6ddc27",
   "metadata": {},
   "outputs": [],
   "source": [
    "# calculating rmse\n",
    "train_rmse = mean_squared_error(train_pred, y_train) ** 0.5\n",
    "print(train_rmse)\n",
    "test_rmse = mean_squared_error(test_pred, y_test) ** 0.5\n",
    "print(test_rmse)\n",
    "# rounding off the predicted values for test set\n",
    "predicted_data = np.round_(test_pred)\n",
    "print(predicted_data)\n",
    "print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_pred))\n",
    "print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_pred))\n",
    "print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))\n",
    "# displaying coefficients of each feature\n",
    "coeffecients = pd.DataFrame(regressor.coef_,features) coeffecients.columns = ['Coeffecient'] \n",
    "print(coeffecients)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
