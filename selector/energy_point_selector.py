"""Energy point selector using RFE regression."""

import numpy as np

from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, RepeatedKFold


class Selector:
    """Select the features that most describe the reference standards."""

    def __init__(self, data, coeffs):
        """
        Init function.

        Attributes:
            Data - Spectra that are made of linear combinations of
                reference spectra.
            Coefficients - The coefficients of each reference that
                generated the dataset.  Must add to one.
        """
        self.Data = data
        self.Coeffs = coeffs

    def select_energy_points(self, n_points=None, estimator='dt', **kwargs):
        """Return the n_points that are most important."""
        if n_points is None:
            n_points = len(self.Data)

        if estimator.lower().replace(' ', '') in ['linear',
                                                  'linearregression']:
            model = LinearRegression()

        elif estimator.lower().replace(' ', '') in ['dt', 'decisiontree']:
            model = DecisionTreeRegressor()

        elif estimator.lower().replace(' ', '') in ['rf', 'randomforest']:
            model = RandomForestRegressor()

        else:
            print("Estimator model not supported. " +
                  "Setting to default Decision Tree Regressor.")
            model = DecisionTreeRegressor()

        rfe = RFE(model, n_features_to_select=n_points, step=1)
        self.rfe = rfe.fit(self.Data, self.Coeffs)

        return self.rfe, self.evaluate_rfe(**kwargs)

    def evaluate_rfe(self, **kwargs):
        """Evaluate model using cross validation."""
        cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
        n_scores = cross_val_score(self.rfe.estimator_, self.Data,
                                   self.Coeffs, cv=cv, **kwargs)
        print('Score: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
        return np.mean(n_scores)
