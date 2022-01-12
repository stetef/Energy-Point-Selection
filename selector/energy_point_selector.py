"""Energy point selector."""

import numpy as np

from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, RepeatedKFold


class Selector:
    """Select the features that most describe the reference standards."""

    def __init__(self, references, coeffs):
        """Init function."""
        self.References = references
        self.Coeffs = coeffs

    def select_energy_points(self, n_points=None, estimator='dt', **kwargs):
        """Return the n_points that are most important."""
        if n_points is None:
            n_points = len(self.References)

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
        rfe = rfe.fit(self.References, self.Coeffs)

        scores = self.evaluate_rfe(rfe, **kwargs)
        return rfe, scores

    def evaluate_rfe(self, rfe, **kwargs):
        """Evaluate model using cross validation."""
        cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
        n_scores = cross_val_score(rfe.estimator_, self.References,
                                   self.Coeffs, cv=cv, **kwargs)
        print('Score: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
        return np.mean(n_scores)
