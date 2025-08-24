import pickle
import random
from sklearn.base import BaseEstimator

# Dummy model class
class DummyLoanModel(BaseEstimator):
    def predict(self, X):
        return [random.choice([0, 1]) for _ in range(len(X))]

# Model object
model = DummyLoanModel()

# Save to model.pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Dummy model saved as model.pkl")

