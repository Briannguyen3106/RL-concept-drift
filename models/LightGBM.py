import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED, CATEGORICAL_COLS

class LightGBM:
    DEFAULT_PARAMS = {
        "objective"        : "binary",
        "metric"           : "binary_error",
        "verbosity"        : -1,          # Tắt log của LightGBM
        "random_state"     : RANDOM_SEED,
        "n_estimators"     : 100,
        "learning_rate"    : 0.05,
        "max_depth"        : -1,
        "num_leaves"       : 31,
        "min_child_samples": 20,
    }

    def __init__(self, params = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = None
        self.is_trained = False
        self.n_update = 0

    def _build_model(self):
        return lgb.LGBMClassifier(**self.params)
    
    def train(self, X, y):
        print(f"[Classifier] Training on {len(X):,} samples...")
        self.model = self._build_model()
        self.model.fit(
            X, y,
            categorical_feature=CATEGORICAL_COLS
        )
        self.is_trained = True
        self.n_updates  = 0
        print(f"[Classifier] Training complete.")

    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("model has not trainned")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise RuntimeError("Model chưa được train. Gọi train() trước.")
        return self.model.predict_proba(X)[:, 1]
    
    def full_retrain(self, X, y):
        self.model = self._build_model()
        self.model.fit(
            X, y,
            categorical_feature=CATEGORICAL_COLS
        )
        self.is_trained = True
        self.n_updates += 1

    def partial_update(self, X, y):
        self.model = self._build_model()
        self.model.fit(
            X, y,
            categorical_feature=CATEGORICAL_COLS
        )
        self.n_updates += 1
        
    def get_error_rate(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Tính error rate trên một batch.
        Được dùng bởi stream_metrics và drift_env để tính state.
 
        Args:
            X: Features batch
            y: True labels
 
        Returns:
            float: tỉ lệ dự đoán sai trong [0, 1]
        """
        preds = self.predict(X)
        return float(np.mean(preds != y.values))
    
    @property
    def feature_importances(self) -> Optional[dict]:
        """Trả về feature importances nếu model đã train."""
        if not self.is_trained:
            return None
        return dict(zip(
            self.model.feature_name_,
            self.model.feature_importances_
        ))
    

# ------------------------------------------------------------------ #
#  QUICK TEST                                                          #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.data_loader import AirlinesDataLoader
 
    loader = AirlinesDataLoader()
    X_train, y_train = loader.get_initial_train_data()
 
    clf = LightGBM()
    clf.train(X_train, y_train)
 
    # Test trên batch đầu tiên
    for X_batch, y_batch, idx in loader.stream_batches():
        error = clf.get_error_rate(X_batch, y_batch)
        print(f"Batch {idx} | Error rate: {error:.4f} | Accuracy: {1-error:.4f}")
        if idx == 4:
            break
 
    # Test partial update
    print("\n--- Partial update test ---")
    X_recent = loader.X_stream.iloc[:2000]
    y_recent = loader.y_stream.iloc[:2000]
    clf.partial_update(X_recent, y_recent)
 
    print(f"\nFeature importances:")
    for feat, imp in sorted(clf.feature_importances.items(),
                            key=lambda x: x[1], reverse=True):
        print(f"  {feat:>15}: {imp:.4f}")