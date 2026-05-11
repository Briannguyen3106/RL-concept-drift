import pandas as pd
import numpy as np
from typing import Generator, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_PATH, TARGET_COL, DROP_COLS,
    CATEGORICAL_COLS, NUMERICAL_COLS,
    INITIAL_TRAIN_SIZE, BATCH_SIZE, RANDOM_SEED,
    STREAM_TRAIN_RATIO
)


class AirlinesDataLoader:
    """
    Load, preprocess và stream Airlines dataset.
    Categorical columns được giữ nguyên dưới dạng 'category' dtype
    để LightGBM xử lý natively.

    Stream được chia thành 2 phần:
        - Train stream (STREAM_TRAIN_RATIO): agent training
        - Test stream  (1-STREAM_TRAIN_RATIO): evaluation

    Usage:
        loader = AirlinesDataLoader()
        X_train, y_train = loader.get_initial_train_data()

        # Agent training
        for X_batch, y_batch, idx in loader.stream_train_batches():
            ...

        # Warm-up data (N batches cuối train stream)
        X_warm, y_warm = loader.get_warmup_data(n_batches=10)

        # Evaluation
        for X_batch, y_batch, idx in loader.stream_test_batches():
            ...
    """

    def __init__(self):
        self.feature_cols: list[str] = CATEGORICAL_COLS + NUMERICAL_COLS
        self.X_train  : pd.DataFrame = None
        self.y_train  : pd.Series    = None
        self.X_stream : pd.DataFrame = None
        self.y_stream : pd.Series    = None

        # Tính train/test split boundary (làm tròn theo batch)
        self._load_and_preprocess()
        total_batches       = len(self.X_stream) // BATCH_SIZE
        self._n_train_batches = int(total_batches * STREAM_TRAIN_RATIO)
        self._n_test_batches  = total_batches - self._n_train_batches
        self._train_end_idx   = self._n_train_batches * BATCH_SIZE

        print(f"[DataLoader] Stream split → "
              f"Train: {self._n_train_batches} batches | "
              f"Test: {self._n_test_batches} batches")

    # ------------------------------------------------------------------ #
    #  PRIVATE                                                             #
    # ------------------------------------------------------------------ #

    def _load_and_preprocess(self) -> None:
        """Load CSV và thực hiện preprocessing."""
        print(f"[DataLoader] Loading data from: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        print(f"[DataLoader] Raw shape: {df.shape}")

        df = df.drop(columns=DROP_COLS, errors="ignore")

        for col in CATEGORICAL_COLS:
            df[col] = df[col].astype("category")
            print(f"[DataLoader] '{col}' → category dtype "
                  f"({df[col].nunique()} unique values)")

        X = df[self.feature_cols]
        y = df[TARGET_COL]

        self.X_train  = X.iloc[:INITIAL_TRAIN_SIZE].reset_index(drop=True)
        self.y_train  = y.iloc[:INITIAL_TRAIN_SIZE].reset_index(drop=True)
        self.X_stream = X.iloc[INITIAL_TRAIN_SIZE:].reset_index(drop=True)
        self.y_stream = y.iloc[INITIAL_TRAIN_SIZE:].reset_index(drop=True)

        print(f"[DataLoader] Initial train size : {len(self.X_train):,}")
        print(f"[DataLoader] Stream size        : {len(self.X_stream):,}")
        print(f"[DataLoader] Class distribution (train) – "
              f"0: {(self.y_train == 0).sum():,} | "
              f"1: {(self.y_train == 1).sum():,}")

    def _make_generator(
        self,
        start_batch : int,
        end_batch   : int,
        batch_size  : int = BATCH_SIZE
    ) -> Generator[Tuple[pd.DataFrame, pd.Series, int], None, None]:
        """Helper tạo generator từ start_batch đến end_batch."""
        for i in range(start_batch, end_batch):
            start = i * batch_size
            end   = start + batch_size
            yield (
                self.X_stream.iloc[start:end].reset_index(drop=True),
                self.y_stream.iloc[start:end].reset_index(drop=True),
                i
            )

    # ------------------------------------------------------------------ #
    #  PUBLIC – Data access                                                #
    # ------------------------------------------------------------------ #

    def get_initial_train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Trả về (X_train, y_train) để train model lần đầu."""
        return self.X_train, self.y_train

    def get_warmup_data(
        self,
        n_batches: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Lấy N batches cuối của train stream để warm-up model
        trước khi bắt đầu evaluate.

        Dùng để đảm bảo tất cả strategies (static, periodic, MC agent...)
        bắt đầu evaluate với model được train trên cùng data gần nhất.

        Args:
            n_batches: số batches cuối train stream cần lấy

        Returns:
            (X_warmup, y_warmup): data để retrain model
        """
        n_batches = min(n_batches, self._n_train_batches)
        start_idx = (self._n_train_batches - n_batches) * BATCH_SIZE
        end_idx   = self._train_end_idx

        X_warm = self.X_stream.iloc[start_idx:end_idx].reset_index(drop=True)
        y_warm = self.y_stream.iloc[start_idx:end_idx].reset_index(drop=True)

        for col in CATEGORICAL_COLS:
            X_warm[col] = X_warm[col].astype("category")

        return X_warm, y_warm

    # ------------------------------------------------------------------ #
    #  PUBLIC – Stream generators                                          #
    # ------------------------------------------------------------------ #

    def stream_batches(
        self,
        batch_size: int = BATCH_SIZE
    ) -> Generator[Tuple[pd.DataFrame, pd.Series, int], None, None]:
        """Toàn bộ stream – dùng cho baselines và quick test."""
        yield from self._make_generator(0, self.n_batches, batch_size)

    def stream_train_batches(
        self,
        batch_size: int = BATCH_SIZE
    ) -> Generator[Tuple[pd.DataFrame, pd.Series, int], None, None]:
        """
        Train stream (STREAM_TRAIN_RATIO đầu) – dùng để train agent.
        batch_idx bắt đầu từ 0.
        """
        yield from self._make_generator(0, self._n_train_batches, batch_size)

    def stream_test_batches(
        self,
        batch_size: int = BATCH_SIZE
    ) -> Generator[Tuple[pd.DataFrame, pd.Series, int], None, None]:
        """
        Test stream (phần còn lại) – dùng để evaluate.
        batch_idx tiếp tục từ _n_train_batches để giữ temporal order.
        """
        yield from self._make_generator(
            self._n_train_batches, self.n_batches, batch_size
        )

    # ------------------------------------------------------------------ #
    #  PROPERTIES                                                          #
    # ------------------------------------------------------------------ #

    @property
    def n_batches(self) -> int:
        """Tổng số batches trong toàn bộ stream."""
        return len(self.X_stream) // BATCH_SIZE

    @property
    def n_train_batches(self) -> int:
        """Số batches trong train stream."""
        return self._n_train_batches

    @property
    def n_test_batches(self) -> int:
        """Số batches trong test stream."""
        return self._n_test_batches

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)


# ------------------------------------------------------------------ #
#  QUICK TEST                                                          #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    loader = AirlinesDataLoader()

    print(f"\nTotal batches  : {loader.n_batches}")
    print(f"Train batches  : {loader.n_train_batches}")
    print(f"Test batches   : {loader.n_test_batches}")

    # Test warm-up data
    X_warm, y_warm = loader.get_warmup_data(n_batches=10)
    print(f"\nWarm-up data   : {X_warm.shape}")

    # Test train stream
    print("\nTrain stream (first 2 batches):")
    for X_batch, y_batch, idx in loader.stream_train_batches():
        print(f"  Batch {idx:>3} | X: {X_batch.shape}")
        if idx == 1:
            break

    # Test test stream
    print("\nTest stream (first 2 batches):")
    for X_batch, y_batch, idx in loader.stream_test_batches():
        print(f"  Batch {idx:>3} | X: {X_batch.shape}")
        if idx == loader.n_train_batches + 1:
            break