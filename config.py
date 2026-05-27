import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR     = os.path.join(BASE_DIR, "data")
RESULTS_DIR  = os.path.join(BASE_DIR, "experiments", "results")
MODELS_DIR   = os.path.join(BASE_DIR, "experiments", "saved_models")

DATA_PATH    = os.path.join(DATA_DIR, "Airlines.csv")

# Tự động tạo thư mục nếu chưa có
for _dir in [DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(_dir, exist_ok=True)


# ============================================================
# STREAM SETTINGS
# ============================================================
INITIAL_TRAIN_SIZE = 20_000      # Số samples dùng để train lần đầu
BATCH_SIZE         = 1_000      # Số samples mỗi batch khi stream
# STREAM SPLIT
STREAM_TRAIN_RATIO = 0.7   # 70% stream để train agent, 30% để evaluate

# ============================================================
# METRICS SETTINGS
# ============================================================
WINDOW_SIZE = 500               # Rolling window để tính accuracy / error rate


# ============================================================
# DATASET SCHEMA
# ============================================================
TARGET_COL       = "Delay"
DROP_COLS        = ["id"]                               # Không có giá trị dự đoán
CATEGORICAL_COLS = ["Airline", "AirportFrom", "AirportTo"]
NUMERICAL_COLS   = ["Flight", "DayOfWeek", "Time", "Length"]

# ============================================================
# DRIFT MEASURE
# ============================================================
# Chọn drift measure dùng để build state cho RL Agent
# Các giá trị hợp lệ: "adwin" | "ddm" | "psi" | "kl"
# Đổi giá trị này để chạy experiment khác nhau
DRIFT_MEASURE = "psi"
 
# PSI settings
PSI_BINS = 10                   # Số bins để tính PSI
 
# ADWIN settings
ADWIN_DELTA = 0.002             # Độ nhạy: nhỏ hơn → phát hiện drift sớm hơn
 
# DDM settings
DDM_WARNING_LEVEL = 2.0         # Ngưỡng cảnh báo (std deviations)
DDM_DRIFT_LEVEL   = 3.0         # Ngưỡng drift (std deviations)
 
# KL Divergence settings
KL_BINS     = 10                # Số bins để tính KL divergence
KL_EPSILON  = 1e-10             # Tránh log(0)

#=============================================================
# REWARD FUNCTION
#=============================================================
REWARD_ACCURACY_WEIGHT = 1.0

ACTION_COSTS = {
    'no_action' : 0.0,
    'partial_update': 0.2,
    'full_retrain': 0.3,
}

# Penalty khi bỏ lỡ drift
DRIFT_MISS_PENALTY    = 0.3    # Mức phạt
DRIFT_MISS_THRESHOLD  = 0.50    # Error rate ngưỡng để coi là bỏ lỡ drift

# Checkpoint: lưu model khi accuracy vượt ngưỡng này
CHECKPOINT_THRESHOLD = 0.75

# Partial vs Full retrain window size (tính bằng số batches)
PARTIAL_WINDOW_BATCHES = 2
FULL_WINDOW_BATCHES    = 10
UNCERTAINTY_BINS = [0.0, 0.25, 0.50, 1.0] 

# EXPLORATION
EXPLORATION_STRATEGY = "epsilon_greedy"  # "epsilon_greedy" | "gradient_bandit"

# Epsilon-greedy
EPSILON_START  = 0.5
EPSILON_END    = 0.1
EPSILON_DECAY  = 0.995   # Mỗi episode nhân với decay

# Gradient bandit
GRADIENT_BANDIT_ALPHA = 0.1   # Learning rate cho preference update
GRADIENT_BANDIT_BETA = 0.1    # Learning rate cho baseline R_bar update

#=============================================================
#Agents
#=============================================================
# MC Agent
MC_GAMMA   = 0.9    # Discount factor
MC_ALPHA   = 0.1    # Learning rate cho H update
MC_TEMP_START = 2.0    # Temperature ban đầu (explore nhiều)
MC_TEMP_END   = 0.5    # Temperature cuối (exploit nhiều)
MC_TEMP_DECAY = 0.995   # Decay mỗi episode

MC_EPISODE_STARTS = [0, 72, 145, 218, 291]   # ~363 * [0, .2, .4, .6, .8]

#=============================================================
# TD AGENTS (SARSA, Expected SARSA, Q-Learning, Double Q-Learning)
#=============================================================
TD_GAMMA         = 0.9
TD_ALPHA         = 0.1
TD_EPSILON       = 0.1    # Cố định, không decay

TD_EPISODE_STARTS = [0, 72, 145, 218, 291]
# ============================================================
# GENERAL
# ============================================================
RANDOM_SEED = 42