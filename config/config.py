# ============================================================
# BREAST CANCER XAI - SYSTEM CONFIGURATION
# Explainable Deep Learning Framework for Histopathology
# ============================================================

import os

# ============================================================
# DIRECTORY & PATH DEFINITIONS
# ============================================================

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CONFIG_DIR)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
GRADCAM_DIR = os.path.join(STATIC_DIR, "gradcam")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Ensure required directories exist
for directory in [UPLOAD_DIR, GRADCAM_DIR, RESULTS_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)


# ============================================================
# IMAGE & PREPROCESSING PARAMETERS
# ============================================================

IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
CHANNELS = 3
BATCH_SIZE = 32
RANDOM_SEED = 42

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ============================================================
# HISTOPATHOLOGY CLASS DEFINITIONS (BreakHis 400X)
# ============================================================

CLASS_NAMES = [
    "adenosis",
    "ductal_carcinoma",
    "fibroadenoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma",
    "phyllodes_tumor",
    "tubular_adenoma",
]

NUM_CLASSES = len(CLASS_NAMES)

BENIGN_CLASSES = {
    "adenosis",
    "fibroadenoma",
    "phyllodes_tumor",
    "tubular_adenoma",
}

MALIGNANT_CLASSES = {
    "ductal_carcinoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma",
}

CLASS_DESCRIPTIONS = {
    "adenosis": {
        "type": "Benign",
        "description": "Enlarged lobules containing more glands than usual, presenting benign proliferative change.",
        "risk_level": "Low / Routine Surveillance",
    },
    "fibroadenoma": {
        "type": "Benign",
        "description": "Most common benign tumor of the breast, composed of epithelial and stromal tissue with sharp borders.",
        "risk_level": "Low / Regular Clinical Follow-Up",
    },
    "phyllodes_tumor": {
        "type": "Benign / Borderline",
        "description": "Fibroepithelial neoplasm with leaf-like stromal overgrowth, requires histological margin evaluation.",
        "risk_level": "Moderate / Surgical Excision Recommended",
    },
    "tubular_adenoma": {
        "type": "Benign",
        "description": "Benign epithelial tumor consisting of packed, uniform tubules lined by epithelial and myoepithelial cells.",
        "risk_level": "Low / Observation",
    },
    "ductal_carcinoma": {
        "type": "Malignant",
        "description": "Most common form of invasive breast cancer originating in the milk ducts and invading surrounding stroma.",
        "risk_level": "High / Immediate Oncology Consultation & Multidisciplinary Therapy",
    },
    "lobular_carcinoma": {
        "type": "Malignant",
        "description": "Invasive carcinoma originating in breast lobules characterized by single-file cellular infiltration pattern.",
        "risk_level": "High / Urgent Surgical & Systemic Oncology Evaluation",
    },
    "mucinous_carcinoma": {
        "type": "Malignant",
        "description": "Specialized mucin-producing adenocarcinoma where malignant cells float in pools of extracellular mucus.",
        "risk_level": "High / Surgical Resection & Targeted Chemotherapy",
    },
    "papillary_carcinoma": {
        "type": "Malignant",
        "description": "Rare invasive breast carcinoma with finger-like papillae and fibrovascular cores with cellular atypia.",
        "risk_level": "High / Comprehensive Staging & Treatment Protocol",
    },
}


# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

MODEL_CONFIG = {
    "resnet50": {
        "name": "ResNet50",
        "file": "resnet50.keras",
        "class_names_file": "resnet_class_names.json",
        "architecture": "Residual Network (50 layers)",
        "features": "Deep residual skip connections mitigating vanishing gradients",
        "description": "Transfer learning backbone fine-tuned for 8-class histopathology classification.",
        "accuracy": 84.41,
        "auc": 98.35,
    },
    "cnn": {
        "name": "Custom CNN",
        "file": "cnn_model.keras",
        "class_names_file": "class_names.json",
        "architecture": "4-Stage Sequential Convolutional Neural Network",
        "features": "Conv2D + BatchNorm + MaxPool + Dropout + GAP",
        "description": "Custom designed baseline CNN optimized directly on histopathology features.",
        "accuracy": 80.49,
        "auc": 97.95,
    },
    "efficientnetb0": {
        "name": "EfficientNetB0",
        "file": "efficientnetb0.keras",
        "class_names_file": "efficientnet_class_names.json",
        "architecture": "Compound Scaled Inverted Residual Network (MBConv)",
        "features": "Squeeze-and-Excitation attention with compound depth/width scaling",
        "description": "Lightweight, highly parameter-efficient model for rapid edge inference.",
        "accuracy": 60.14,
        "auc": 89.14,
    },
}


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", 3306))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "root")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "breast_cancer_xai")

DB_CONFIG = {
    "host": MYSQL_HOST,
    "port": MYSQL_PORT,
    "user": MYSQL_USER,
    "password": MYSQL_PASSWORD,
    "database": MYSQL_DATABASE,
}

SQLITE_PATH = os.path.join(BASE_DIR, "database.db")


# ============================================================
# APPLICATION SECURITY & SERVER SETTINGS
# ============================================================

SECRET_KEY = os.environ.get("SECRET_KEY", "breast-cancer-xai-research-secret-2026")
SESSION_MAX_AGE = 60 * 60 * 24  # 24 Hours
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
