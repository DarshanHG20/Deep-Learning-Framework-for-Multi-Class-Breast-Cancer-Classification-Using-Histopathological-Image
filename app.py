# ============================================================
# BREAST CANCER XAI
# FASTAPI COMPLETE END-TO-END APPLICATION
# ============================================================

import os
import uuid
import math
import json
import traceback

import cv2
import numpy as np
import mysql.connector
import tensorflow as tf

from fastapi import (
    FastAPI,
    Request,
    Form,
    UploadFile,
    File,
    HTTPException,
)

from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    Response,
)

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from starlette.middleware.sessions import SessionMiddleware

from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)

from werkzeug.utils import secure_filename


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TEMPLATES_DIR = os.path.join(
    BASE_DIR,
    "templates"
)

STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)

UPLOAD_DIR = os.path.join(
    STATIC_DIR,
    "uploads"
)

GRADCAM_DIR = os.path.join(
    STATIC_DIR,
    "gradcam"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "saved_models"
)


# Create required directories

os.makedirs(
    TEMPLATES_DIR,
    exist_ok=True
)

os.makedirs(
    STATIC_DIR,
    exist_ok=True
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    GRADCAM_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Breast Cancer XAI",
    description=(
        "An Explainable Deep Learning Framework "
        "for Multi-Class Breast Cancer Classification "
        "Using Histopathological Images."
    ),
    version="1.0.0",
)


# ============================================================
# SESSION MIDDLEWARE
# ============================================================

app.add_middleware(
    SessionMiddleware,
    secret_key="breast-cancer-xai-secret-key-2026",
    max_age=60 * 60 * 24,
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=STATIC_DIR
    ),
    name="static",
)


# ============================================================
# JINJA2 TEMPLATE CONTEXT
# ============================================================
#
# This fixes:
#
# UndefinedError: 'session' is undefined
#
# and also makes:
#
# user
# session
# flashes
#
# available in every template.
# ============================================================

def template_context_processor(
    request: Request
):
    def safe_url_for(name: str, **path_params):
        if name == "static":
            if "filename" in path_params and "path" not in path_params:
                path_params["path"] = path_params.pop("filename")
            if "path" in path_params and isinstance(path_params["path"], str):
                if path_params["path"].startswith("static/"):
                    path_params["path"] = path_params["path"][7:]
        return str(request.url_for(name, **path_params))

    return {
        "request": request,
        "url_for": safe_url_for,
        "session": request.session,
        "user": request.session.get(
            "username"
        ),
        "flashes": request.session.get(
            "_flashes",
            []
        ),
    }


templates = Jinja2Templates(
    directory=TEMPLATES_DIR,
    context_processors=[
        template_context_processor
    ],
)


# ============================================================
# MYSQL CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "root",
    "database": "breast_cancer_xai",
}


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db():

    return mysql.connector.connect(
        **DB_CONFIG
    )


# ============================================================
# CLASS NAMES
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


# ============================================================
# BENIGN / MALIGNANT CLASSES
# ============================================================

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


# ============================================================
# MODEL CONFIGURATION
# ============================================================
#
# These are the actual trained models in your folder.
#
# NO DENSENET
# NO VGG16
#
# ============================================================

MODEL_CONFIG = {

    "cnn": {

        "name": "CNN",

        "file": "cnn_model.keras",

        "class_names_file": "class_names.json",

    },

    "resnet50": {

        "name": "ResNet50",

        "file": "resnet50.keras",

        "class_names_file": "resnet_class_names.json",

    },

    "efficientnetb0": {

        "name": "EfficientNetB0",

        "file": "efficientnetb0.keras",

        "class_names_file":
            "efficientnet_class_names.json",

    },

}


# ============================================================
# GLOBAL MODEL STORAGE
# ============================================================

MODELS = {}

MODEL_CLASS_NAMES = {}


# ============================================================
# CURRENT USER
# ============================================================

def get_current_user(
    request: Request
):

    return request.session.get(
        "username"
    )


# ============================================================
# LOGIN REQUIRED
# ============================================================

def require_login(
    request: Request
):

    username = get_current_user(
        request
    )

    if not username:

        return RedirectResponse(
            url="/login",
            status_code=303,
        )

    return None


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_database():

    print()
    print("========================================")
    print("Checking MySQL database...")
    print("========================================")

    conn = None
    cursor = None

    try:
        # Auto-create MySQL database if it does not exist yet
        try:
            temp_config = {k: v for k, v in DB_CONFIG.items() if k != "database"}
            temp_conn = mysql.connector.connect(**temp_config)
            temp_cur = temp_conn.cursor()
            temp_cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}`")
            temp_cur.close()
            temp_conn.close()
        except Exception as db_err:
            print(f"[INFO] Database ensure check: {db_err}")

        conn = get_db()

        cursor = conn.cursor()

        # ----------------------------------------------------
        # USERS
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (

                id INT AUTO_INCREMENT PRIMARY KEY,

                username VARCHAR(100)
                    UNIQUE NOT NULL,

                email VARCHAR(150)
                    UNIQUE NOT NULL,

                password VARCHAR(255)
                    NOT NULL,

                created_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (

                id INT AUTO_INCREMENT PRIMARY KEY,

                username VARCHAR(100)
                    NOT NULL,

                image_name VARCHAR(255)
                    NOT NULL,

                predicted_class VARCHAR(100)
                    NOT NULL,

                confidence DECIMAL(6,2)
                    NOT NULL,

                gradcam_name VARCHAR(255),

                model_name VARCHAR(100)
                    DEFAULT 'ResNet50',

                created_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (username)
                    REFERENCES users(username)
                    ON DELETE CASCADE
            )
            """
        )

        # ----------------------------------------------------
        # Add model_name to old table if necessary
        # ----------------------------------------------------

        cursor.execute(
            """
            SHOW COLUMNS
            FROM predictions
            LIKE 'model_name'
            """
        )

        if cursor.fetchone() is None:

            print(
                "[INFO] Adding model_name column..."
            )

            cursor.execute(
                """
                ALTER TABLE predictions
                ADD COLUMN model_name
                VARCHAR(100)
                DEFAULT 'ResNet50'
                """
            )

        conn.commit()

        print(
            "[OK] MySQL database ready."
        )

    except Exception as e:

        print(
            "[ERROR] Database initialization failed."
        )

        print(e)

        traceback.print_exc()

        raise

    finally:

        if cursor:

            try:
                cursor.close()
            except Exception:
                pass

        if conn:

            try:
                conn.close()
            except Exception:
                pass


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names(
    filename
):

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    if not os.path.exists(path):

        print(
            f"[INFO] {filename} not found. "
            "Using default classes."
        )

        return CLASS_NAMES.copy()

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        # ----------------------------------------------------
        # List
        # ----------------------------------------------------

        if isinstance(data, list):

            if len(data) == 8:

                return [
                    str(x)
                    for x in data
                ]

        # ----------------------------------------------------
        # Dictionary
        # ----------------------------------------------------

        if isinstance(data, dict):

            if all(
                str(key).isdigit()
                for key in data.keys()
            ):

                items = sorted(
                    data.items(),
                    key=lambda x:
                        int(x[0])
                )

            else:

                items = list(
                    data.items()
                )

            names = [
                str(value)
                for _, value in items
            ]

            if len(names) == 8:

                return names

    except Exception as e:

        print(
            f"[WARNING] Could not load "
            f"{filename}: {e}"
        )

    return CLASS_NAMES.copy()


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    print()
    print("========================================")
    print("Loading trained models...")
    print("========================================")

    MODELS.clear()

    MODEL_CLASS_NAMES.clear()

    for model_key, config in MODEL_CONFIG.items():

        model_path = os.path.join(
            MODEL_DIR,
            config["file"]
        )

        print()
        print("----------------------------------------")
        print(
            f"Model: {config['name']}"
        )
        print(
            f"File : {model_path}"
        )
        print("----------------------------------------")

        if not os.path.exists(
            model_path
        ):

            print(
                f"[SKIP] Model file not found: "
                f"{config['file']}"
            )

            continue

        try:

            model = tf.keras.models.load_model(
                model_path,
                compile=False
            )

            class_names = load_class_names(
                config["class_names_file"]
            )

            if len(class_names) != 8:

                class_names = (
                    CLASS_NAMES.copy()
                )

            MODELS[
                model_key
            ] = model

            MODEL_CLASS_NAMES[
                model_key
            ] = class_names

            print(
                f"[OK] {config['name']} "
                "loaded successfully."
            )

            print(
                "Classes:",
                class_names
            )

        except Exception as e:

            print(
                f"[ERROR] Could not load "
                f"{config['name']}"
            )

            print(e)

            traceback.print_exc()

    print()
    print("========================================")
    print(
        "Available models:",
        list(MODELS.keys())
    )
    print("========================================")
    print()


# ============================================================
# AVAILABLE MODELS FOR TEMPLATE
# ============================================================

def get_available_models():

    return [

        {
            "key": key,
            "name": config["name"],
        }

        for key, config
        in MODEL_CONFIG.items()

        if key in MODELS

    ]


# ============================================================
# IMAGE VALIDATION
# ============================================================

ALLOWED_EXTENSIONS = {

    ".jpg",

    ".jpeg",

    ".png",

}

MAX_FILE_SIZE = (
    10 * 1024 * 1024
)


def validate_image_filename(
    filename
):

    if not filename:

        return False

    extension = os.path.splitext(
        filename
    )[1].lower()

    return (
        extension
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image_path
):

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise ValueError(
            "Unable to read image."
        )

    # BGR -> RGB

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # 224 x 224

    image = cv2.resize(
        image,
        (224, 224)
    )

    # float32

    image = image.astype(
        np.float32
    )

    # IMPORTANT:
    #
    # Do NOT divide by 255 here.
    #
    # ResNet50 uses its own preprocessing
    # in the trained architecture.
    #
    # CNN was trained with raw image values.
    #
    # EfficientNet model contains its own
    # preprocessing behavior.

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(
    model_key,
    image_path
):

    if model_key not in MODELS:

        raise ValueError(
            f"Model '{model_key}' "
            "is not loaded."
        )

    model = MODELS[
        model_key
    ]

    class_names = MODEL_CLASS_NAMES.get(
        model_key,
        CLASS_NAMES
    )

    image = preprocess_image(
        image_path
    )

    predictions = model.predict(
        image,
        verbose=0
    )

    predictions = np.asarray(
        predictions
    )[0]

    if len(predictions) != len(
        class_names
    ):

        raise ValueError(
            "Model output class count "
            "does not match class names."
        )

    if (
        np.any(
            np.isnan(predictions)
        )
        or
        np.any(
            np.isinf(predictions)
        )
    ):

        raise ValueError(
            "Model returned invalid "
            "prediction values."
        )

    # --------------------------------------------------------
    # Convert logits to probabilities
    # --------------------------------------------------------

    total = float(
        np.sum(predictions)
    )

    if (
        np.min(predictions) < 0
        or
        not np.isclose(
            total,
            1.0,
            atol=0.01
        )
    ):

        predictions = (
            tf.nn.softmax(
                predictions
            ).numpy()
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    probability_sum = float(
        np.sum(predictions)
    )

    if probability_sum > 0:

        predictions = (
            predictions /
            probability_sum
        )

    # --------------------------------------------------------
    # Best class
    # --------------------------------------------------------

    class_index = int(
        np.argmax(
            predictions
        )
    )

    predicted_class = (
        class_names[
            class_index
        ]
    )

    confidence = float(
        predictions[
            class_index
        ] * 100
    )

    # --------------------------------------------------------
    # All probabilities
    # --------------------------------------------------------

    probabilities = []

    for index, class_name in enumerate(
        class_names
    ):

        probabilities.append(
            {
                "class_name":
                    class_name,

                "probability":
                    float(
                        predictions[index]
                        * 100
                    ),
            }
        )

    return {

        "predicted_class":
            predicted_class,

        "class_index":
            class_index,

        "confidence":
            confidence,

        "probabilities":
            probabilities,

    }


# ============================================================
# COMPUTE GRAD-CAM TENSORS (CNN, RESNET50, EFFICIENTNET)
# ============================================================

def compute_gradcam_tensors(
    model,
    input_tensor,
    class_index
):
    base_model = None
    conv_layer = None
    base_index = None

    # Search for nested pretrained base model (e.g. ResNet50, EfficientNet)
    for idx, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            base_index = idx
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    conv_layer = sub_layer
                    break
            if conv_layer is not None:
                break

    if base_model is not None and conv_layer is not None:
        grad_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=[conv_layer.output, base_model.output]
        )

        x = input_tensor
        for layer in model.layers[:base_index]:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x, training=False)

        with tf.GradientTape() as tape:
            conv_outputs, base_output = grad_model(x, training=False)
            y = base_output
            for layer in model.layers[base_index + 1:]:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                y = layer(y, training=False)
            target = y[:, class_index]

        gradients = tape.gradient(target, conv_outputs)
        if gradients is None:
            raise ValueError("Gradients could not be calculated for nested model.")
        return conv_outputs, gradients

    # Sequential / Direct model (e.g. Custom CNN)
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layer = layer
            break

    if conv_layer is None:
        raise ValueError("No Conv2D layer found in model.")

    with tf.GradientTape() as tape:
        x = input_tensor
        conv_outputs = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x, training=False)
            if layer is conv_layer:
                conv_outputs = x
                tape.watch(conv_outputs)
        target = x[:, class_index]

    gradients = tape.gradient(target, conv_outputs)
    if gradients is None:
        raise ValueError("Gradients could not be calculated for sequential model.")
    return conv_outputs, gradients

# ============================================================
# CREATE GRAD-CAM
# ============================================================

def create_gradcam(
    model_key,
    image_path,
    class_index
):

    if model_key not in MODELS:

        raise ValueError(
            "Selected model is not loaded."
        )

    model = MODELS[
        model_key
    ]

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    original = cv2.imread(
        image_path
    )

    if original is None:

        raise ValueError(
            "Cannot read uploaded image."
        )

    rgb = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    resized = cv2.resize(
        rgb,
        (224, 224)
    )

    input_tensor = np.expand_dims(
        resized.astype(
            np.float32
        ),
        axis=0
    )

    # --------------------------------------------------------
    # Gradients
    # --------------------------------------------------------

    conv_outputs, gradients = compute_gradcam_tensors(
        model,
        input_tensor,
        class_index
    )

    # --------------------------------------------------------
    # Global average pooling
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]

    pooled_gradients = (
        pooled_gradients[0]
    )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_outputs *
        pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    if float(max_value) > 0:

        heatmap = (
            heatmap /
            max_value
        )

    heatmap = heatmap.numpy()

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    heatmap = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    # --------------------------------------------------------
    # Colored heatmap
    # --------------------------------------------------------

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    colored_heatmap = (
        cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        original,
        0.60,
        colored_heatmap,
        0.40,
        0
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    unique_id = uuid.uuid4().hex

    heatmap_filename = (
        unique_id +
        "_heatmap.jpg"
    )

    overlay_filename = (
        unique_id +
        "_gradcam.jpg"
    )

    heatmap_path = os.path.join(
        GRADCAM_DIR,
        heatmap_filename
    )

    overlay_path = os.path.join(
        GRADCAM_DIR,
        overlay_filename
    )

    cv2.imwrite(
        heatmap_path,
        colored_heatmap
    )

    cv2.imwrite(
        overlay_path,
        overlay
    )

    return {

        "heatmap_name":
            heatmap_filename,

        "overlay_name":
            overlay_filename,

    }


# ============================================================
# HOME
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
    name="index"
)
async def index(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ============================================================
# REGISTER GET
# ============================================================

@app.get(
    "/register",
    response_class=HTMLResponse,
    name="register"
)
async def register_page(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={
            "error": None
        }
    )


# ============================================================
# REGISTER POST
# ============================================================

@app.post("/register")
async def register_user(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):

    username = username.strip()

    email = email.strip().lower()

    if not username:

        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "error":
                    "Username is required."
            },
            status_code=400
        )

    if not email:

        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "error":
                    "Email is required."
            },
            status_code=400
        )

    if len(password) < 4:

        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "error":
                    "Password must contain at least 4 characters."
            },
            status_code=400
        )

    conn = None
    cursor = None

    try:

        conn = get_db()

        cursor = conn.cursor()

        # Check existing user

        cursor.execute(
            """
            SELECT id
            FROM users
            WHERE username=%s
            OR email=%s
            """,
            (
                username,
                email
            )
        )

        if cursor.fetchone():

            return templates.TemplateResponse(
                request=request,
                name="register.html",
                context={
                    "error":
                        "Username or email already exists."
                },
                status_code=400
            )

        hashed_password = (
            generate_password_hash(
                password
            )
        )

        cursor.execute(
            """
            INSERT INTO users
            (
                username,
                email,
                password
            )
            VALUES
            (%s, %s, %s)
            """,
            (
                username,
                email,
                hashed_password
            )
        )

        conn.commit()

        return RedirectResponse(
            url="/login",
            status_code=303
        )

    except Exception as e:

        print(
            "[ERROR] Registration:",
            e
        )

        traceback.print_exc()

        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "error":
                    "Registration failed."
            },
            status_code=500
        )

    finally:

        if cursor:

            try:
                cursor.close()
            except Exception:
                pass

        if conn:

            try:
                conn.close()
            except Exception:
                pass


# ============================================================
# LOGIN GET
# ============================================================

@app.get(
    "/login",
    response_class=HTMLResponse,
    name="login"
)
async def login_page(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "error": None
        }
    )


# ============================================================
# LOGIN POST
# ============================================================

@app.post("/login")
async def login_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):

    username = username.strip()

    conn = None
    cursor = None
    user_record = None

    try:

        conn = get_db()

        cursor = conn.cursor(
            dictionary=True
        )

        cursor.execute(
            """
            SELECT *
            FROM users
            WHERE username=%s
            """,
            (
                username,
            )
        )

        user_record = cursor.fetchone()

    except Exception as e:

        print(
            "[ERROR] Login database:",
            e
        )

        traceback.print_exc()

        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "error":
                    "Database connection failed."
            },
            status_code=500
        )

    finally:

        if cursor:

            try:
                cursor.close()
            except Exception:
                pass

        if conn:

            try:
                conn.close()
            except Exception:
                pass

    # --------------------------------------------------------
    # Password verification
    # --------------------------------------------------------

    valid_password = False

    if user_record:

        stored_password = (
            user_record["password"]
        )

        try:

            valid_password = (
                check_password_hash(
                    stored_password,
                    password
                )
            )

        except Exception:

            # Compatibility with old plaintext test user

            valid_password = (
                stored_password == password
            )

    if not user_record or not valid_password:

        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "error":
                    "Invalid username or password."
            },
            status_code=401
        )

    # --------------------------------------------------------
    # SESSION
    # --------------------------------------------------------

    request.session.clear()

    request.session["username"] = (
        user_record["username"]
    )

    request.session["user_id"] = (
        user_record["id"]
    )

    return RedirectResponse(
        url="/dashboard",
        status_code=303
    )


# ============================================================
# LOGOUT
# ============================================================

@app.get(
    "/logout",
    name="logout"
)
async def logout(
    request: Request
):

    request.session.clear()

    return RedirectResponse(
        url="/",
        status_code=303
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.get(
    "/dashboard",
    response_class=HTMLResponse,
    name="dashboard"
)
async def dashboard(
    request: Request
):

    username = get_current_user(
        request
    )

    if not username:

        return RedirectResponse(
            url="/login",
            status_code=303
        )

    conn = None
    cursor = None

    prediction = None

    total_predictions = 0

    benign_count = 0

    malignant_count = 0

    try:

        conn = get_db()

        cursor = conn.cursor(
            dictionary=True
        )

        # Latest prediction

        cursor.execute(
            """
            SELECT
                id,
                image_name AS filename,
                predicted_class AS prediction,
                predicted_class,
                confidence,
                gradcam_name,
                model_name,
                created_at
            FROM predictions
            WHERE username=%s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (
                username,
            )
        )

        prediction = cursor.fetchone()
        if prediction:
            fname = prediction.get("filename") or prediction.get("image_name")
            if fname:
                prediction["image_path"] = f"uploads/{fname}"

        # Total

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM predictions
            WHERE username=%s
            """,
            (
                username,
            )
        )

        row = cursor.fetchone()

        if row:

            total_predictions = int(
                row["total"]
            )

        # Benign

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM predictions
            WHERE username=%s
            AND predicted_class IN
            (
                'adenosis',
                'fibroadenoma',
                'phyllodes_tumor',
                'tubular_adenoma'
            )
            """,
            (
                username,
            )
        )

        row = cursor.fetchone()

        if row:

            benign_count = int(
                row["total"]
            )

        # Malignant

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM predictions
            WHERE username=%s
            AND predicted_class IN
            (
                'ductal_carcinoma',
                'lobular_carcinoma',
                'mucinous_carcinoma',
                'papillary_carcinoma'
            )
            """,
            (
                username,
            )
        )

        row = cursor.fetchone()

        if row:

            malignant_count = int(
                row["total"]
            )

    except Exception as e:

        print(
            "[ERROR] Dashboard:",
            e
        )

        traceback.print_exc()

    finally:

        if cursor:

            try:
                cursor.close()
            except Exception:
                pass

        if conn:

            try:
                conn.close()
            except Exception:
                pass

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={

            "username":
                username,

            "active_model":
                "ResNet50",

            "prediction":
                prediction,

            "total_predictions":
                total_predictions,

            "benign_count":
                benign_count,

            "malignant_count":
                malignant_count,

        }
    )


# ============================================================
# UPLOAD GET
# ============================================================

@app.get(
    "/upload",
    response_class=HTMLResponse,
    name="upload"
)
async def upload_page(
    request: Request
):

    username = get_current_user(
        request
    )

    if not username:

        return RedirectResponse(
            url="/login",
            status_code=303
        )

    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={

            "models":
                get_available_models(),

            "error":
                None,

        }
    )


# ============================================================
# UPLOAD / PREDICT
# ============================================================

@app.post("/upload")
async def upload_image(
    request: Request,
    image: UploadFile = File(...),
    model: str = Form("resnet50"),
    generate_gradcam: bool = Form(False)
):

    username = get_current_user(
        request
    )

    if not username:

        return RedirectResponse(
            url="/login",
            status_code=303
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    if model not in MODELS:

        return templates.TemplateResponse(
            request=request,
            name="upload.html",
            context={
                "models":
                    get_available_models(),
                "error":
                    "Selected model is not available."
            },
            status_code=400
        )

    # --------------------------------------------------------
    # Filename
    # --------------------------------------------------------

    original_filename = secure_filename(
        image.filename or ""
    )

    if not validate_image_filename(
        original_filename
    ):

        return templates.TemplateResponse(
            request=request,
            name="upload.html",
            context={
                "models":
                    get_available_models(),
                "error":
                    "Only JPG, JPEG and PNG images are allowed."
            },
            status_code=400
        )

    # --------------------------------------------------------
    # Read
    # --------------------------------------------------------

    try:

        file_data = await image.read()

    except Exception:

        return templates.TemplateResponse(
            request=request,
            name="upload.html",
            context={
                "models":
                    get_available_models(),
                "error":
                    "Could not read uploaded file."
            },
            status_code=400
        )

    if not file_data:

        return templates.TemplateResponse(
            request=request,
            name="upload.html",
            context={
                "models":
                    get_available_models(),
                "error":
                    "Uploaded file is empty."
            },
            status_code=400
        )

    # --------------------------------------------------------
    # Size
    # --------------------------------------------------------

    if len(file_data) > MAX_FILE_SIZE:

        return templates.TemplateResponse(
            request=request,
            name="upload.html",
            context={
                "models":
                    get_available_models(),
                "error":
                    "File size must be less than 10 MB."
            },
            status_code=400
        )

    # --------------------------------------------------------
    # Unique filename
    # --------------------------------------------------------

    extension = os.path.splitext(
        original_filename
    )[1].lower()

    unique_filename = (
        uuid.uuid4().hex
        + extension
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        unique_filename
    )

    conn = None
    cursor = None

    try:

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        with open(
            file_path,
            "wb"
        ) as f:

            f.write(file_data)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        result = predict_image(
            model,
            file_path
        )

        print()
        print("----------------------------------------")
        print("Prediction")
        print("----------------------------------------")

        print(
            "Model:",
            MODEL_CONFIG[
                model
            ]["name"]
        )

        print(
            "Class:",
            result[
                "predicted_class"
            ]
        )

        print(
            "Confidence:",
            f"{result['confidence']:.2f}%"
        )

        # ----------------------------------------------------
        # Grad-CAM
        # ----------------------------------------------------

        gradcam_name = None

        if generate_gradcam:

            try:

                gradcam_result = (
                    create_gradcam(
                        model,
                        file_path,
                        result[
                            "class_index"
                        ]
                    )
                )

                gradcam_name = (
                    gradcam_result[
                        "overlay_name"
                    ]
                )

                print(
                    "[OK] Grad-CAM generated."
                )

            except Exception as e:

                print(
                    "[WARNING] Grad-CAM failed:"
                )

                print(e)

                traceback.print_exc()

                # Prediction still continues

                gradcam_name = None

        # ----------------------------------------------------
        # Save to MySQL
        # ----------------------------------------------------

        conn = get_db()

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO predictions
            (
                username,
                image_name,
                predicted_class,
                confidence,
                gradcam_name,
                model_name
            )
            VALUES
            (%s, %s, %s, %s, %s, %s)
            """,
            (
                username,

                unique_filename,

                result[
                    "predicted_class"
                ],

                result[
                    "confidence"
                ],

                gradcam_name,

                MODEL_CONFIG[
                    model
                ]["name"],

            )
        )

        prediction_id = (
            cursor.lastrowid
        )

        conn.commit()

        print(
            "[OK] Prediction saved. ID:",
            prediction_id
        )

        return RedirectResponse(
            url=(
                f"/prediction/"
                f"{prediction_id}"
            ),
            status_code=303
        )

    except Exception as e:

        print()
        print("----------------------------------------")
        print("[ERROR] Prediction failed")
        print("----------------------------------------")

        print(e)

        traceback.print_exc()

        if conn:

            try:
                conn.rollback()
            except Exception:
                pass

        if os.path.exists(
            file_path
        ):

            try:
                os.remove(
                    file_path
                )
            except Exception:
                pass

        return templates.TemplateResponse(
            request=request,
            name="upload.html",
            context={
                "models":
                    get_available_models(),
                "error":
                    f"Prediction failed: {e}"
            },
            status_code=500
        )

    finally:

        if cursor:

            try:
                cursor.close()
            except Exception:
                pass

        if conn:

            try:
                conn.close()
            except Exception:
                pass


# ============================================================
# /predict ALIAS
# ============================================================

@app.post(
    "/predict",
    name="predict"
)
async def predict_alias(
    request: Request,
    image: UploadFile = File(...),
    model: str = Form("resnet50"),
    generate_gradcam: bool = Form(False)
):

    return await upload_image(
        request=request,
        image=image,
        model=model,
        generate_gradcam=generate_gradcam
    )


# ============================================================
# PREDICTION RESULT
# ============================================================

@app.get(
    "/prediction/{prediction_id}",
    response_class=HTMLResponse,
    name="prediction"
)
async def prediction_result(
    request: Request,
    prediction_id: int
):

    username = get_current_user(
        request
    )

    if not username:

        return RedirectResponse(
            url="/login",
            status_code=303
        )

    conn = None
    cursor = None
    prediction = None

    try:

        conn = get_db()

        cursor = conn.cursor(
            dictionary=True
        )

        cursor.execute(
            """
            SELECT
                id,
                image_name AS filename,
                predicted_class AS prediction,
                predicted_class,
                confidence,
                gradcam_name,
                model_name,
                created_at
            FROM predictions
            WHERE id=%s
            AND username=%s
            """,
            (
                prediction_id,
                username
            )
        )

        prediction = cursor.fetchone()

    except Exception as e:

        print(
            "[ERROR] Prediction result:",
            e
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail="Could not load prediction result."
        )

    finally:

        if cursor:

            try:
                cursor.close()
            except Exception:
                pass

        if conn:

            try:
                conn.close()
            except Exception:
                pass

    if not prediction:

        raise HTTPException(
            status_code=404,
            detail="Prediction not found."
        )

    image_url = (
        "/static/uploads/"
        + str(
            prediction["filename"]
        )
    )

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={

            "prediction":
                prediction,

            "prediction_id":
                prediction["id"],

            "predicted_class":
                prediction[
                    "predicted_class"
                ],

            "confidence":
                float(
                    prediction[
                        "confidence"
                    ]
                ),

            "model_name":
                prediction[
                    "model_name"
                ]
                or "ResNet50",

            "image_url":
                image_url,

            "filename":
                prediction[
                    "filename"
                ],

        }
    )


# ============================================================
# HISTORY
# ============================================================

@app.get(
    "/history",
    response_class=HTMLResponse,
    name="history"
)
async def history(
    request: Request,
    page: int = 1
):

    username = get_current_user(
        request
    )

    if not username:

        return RedirectResponse(
            url="/login",
            status_code=303
        )

    if page < 1:

        page = 1

    per_page = 10

    offset = (
        page - 1
    ) * per_page

    conn = None
    cursor = None

    total_predictions = 0

    predictions = []

    try:

        conn = get_db()

        cursor = conn.cursor(
            dictionary=True
        )

        # Total

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM predictions
            WHERE username=%s
            """,
            (
                username,
            )
        )

        row = cursor.fetchone()

        if row:

            total_predictions = int(
                row["total"]
            )

        # Page

        cursor.execute(
            """
            SELECT
                id,
                image_name AS filename,
                predicted_class AS prediction,
                confidence,
                model_name,
                created_at
            FROM predictions
            WHERE username=%s
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """,
            (
                username,
                per_page,
                offset
            )
        )

        predictions = (
            cursor.fetchall()
        )

    except Exception as e:

        print(
            "[ERROR] History:",
            e
        )

        traceback.print_exc()

    finally:

        if cursor:

            try:
                cursor.close()
            except Exception:
                pass

        if conn:

            try:
                conn.close()
            except Exception:
                pass

    # --------------------------------------------------------
    # Template values
    # --------------------------------------------------------

    for prediction in predictions:

        prediction[
            "prediction_lower"
        ] = str(
            prediction[
                "prediction"
            ]
        ).lower()

        prediction[
            "conf_percent"
        ] = float(
            prediction[
                "confidence"
            ]
        )

    total_pages = max(
        1,
        math.ceil(
            total_predictions
            / per_page
        )
    )

    if page > total_pages:

        page = total_pages

    pagination = {

        "next_num":
            page + 1
            if page < total_pages
            else None,

        "prev_num":
            page - 1
            if page > 1
            else None,

    }

    benign_count = 0

    malignant_count = 0

    for prediction in predictions:

        predicted_class = (
            prediction[
                "prediction"
            ]
        )

        if predicted_class in (
            BENIGN_CLASSES
        ):

            benign_count += 1

        elif predicted_class in (
            MALIGNANT_CLASSES
        ):

            malignant_count += 1

    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={

            "predictions":
                predictions,

            "page_num":
                page,

            "pagination":
                pagination,

            "total_predictions":
                total_predictions,

            "benign_count":
                benign_count,

            "malignant_count":
                malignant_count,

        }
    )


# ============================================================
# GRAD-CAM PAGE
# ============================================================

@app.get(
    "/gradcam/{prediction_id}",
    response_class=HTMLResponse,
    name="gradcam"
)
async def gradcam_page(
    request: Request,
    prediction_id: int
):

    username = get_current_user(
        request
    )

    if not username:

        return RedirectResponse(
            url="/login",
            status_code=303
        )

    conn = None
    cursor = None
    prediction = None

    try:

        conn = get_db()

        cursor = conn.cursor(
            dictionary=True
        )

        cursor.execute(
            """
            SELECT *
            FROM predictions
            WHERE id=%s
            AND username=%s
            """,
            (
                prediction_id,
                username
            )
        )

        prediction = cursor.fetchone()

    except Exception as e:

        print(
            "[ERROR] Grad-CAM database:",
            e
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail="Could not load Grad-CAM information."
        )

    finally:

        if cursor:

            try:
                cursor.close()
            except Exception:
                pass

        if conn:

            try:
                conn.close()
            except Exception:
                pass

    if not prediction:

        raise HTTPException(
            status_code=404,
            detail="Prediction not found."
        )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    original_url = (
        "/static/uploads/"
        + str(
            prediction[
                "image_name"
            ]
        )
    )

    image_path = os.path.join(
        UPLOAD_DIR,
        prediction[
            "image_name"
        ]
    )

    if not os.path.exists(
        image_path
    ):

        return templates.TemplateResponse(
            request=request,
            name="gradcam.html",
            context={

                "prediction":
                    prediction,

                "predicted_class":
                    prediction[
                        "predicted_class"
                    ],

                "original_image":
                    original_url,

                "overlay_image":
                    None,

                "heatmap_image":
                    None,

                "error":
                    "Original uploaded image was not found.",

            },
            status_code=404
        )

    # --------------------------------------------------------
    # Model key
    # --------------------------------------------------------

    model_name = (
        prediction[
            "model_name"
        ]
        or "ResNet50"
    )

    model_key = None

    for key, config in (
        MODEL_CONFIG.items()
    ):

        if (
            config["name"]
            == model_name
        ):

            model_key = key

            break

    if model_key not in MODELS:

        model_key = "resnet50"

    if model_key not in MODELS:

        return templates.TemplateResponse(
            request=request,
            name="gradcam.html",
            context={

                "prediction":
                    prediction,

                "predicted_class":
                    prediction[
                        "predicted_class"
                    ],

                "original_image":
                    original_url,

                "overlay_image":
                    None,

                "heatmap_image":
                    None,

                "error":
                    "Required model is not available.",

            }
        )

    # --------------------------------------------------------
    # Existing Grad-CAM
    # --------------------------------------------------------

    overlay_name = (
        prediction[
            "gradcam_name"
        ]
    )

    # --------------------------------------------------------
    # Generate if missing
    # --------------------------------------------------------

    if (
        not overlay_name
        or
        not os.path.exists(
            os.path.join(
                GRADCAM_DIR,
                str(
                    overlay_name
                )
            )
        )
    ):

        try:

            prediction_result = (
                predict_image(
                    model_key,
                    image_path
                )
            )

            gradcam_result = (
                create_gradcam(
                    model_key,
                    image_path,
                    prediction_result[
                        "class_index"
                    ]
                )
            )

            overlay_name = (
                gradcam_result[
                    "overlay_name"
                ]
            )

            # Save generated Grad-CAM name

            conn = get_db()

            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE predictions
                SET gradcam_name=%s
                WHERE id=%s
                AND username=%s
                """,
                (
                    overlay_name,
                    prediction_id,
                    username
                )
            )

            conn.commit()

            cursor.close()

            conn.close()

            conn = None
            cursor = None

        except Exception as e:

            print(
                "[ERROR] Grad-CAM generation:",
                e
            )

            traceback.print_exc()

            return templates.TemplateResponse(
                request=request,
                name="gradcam.html",
                context={

                    "prediction":
                        prediction,

                    "predicted_class":
                        prediction[
                            "predicted_class"
                        ],

                    "original_image":
                        original_url,

                    "overlay_image":
                        None,

                    "heatmap_image":
                        None,

                    "error":
                        f"Grad-CAM failed: {e}",

                },
                status_code=500
            )

        finally:

            if cursor:

                try:
                    cursor.close()
                except Exception:
                    pass

            if conn:

                try:
                    conn.close()
                except Exception:
                    pass

    # --------------------------------------------------------
    # URLs
    # --------------------------------------------------------

    overlay_url = (
        "/static/gradcam/"
        + str(
            overlay_name
        )
    )

    return templates.TemplateResponse(
        request=request,
        name="gradcam.html",
        context={

            "prediction":
                prediction,

            "predicted_class":
                prediction[
                    "predicted_class"
                ],

            "original_image":
                original_url,

            "overlay_image":
                overlay_url,

            # Current database stores
            # overlay filename.

            "heatmap_image":
                overlay_url,

        }
    )


# ============================================================
# REPORT
# ============================================================

@app.get(
    "/report/{prediction_id}",
    response_class=HTMLResponse,
    name="report"
)
async def report_page(
    request: Request,
    prediction_id: int
):

    username = get_current_user(
        request
    )

    if not username:

        return RedirectResponse(
            url="/login",
            status_code=303
        )

    conn = None
    cursor = None
    prediction = None

    try:

        conn = get_db()

        cursor = conn.cursor(
            dictionary=True
        )

        cursor.execute(
            """
            SELECT *
            FROM predictions
            WHERE id=%s
            AND username=%s
            """,
            (
                prediction_id,
                username
            )
        )

        prediction = cursor.fetchone()
        if prediction:
            fname = prediction.get("image_name") or prediction.get("filename")
            if fname:
                prediction["image_path"] = f"uploads/{fname}"
                prediction["filename"] = fname

    except Exception as e:

        print(
            "[ERROR] Report:",
            e
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail="Could not load report."
        )

    finally:

        if cursor:

            try:
                cursor.close()
            except Exception:
                pass

        if conn:

            try:
                conn.close()
            except Exception:
                pass

    if not prediction:

        raise HTTPException(
            status_code=404,
            detail="Report not found."
        )

    return templates.TemplateResponse(
        request=request,
        name="report.html",
        context={

            "prediction":
                prediction,

            "confidence_value":
                float(
                    prediction[
                        "confidence"
                    ]
                ),

        }
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

@app.get(
    "/model-comparison",
    response_class=HTMLResponse,
    name="model_comparison"
)
@app.get(
    "/model_comparison",
    response_class=HTMLResponse
)
async def model_comparison(
    request: Request
):

    comparison = [

        {
            "model_name":
                "CNN",

            "accuracy":
                80.49,

            "precision":
                84.67,

            "recall":
                80.49,

            "f1":
                81.03,

            "auc":
                97.95,
        },

        {
            "model_name":
                "ResNet50",

            "accuracy":
                84.41,

            "precision":
                84.76,

            "recall":
                84.41,

            "f1":
                83.44,

            "auc":
                98.35,
        },

        {
            "model_name":
                "EfficientNetB0",

            "accuracy":
                60.14,

            "precision":
                62.82,

            "recall":
                60.14,

            "f1":
                52.17,

            "auc":
                89.14,
        },

    ]

    return templates.TemplateResponse(
        request=request,
        name="model_comparison.html",
        context={

            "models":
                comparison,

            "best_model":
                "ResNet50",

        }
    )


# ============================================================
# PREPROCESSING
# ============================================================

@app.get(
    "/preprocessing",
    response_class=HTMLResponse,
    name="preprocessing"
)
async def preprocessing_page(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="preprocessing.html",
        context={}
    )


# ============================================================
# ABOUT
# ============================================================

@app.get(
    "/about",
    response_class=HTMLResponse,
    name="about"
)
async def about_page(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={}
    )


# ============================================================
# CONTACT
# ============================================================

@app.get(
    "/contact",
    response_class=HTMLResponse,
    name="contact"
)
async def contact_page(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="contact.html",
        context={}
    )


# ============================================================
# /prediction REDIRECT
# ============================================================

@app.get(
    "/prediction",
    name="prediction_list"
)
async def prediction_redirect(
    request: Request
):

    if not get_current_user(
        request
    ):

        return RedirectResponse(
            url="/login",
            status_code=303
        )

    return RedirectResponse(
        url="/upload",
        status_code=303
    )


# ============================================================
# FAVICON
# ============================================================

@app.get(
    "/favicon.ico",
    include_in_schema=False
)
async def favicon():

    return Response(
        status_code=204
    )


# ============================================================
# 404 HANDLER
# ============================================================
#
# IMPORTANT:
# Do NOT render index.html here.
# If index itself has an error, that causes
# recursive template errors.
# ============================================================

@app.exception_handler(404)
async def not_found(
    request: Request,
    exc
):

    return HTMLResponse(
        content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Page Not Found</title>

            <style>
                body {
                    font-family: Arial;
                    text-align: center;
                    padding: 80px;
                }

                a {
                    color: #7c3aed;
                    text-decoration: none;
                }
            </style>

        </head>

        <body>

            <h1>404</h1>

            <h2>Page Not Found</h2>

            <p>
                The requested page does not exist.
            </p>

            <a href="/">
                Go to Home
            </a>

        </body>
        </html>
        """,
        status_code=404
    )


# ============================================================
# GENERAL EXCEPTION HANDLER
# ============================================================

@app.exception_handler(Exception)
async def general_exception(
    request: Request,
    exc: Exception
):

    print()
    print("========================================")
    print("[UNHANDLED ERROR]")
    print("========================================")

    print(exc)

    traceback.print_exc()

    return HTMLResponse(
        content="""
        <!DOCTYPE html>
        <html>

        <head>

            <title>
                Internal Server Error
            </title>

            <style>

                body {
                    font-family: Arial;
                    text-align: center;
                    padding: 80px;
                }

                a {
                    color: #7c3aed;
                    text-decoration: none;
                }

            </style>

        </head>

        <body>

            <h1>500</h1>

            <h2>Internal Server Error</h2>

            <p>
                Please check the terminal
                for the detailed error.
            </p>

            <a href="/">
                Go to Home
            </a>

        </body>

        </html>
        """,
        status_code=500
    )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    print()
    print("==================================================")
    print("       BREAST CANCER XAI - FASTAPI")
    print("==================================================")

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    try:

        init_database()

    except Exception as e:

        print()
        print(
            "[ERROR] MySQL startup failed."
        )

        print(e)

        print()
        print("Check:")

        print(
            "1. MySQL server is running"
        )

        print(
            "2. Database breast_cancer_xai exists"
        )

        print(
            "3. MySQL username/password are correct"
        )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    try:

        load_models()

    except Exception as e:

        print()
        print(
            "[ERROR] Model startup failed."
        )

        print(e)

        traceback.print_exc()

    print()
    print("==================================================")

    print(
        "FastAPI application startup complete."
    )

    print("==================================================")

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )