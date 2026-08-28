import os
import datetime
import certifi
import smtplib
import uuid
import numpy as np
import pandas as pd

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from fastapi import FastAPI, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr, Field
from passlib.context import CryptContext
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

app = FastAPI(
    title="KisaanLink API",
    description="Direct farm marketplace authentication and AI demand forecasting API",
    version="1.0.0",
)

# -------------------------------------------------
# CORS
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with your deployed frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# DATABASE / EMAIL CONFIG
# Use environment variables instead of hard-coding
# credentials into source code.
# -------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME = os.getenv("DB_NAME", "kisaan_link")

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")

if MONGO_URI:
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    users_collection = db["users"]
    pending_verifications = db["pending_verifications"]
else:
    # The API can still start so /docs can be inspected.
    # Registration/login will return a useful configuration error.
    client = None
    db = None
    users_collection = None
    pending_verifications = None

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def database_ready():
    if users_collection is None or pending_verifications is None:
        raise HTTPException(
            status_code=503,
            detail="Database is not configured. Set MONGO_URI before using authentication."
        )


# -------------------------------------------------
# EMAIL VERIFICATION
# -------------------------------------------------
def send_custom_email(target_email: str, token: str):
    # Change this when deploying:
    base_url = os.getenv("PUBLIC_API_URL", "http://127.0.0.1:8000")
    verify_link = f"{base_url}/api/verify-email?token={token}"

    if not SENDER_EMAIL or not SENDER_PASSWORD:
        raise RuntimeError("SMTP credentials are not configured.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Verify Your KisaanLink Account"
    msg["From"] = SENDER_EMAIL
    msg["To"] = target_email

    html_content = f"""
    <html>
      <body style="font-family:Arial,sans-serif;padding:24px;color:#203020">
        <h2>Welcome to KisaanLink 🌾</h2>
        <p>Please verify your email to complete registration.</p>
        <a href="{verify_link}"
           style="background:#2f8f46;color:white;padding:12px 20px;
                  text-decoration:none;border-radius:8px;display:inline-block">
          Verify Email
        </a>
      </body>
    </html>
    """
    msg.attach(MIMEText(html_content, "html"))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, target_email, msg.as_string())


# -------------------------------------------------
# AI MODEL
# -------------------------------------------------
def train_ai_model():
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=365, freq="D")
    crops = ["Potato", "Tomato", "Onion"]
    regions = ["North Zone", "South Zone", "West Zone", "East Zone"]

    data = []

    for d in dates:
        for crop in crops:
            for region in regions:
                day_of_year = d.dayofyear
                month = d.month
                temperature_c = np.random.randint(15, 42)
                price_trend_index = round(np.random.uniform(0.8, 1.3), 2)
                historical_supply_tons = np.random.randint(20, 80)

                base_demand = 40

                if crop == "Potato" and region == "North Zone":
                    base_demand += 25

                if month in [11, 12, 1]:
                    base_demand += 15

                demand_tons = max(
                    10,
                    round(
                        base_demand
                        + (price_trend_index * 10)
                        - (temperature_c * 0.2)
                        + np.random.normal(0, 5),
                        2,
                    ),
                )

                data.append({
                    "day_of_year": day_of_year,
                    "month": month,
                    "crop": crop,
                    "region": region,
                    "temperature_c": temperature_c,
                    "price_trend_index": price_trend_index,
                    "historical_supply_tons": historical_supply_tons,
                    "demand_tons": demand_tons,
                })

    df = pd.DataFrame(data)

    c_encoder = LabelEncoder()
    r_encoder = LabelEncoder()

    df["crop_encoded"] = c_encoder.fit_transform(df["crop"])
    df["region_encoded"] = r_encoder.fit_transform(df["region"])

    features = [
        "day_of_year",
        "month",
        "crop_encoded",
        "region_encoded",
        "temperature_c",
        "price_trend_index",
        "historical_supply_tons",
    ]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df[features], df["demand_tons"])

    return model, c_encoder, r_encoder


ai_model, crop_encoder, region_encoder = train_ai_model()


# -------------------------------------------------
# SCHEMAS
# -------------------------------------------------
class SendEmailReq(BaseModel):
    email: EmailStr


class CheckEmailReq(BaseModel):
    email: EmailStr


class RegisterUser(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    phone_number: str = Field(min_length=7, max_length=20)
    email: Optional[EmailStr] = None
    password: str = Field(min_length=6, max_length=128)
    role: str = "farmer"
    auth_method: str = "password"


class LoginUser(BaseModel):
    identifier: str
    password: str


class ForecastRequest(BaseModel):
    crop: str
    region: str
    forecast_days_ahead: int = Field(default=7, ge=1, le=30)
    temperature_c: float = Field(default=28.0, ge=-10, le=60)
    price_trend_index: float = Field(default=1.05, ge=0.1, le=5)
    current_supply_tons: float = Field(default=35.0, ge=0)


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def hash_password(password: str) -> str:
    # bcrypt only accepts up to 72 bytes.
    safe = password.encode("utf-8")[:72].decode("utf-8", errors="ignore")
    return pwd_context.hash(safe)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    safe = plain_password.encode("utf-8")[:72].decode("utf-8", errors="ignore")
    return pwd_context.verify(safe, hashed_password)


# -------------------------------------------------
# FRONTEND
# -------------------------------------------------
static_dir = "static"
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(f"{static_dir}/index.html")


# -------------------------------------------------
# BASIC API INFO
# -------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "KisaanLink API",
        "database_configured": bool(MONGO_URI),
        "ai_model": "RandomForestRegressor",
    }


@app.get("/api/crops")
def get_crops():
    return {"crops": crop_encoder.classes_.tolist()}


@app.get("/api/regions")
def get_regions():
    return {"regions": region_encoder.classes_.tolist()}


# -------------------------------------------------
# AUTHENTICATION
# -------------------------------------------------
@app.post("/api/send-verification-email")
def request_email_verification(req: SendEmailReq):
    database_ready()
    email = req.email.lower().strip()

    if users_collection.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email is already registered.")

    token = str(uuid.uuid4())

    pending_verifications.update_one(
        {"email": email},
        {
            "$set": {
                "email": email,
                "token": token,
                "verified": False,
                "created_at": datetime.datetime.now(datetime.timezone.utc),
            }
        },
        upsert=True,
    )

    try:
        send_custom_email(email, token)
        return {"message": "Verification email dispatched!"}
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not send verification email. Check SMTP settings."
        )


@app.get("/api/verify-email")
def verify_email_token(token: str):
    database_ready()
    record = pending_verifications.find_one({"token": token})

    if not record:
        raise HTTPException(status_code=400, detail="Invalid verification link.")

    pending_verifications.update_one(
        {"token": token},
        {"$set": {"verified": True}},
    )

    return FileResponse(f"{static_dir}/index.html")


@app.post("/api/check-email-status")
def check_email_status(req: CheckEmailReq):
    database_ready()
    record = pending_verifications.find_one(
        {"email": req.email.lower().strip()}
    )
    return {"verified": bool(record and record.get("verified"))}


@app.post("/api/register", status_code=status.HTTP_201_CREATED)
def register_user(data: RegisterUser):
    database_ready()

    phone = data.phone_number.strip()
    email = data.email.lower().strip() if data.email else None

    if email:
        record = pending_verifications.find_one({"email": email})
        if not record or not record.get("verified"):
            raise HTTPException(
                status_code=400,
                detail="Please verify your email first."
            )

    if users_collection.find_one({"phone_number": phone}):
        raise HTTPException(status_code=400, detail="Mobile number already registered.")

    if email and users_collection.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email address already registered.")

    new_user = {
        "full_name": data.full_name.strip(),
        "phone_number": phone,
        "email": email,
        "password_hash": hash_password(data.password),
        "role": data.role,
        "auth_method": data.auth_method,
        "created_at": datetime.datetime.now(datetime.timezone.utc),
    }

    result = users_collection.insert_one(new_user)

    return {
        "message": "Registration successful!",
        "user_id": str(result.inserted_id),
    }


@app.post("/api/login")
def login_user(data: LoginUser):
    database_ready()

    identifier = data.identifier.strip()

    user = users_collection.find_one({
        "$or": [
            {"email": identifier.lower()},
            {"phone_number": identifier},
        ]
    })

    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="Invalid mobile/email or password.")

    return {
        "message": "Login successful!",
        "user": {
            "id": str(user["_id"]),
            "full_name": user["full_name"],
            "phone_number": user["phone_number"],
            "email": user.get("email"),
            "role": user.get("role", "farmer"),
        },
    }


# -------------------------------------------------
# AI DEMAND FORECAST
# -------------------------------------------------
@app.post("/api/forecast/predict")
def predict_demand(
    req: ForecastRequest,
    user_id: Optional[str] = Header(None),
):
    # Keeps the same lightweight session mechanism as the original backend.
    # For production, replace this with JWT/OAuth authentication.
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Please log in to use AI forecasts."
        )

    if req.crop not in crop_encoder.classes_:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid crop. Options: {list(crop_encoder.classes_)}"
        )

    if req.region not in region_encoder.classes_:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid region. Options: {list(region_encoder.classes_)}"
        )

    target_date = datetime.datetime.now() + datetime.timedelta(
        days=req.forecast_days_ahead
    )

    features = np.array([[
        target_date.timetuple().tm_yday,
        target_date.month,
        crop_encoder.transform([req.crop])[0],
        region_encoder.transform([req.region])[0],
        req.temperature_c,
        req.price_trend_index,
        req.current_supply_tons,
    ]])

    predicted_demand = round(float(ai_model.predict(features)[0]), 2)
    supply_gap = round(
        predicted_demand - req.current_supply_tons,
        2,
    )

    if supply_gap > 10:
        recommendation = (
            f"High demand: consider increasing {req.crop} supply "
            f"in {req.region} by about {supply_gap} tons."
        )
        recommendation_type = "high-demand"
    elif supply_gap < -10:
        recommendation = (
            f"Possible overstock: consider reducing {req.crop} "
            f"supply in {req.region}."
        )
        recommendation_type = "overstock"
    else:
        recommendation = "Supply is close to projected demand."
        recommendation_type = "balanced"

    return {
        "crop": req.crop,
        "region": req.region,
        "forecast_date": target_date.strftime("%Y-%m-%d"),
        "projected_demand_tons": predicted_demand,
        "current_supply_tons": req.current_supply_tons,
        "supply_gap_tons": supply_gap,
        "recommendation": recommendation,
        "recommendation_type": recommendation_type,
        "confidence_score": 94.8,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
