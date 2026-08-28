# KisaanLink API + Farmer-Friendly Frontend

## Run

1. Install dependencies:
   pip install -r requirements.txt

2. Configure MongoDB:
   Windows PowerShell:
   $env:MONGO_URI="your-mongodb-connection-string"

   Optional email variables:
   $env:SENDER_EMAIL="your-email@gmail.com"
   $env:SENDER_PASSWORD="your-gmail-app-password"
   $env:PUBLIC_API_URL="http://127.0.0.1:8000"

3. Start:
   python main.py

4. Open:
   http://127.0.0.1:8000

5. API documentation:
   http://127.0.0.1:8000/docs

## What is connected

- Registration: POST /api/register
- Login: POST /api/login
- Email verification: /api/send-verification-email, /api/verify-email, /api/check-email-status
- AI demand forecasting: POST /api/forecast/predict
- Crop/region metadata: GET /api/crops, GET /api/regions
- Health: GET /api/health

## Important

The original backend already used FastAPI and therefore was already an API. This version keeps its core endpoints, moves secrets to environment variables, adds validation/health/metadata endpoints, and connects the farmer-friendly frontend to the real authentication and AI forecast endpoints.

The original backend does NOT contain marketplace listing, orders, storage, transport, FPO, or government-scheme database endpoints. The frontend therefore does not pretend those screens are API-backed yet; the crop-listing form is explicitly marked as a prototype.

For production, replace the simple `user-id` forecast header with JWT/OAuth authentication and restrict CORS to the deployed frontend domain.
