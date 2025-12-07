from fastapi import FastAPI
from backend.app.api.endpoints import router as api_router

app = FastAPI()

# include your API router
app.include_router(api_router)

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend is running"}
