from fastapi import FastAPI
# 우리가 만든 라우터(measure.py)를 불러옵니다.
from app.routers import measure

app = FastAPI(title="H-ALIGN Server")

# [핵심] 여기서 measure 라우터를 서버에 등록합니다.
# 주소 앞에 자동으로 '/api/v1'이 붙습니다.
app.include_router(measure.router, prefix="/api/v1", tags=["measurement"])

@app.get("/")
def read_root():
    return {"message": "Hello Grow-Up Project!"}