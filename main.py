"""
Cloud-Based Secure Hospital Service Management System
Backend - FastAPI with Strict Medical Privacy
"""

import os
import uuid
import logging
import bleach
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum as PyEnum

from fastapi import FastAPI, APIRouter, Depends, HTTPException, status, Request, Header
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Boolean, Text, Enum, func, JSON, Index
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship
from pydantic import BaseModel, EmailStr, Field, validator
from passlib.context import CryptContext
from jose import jwt, JWTError
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

# ==================== CONFIGURATION ====================
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
IOT_API_KEY = os.getenv("IOT_API_KEY")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RESEND_FROM = os.getenv("RESEND_FROM", "noreply@cura-health.com")

required_vars = {
    "DATABASE_URL": DATABASE_URL,
    "SECRET_KEY": SECRET_KEY,
    "IOT_API_KEY": IOT_API_KEY,
    "ADMIN_EMAIL": ADMIN_EMAIL,
    "ADMIN_PASSWORD": ADMIN_PASSWORD,
}
missing = [name for name, value in required_vars.items() if not value]
if missing:
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")

limiter = Limiter(key_func=get_remote_address)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== DATABASE ====================
if ENVIRONMENT == "production":
    engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=10, pool_pre_ping=True)
else:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==================== ENUMS ====================
class UserRole(str, PyEnum):
    USER = "USER"
    ADMIN = "ADMIN"

class ServiceType(str, PyEnum):
    DOCTOR_CONSULTATION = "DOCTOR_CONSULTATION"
    THERAPY_SESSION = "THERAPY_SESSION"
    FOLLOW_UP_APPOINTMENT = "FOLLOW_UP_APPOINTMENT"

class RequestStatus(str, PyEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    COMPLETED = "COMPLETED"

class RoomStatus(str, PyEnum):
    AVAILABLE = "AVAILABLE"
    BUSY = "BUSY"

# ==================== MODELS ====================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    role = Column(String, default=UserRole.USER, index=True)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    service_requests = relationship("ServiceRequest", back_populates="user", cascade="all, delete-orphan")
    medical_reports = relationship("MedicalReport", back_populates="user", cascade="all, delete-orphan")

class ServiceRequest(Base):
    __tablename__ = "service_requests"
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    service_type = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    status = Column(String, default=RequestStatus.PENDING, index=True)
    admin_reason = Column(Text, nullable=True)
    scheduled_time = Column(DateTime, nullable=True)
    assigned_room = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="service_requests")
    medical_report = relationship("MedicalReport", back_populates="service_request", uselist=False)

class MedicalReport(Base):
    __tablename__ = "medical_reports"
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    request_id = Column(Integer, ForeignKey("service_requests.id"), nullable=False, index=True)
    diagnosis = Column(Text, nullable=False)
    prescription = Column(Text, nullable=True)
    doctor_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    user = relationship("User", back_populates="medical_reports")
    service_request = relationship("ServiceRequest", back_populates="medical_report")

class RoomIoT(Base):
    __tablename__ = "room_iot"
    id = Column(Integer, primary_key=True, index=True)
    room_id = Column(String, unique=True, nullable=False, index=True)
    room_type = Column(String, nullable=False, index=True)
    status = Column(String, default=RoomStatus.AVAILABLE, index=True)
    last_updated = Column(DateTime, default=datetime.utcnow)

# ==================== EMAIL SERVICE ====================
def send_email(to: str, subject: str, html: str):
    if not RESEND_API_KEY:
        logger.info(f"[EMAIL DISABLED] To: {to} | Subject: {subject}")
        return None
    try:
        import resend
        resend.api_key = RESEND_API_KEY
        response = resend.Emails.send({
            "from": RESEND_FROM,
            "to": [to],
            "subject": subject,
            "html": html
        })
        logger.info(f"Email sent to {to}")
        return response
    except Exception as e:
        logger.error(f"Email failed: {e}")
        return None

# ==================== SCHEMAS ====================
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    phone: Optional[str] = None
    @validator('full_name', 'phone', pre=True, always=True)
    def sanitize_strings(cls, v):
        return bleach.clean(v, tags=[], strip=True) if v else v

class UserResponse(BaseModel):
    id: int
    uuid: str
    email: EmailStr
    full_name: Optional[str]
    phone: Optional[str]
    role: str
    is_active: bool
    created_at: datetime
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class ServiceRequestCreate(BaseModel):
    service_type: ServiceType
    description: str = Field(..., min_length=10, max_length=5000)
    scheduled_time: Optional[datetime] = None
    @validator('description')
    def sanitize_description(cls, v):
        return bleach.clean(v, tags=[], strip=True)
    @validator('scheduled_time')
    def validate_future_time(cls, v):
        if v and v < datetime.utcnow():
            raise ValueError('Scheduled time must be in the future')
        return v

class ServiceRequestUserView(BaseModel):
    id: int
    uuid: str
    service_type: str
    description: str
    status: str
    admin_reason: Optional[str]
    scheduled_time: Optional[datetime]
    assigned_room: Optional[str]
    created_at: datetime
    has_report: bool
    class Config:
        from_attributes = True

class ServiceRequestAdminView(BaseModel):
    id: int
    uuid: str
    user_email: str
    service_type: str
    status: str
    admin_reason: Optional[str]
    scheduled_time: Optional[datetime]
    assigned_room: Optional[str]
    created_at: datetime
    class Config:
        from_attributes = True

class AdminDecision(BaseModel):
    status: RequestStatus
    reason: str = Field(..., min_length=5)
    assigned_room: Optional[str] = None

class MedicalReportCreate(BaseModel):
    request_uuid: str
    diagnosis: str = Field(..., min_length=5)
    prescription: Optional[str] = None
    doctor_notes: Optional[str] = None
    @validator('diagnosis', 'prescription', 'doctor_notes', pre=True, always=True)
    def sanitize_medical(cls, v):
        return bleach.clean(v, tags=[], strip=True) if v else v

class MedicalReportView(BaseModel):
    id: int
    uuid: str
    diagnosis: str
    prescription: Optional[str]
    doctor_notes: Optional[str]
    service_type: str
    created_at: datetime
    class Config:
        from_attributes = True

class RoomStatusUpdate(BaseModel):
    room_id: str
    status: RoomStatus

class RoomView(BaseModel):
    room_id: str
    room_type: str
    status: RoomStatus
    last_updated: datetime
    class Config:
        from_attributes = True

# ==================== AUTH ====================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

class AuthService:
    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)
    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        return pwd_context.verify(plain, hashed)
    @staticmethod
    def create_token(data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    @staticmethod
    def verify_token(token: str) -> Optional[Dict]:
        try:
            return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        except JWTError:
            return None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    payload = AuthService.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid credentials", headers={"WWW-Authenticate": "Bearer"})
    user = db.query(User).filter(User.id == payload.get("sub"), User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return user

def require_admin(current_user: User = Depends(get_current_user)):
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

def verify_iot_key(api_key: str = Header(None, alias="X-API-Key")):
    if api_key != IOT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key

# ==================== ROUTERS ====================
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])
user_router = APIRouter(prefix="/user", tags=["User"])
admin_router = APIRouter(prefix="/admin", tags=["Admin"])
iot_router = APIRouter(prefix="/iot", tags=["IoT"])

# ==================== AUTH ENDPOINTS ====================
@auth_router.post("/register", response_model=Token)
@limiter.limit("5/hour")
async def register(request: Request, user_data: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        email=user_data.email,
        password_hash=AuthService.hash_password(user_data.password),
        full_name=user_data.full_name,
        phone=user_data.phone
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = AuthService.create_token({"sub": user.id, "role": user.role})
    return {"access_token": token, "user": user}

@auth_router.post("/login", response_model=Token)
@limiter.limit("5/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username, User.is_active == True).first()
    if not user or not AuthService.verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = AuthService.create_token({"sub": user.id, "role": user.role})
    return {"access_token": token, "user": user}

# ==================== USER ENDPOINTS ====================
@user_router.post("/requests", response_model=ServiceRequestUserView)
async def create_request(request: Request, request_data: ServiceRequestCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    req = ServiceRequest(
        user_id=current_user.id,
        service_type=request_data.service_type,
        description=request_data.description,
        scheduled_time=request_data.scheduled_time
    )
    db.add(req)
    db.commit()
    db.refresh(req)
    return {**req.__dict__, "has_report": False}

@user_router.get("/requests", response_model=List[ServiceRequestUserView])
async def get_my_requests(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    requests = db.query(ServiceRequest).filter(ServiceRequest.user_id == current_user.id).order_by(ServiceRequest.created_at.desc()).all()
    result = []
    for req in requests:
        has_report = db.query(MedicalReport).filter(MedicalReport.request_id == req.id).first() is not None
        result.append({**req.__dict__, "has_report": has_report})
    return result

@user_router.get("/reports/{request_uuid}", response_model=MedicalReportView)
async def get_my_medical_report(request_uuid: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    request = db.query(ServiceRequest).filter(ServiceRequest.uuid == request_uuid, ServiceRequest.user_id == current_user.id).first()
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")
    report = db.query(MedicalReport).filter(MedicalReport.request_id == request.id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Medical report not yet available")
    return {
        "id": report.id,
        "uuid": report.uuid,
        "diagnosis": report.diagnosis,
        "prescription": report.prescription,
        "doctor_notes": report.doctor_notes,
        "service_type": request.service_type,
        "created_at": report.created_at
    }

# ==================== ADMIN ENDPOINTS ====================
@admin_router.get("/requests", response_model=List[ServiceRequestAdminView])
async def admin_get_requests(status: Optional[RequestStatus] = None, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    query = db.query(ServiceRequest).join(User)
    if status:
        query = query.filter(ServiceRequest.status == status)
    requests = query.order_by(ServiceRequest.created_at.desc()).all()
    result = []
    for req in requests:
        user = db.query(User).filter(User.id == req.user_id).first()
        result.append({
            "id": req.id,
            "uuid": req.uuid,
            "user_email": user.email,
            "service_type": req.service_type,
            "status": req.status,
            "admin_reason": req.admin_reason,
            "scheduled_time": req.scheduled_time,
            "assigned_room": req.assigned_room,
            "created_at": req.created_at
        })
    return result

@admin_router.put("/requests/{request_uuid}/decide")
async def admin_decide_request(request: Request, request_uuid: str, decision: AdminDecision, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    service_request = db.query(ServiceRequest).filter(ServiceRequest.uuid == request_uuid).first()
    if not service_request:
        raise HTTPException(status_code=404, detail="Request not found")
    service_request.status = decision.status
    service_request.admin_reason = decision.reason
    service_request.assigned_room = decision.assigned_room
    service_request.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(service_request)
    
    # Send email notification
    user = db.query(User).filter(User.id == service_request.user_id).first()
    if user:
        subject = f"Your service request has been {decision.status.value}"
        html = f"<h2>Request Update</h2><p>Dear {user.full_name or 'Patient'},</p><p>Your request <strong>{service_request.uuid}</strong> has been <strong>{decision.status.value}</strong>.</p><p><strong>Reason:</strong> {decision.reason}</p>{f'<p><strong>Assigned Room:</strong> {decision.assigned_room}</p>' if decision.assigned_room else ''}<p>Thank you for using Cura.</p>"
        send_email(to=user.email, subject=subject, html=html)
    
    return {"message": f"Request {decision.status}", "request_uuid": request_uuid, "status": decision.status, "reason": decision.reason}

@admin_router.post("/reports/create")
async def create_medical_report(request: Request, report_data: MedicalReportCreate, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    service_request = db.query(ServiceRequest).filter(ServiceRequest.uuid == report_data.request_uuid).first()
    if not service_request:
        raise HTTPException(status_code=404, detail="Request not found")
    if service_request.status != RequestStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Request must be completed first")
    existing = db.query(MedicalReport).filter(MedicalReport.request_id == service_request.id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Report already exists")
    report = MedicalReport(
        user_id=service_request.user_id,
        request_id=service_request.id,
        diagnosis=report_data.diagnosis,
        prescription=report_data.prescription,
        doctor_notes=report_data.doctor_notes
    )
    db.add(report)
    db.commit()
    return {"message": "Medical report created successfully", "report_uuid": report.uuid}

@admin_router.get("/stats")
async def admin_stats(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    total = db.query(func.count(ServiceRequest.id)).scalar()
    pending = db.query(func.count(ServiceRequest.id)).filter(ServiceRequest.status == RequestStatus.PENDING).scalar()
    approved = db.query(func.count(ServiceRequest.id)).filter(ServiceRequest.status == RequestStatus.APPROVED).scalar()
    completed = db.query(func.count(ServiceRequest.id)).filter(ServiceRequest.status == RequestStatus.COMPLETED).scalar()
    type_counts = db.query(ServiceRequest.service_type, func.count(ServiceRequest.id)).group_by(ServiceRequest.service_type).all()
    return {
        "total_requests": total,
        "pending": pending,
        "approved": approved,
        "completed": completed,
        "service_distribution": {t: c for t, c in type_counts},
        "privacy_note": "Medical reports excluded from admin view"
    }

# ==================== IOT ENDPOINTS ====================
@iot_router.post("/rooms/update")
async def update_room_status(update: RoomStatusUpdate, api_key: str = Depends(verify_iot_key), db: Session = Depends(get_db)):
    room = db.query(RoomIoT).filter(RoomIoT.room_id == update.room_id).first()
    if not room:
        room = RoomIoT(room_id=update.room_id, room_type="therapy" if "therapy" in update.room_id else "consultation", status=update.status)
        db.add(room)
    else:
        room.status = update.status
        room.last_updated = datetime.utcnow()
    db.commit()
    return {"message": f"Room {update.room_id} status updated to {update.status}"}

@iot_router.get("/rooms", response_model=List[RoomView])
async def get_room_status(room_type: Optional[str] = None, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    query = db.query(RoomIoT)
    if room_type:
        query = query.filter(RoomIoT.room_type == room_type)
    return query.all()

# ==================== APP ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        for room_id, room_type in [("therapy_room_1", "therapy"), ("therapy_room_2", "therapy"), ("doctor_room_1", "consultation"), ("doctor_room_2", "consultation")]:
            if not db.query(RoomIoT).filter(RoomIoT.room_id == room_id).first():
                db.add(RoomIoT(room_id=room_id, room_type=room_type))
        db.commit()
        if not db.query(User).filter(User.role == UserRole.ADMIN).first():
            db.add(User(email=ADMIN_EMAIL, password_hash=AuthService.hash_password(ADMIN_PASSWORD), full_name="Administrator", role=UserRole.ADMIN))
            db.commit()
            logger.info(f"Admin created: {ADMIN_EMAIL}")
    finally:
        db.close()
    yield

app = FastAPI(title="Cura - Secure Hospital Management", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(admin_router)
app.include_router(iot_router)

@app.get("/")
async def root():
    return {
        "service": "Cura",
        "privacy": "User medical data is private and visible only to the user",
        "admin_restriction": "Admin access is limited to maintain data confidentiality",
        "security": ["JWT", "bcrypt", "RBAC", "Rate Limiting", "Input Sanitization"],
        "iot": "Resource availability monitoring"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "environment": ENVIRONMENT}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
