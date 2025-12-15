from enum import StrEnum, auto
from pydantic import BaseModel, EmailStr, Field, validator
from datetime import timedelta, datetime, timezone
from typing import Optional, List

class UserBase(BaseModel):
    email: EmailStr
    name: str

    @validator('name')
    def validate_name_length(cls, v):
        if len(v) < 1 or len(v) > 15:
            raise ValueError('Имя должно быть от 1 до 15 символов')
        return v

class UserSignUp(UserBase):
    password: str

class UserRead(UserBase):
    id: int

class User(UserRead):
    hashed_password: str

class RoomCreate(BaseModel):
    name: str
    
    @validator('name')
    def validate_name_length(cls, v):
        if len(v) < 1 or len(v) > 15:
            raise ValueError('Название комнаты должно быть от 1 до 15 символов')
        return v

class RoomResponse(BaseModel):
    id: int
    owner_id: int
    name: str
    status: str
    created_at: datetime
    is_open: bool
    access_code: Optional[str] = None
    participants: List[UserRead] = Field(default_factory=list)

class ParticipantInfo(BaseModel):
    id: int
    name: str
    email: str
    role: str
    is_current_user: bool
    is_owner: bool

class RoomParticipantsResponse(BaseModel):
    room_id: int
    room_name: str
    owner_id: int
    participants: List[ParticipantInfo]

class UserStats(BaseModel):
    movies_suggested: int
    movies_selected: int
    total_ratings_received: int

class UserInfoResponse(BaseModel):
    user: UserRead
    stats: UserStats

class RoomStatus(StrEnum):
    COLLECTING = "collecting"
    ROULETTE = "roulette"
    WATCHING = "watching"
    REVIEWING = "reviewing"
    FINISHED = "finished"

class UserUpdate(BaseModel):
    name: Optional[str] = None
    profile_picture: Optional[str] = None

    @validator('name')
    def validate_name_length(cls, v):
        if v is not None and (len(v) < 1 or len(v) > 15):
            raise ValueError('Имя должно быть от 1 до 15 символов')
        return v

    @validator('profile_picture')
    def validate_profile_picture(cls, v):
        if v is not None and len(v) > 255:
            raise ValueError('URL картинки слишком длинный')
        return v

class UserProfileResponse(UserRead):
    profile_picture: Optional[str] = None
    registered_at: datetime
    overall_rating: float
    is_active: bool

class UserProfileUpdateResponse(BaseModel):
    message: str
    user: UserProfileResponse

class MovieSuggestion(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)

class MovieInfo(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    duration: Optional[int] = None
    release_year: Optional[int] = None
    poster_url: Optional[str] = None
    rating_kp: Optional[float] = None
    votes_kp: Optional[int] = None
    genres: List[str] = []

class RouletteResult(BaseModel):
    room_id: int
    selected_movie: MovieInfo
    selected_user_id: int
    selected_user_name: str
    candidates_count: int
    spin_duration: int

class ReviewCreate(BaseModel):
    rating: int = Field(..., ge=1, le=10, description="Оценка от 1 до 10")
    comment: Optional[str] = Field(None, max_length=500)

class ReviewResponse(BaseModel):
    id: int
    movie_id: int
    room_id: int
    from_user_id: int
    to_user_id: int
    rating: int
    comment: Optional[str]
    reviewed_at: datetime
    
    movie_title: str
    from_user_name: str
    to_user_name: str

class RoomReviewsResponse(BaseModel):
    movie: dict  
    suggested_by: dict  
    reviews: dict  
    has_user_reviewed: bool