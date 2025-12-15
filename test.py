from enum import StrEnum, auto
from datetime import timedelta, datetime, timezone
from typing import Optional, List
import bcrypt
import jwt
from fastapi import FastAPI, HTTPException, Depends, status, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, validator
import psycopg2
import uvicorn
import hashlib
import secrets
import time
import os
from pathlib import Path
from fastapi import UploadFile, File
import shutil
from fastapi.staticfiles import StaticFiles
from kinopoisk_unofficial.kinopoisk_api_client import KinopoiskApiClient
from kinopoisk_unofficial.request.films.search_by_keyword_request import SearchByKeywordRequest
from kinopoisk_unofficial.request.films.film_request import FilmRequest
from kinopoisk_unofficial.response.films.film_response import FilmResponse
from kinopoisk_unofficial.response.films.search_by_keyword_response import SearchByKeywordResponse
import asyncio
from typing import Dict, List


SECRET_KEY = ""
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

KINOPOISK_TOKEN = ""
VOTE_TIME_SECONDS = 120 
ROULETTE_SPIN_TIME = 20 

kinopoisk_client = KinopoiskApiClient(KINOPOISK_TOKEN)

db_conn_dict = {
    'database': '',
    'user': '',
    'password': '',
    'host': '',
    'port': 5432,
    'options': "-c search_path=cinema"
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
app = FastAPI()

os.makedirs("uploads/profile_pictures", exist_ok=True)

app.mount("/static", StaticFiles(directory="uploads"), name="static")

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


def get_hashed_password(plain_text_password: bytes) -> bytes:
    return bcrypt.hashpw(plain_text_password, bcrypt.gensalt())

def check_password(plain_text_password: bytes, hashed_password: bytes) -> bool:
    return bcrypt.checkpw(plain_text_password, hashed_password)

def get_user(email: str) -> Optional[User]:
    with psycopg2.connect(**db_conn_dict) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT id, name, email, password_hash FROM "user" WHERE email = %s',
                (email,)
            )
            data = cur.fetchone()
            if not data:
                return None
            return User(
                id=data[0],
                name=data[1],
                email=data[2],
                hashed_password=data[3].encode() if isinstance(data[3], str) else data[3],
            )

def authenticate_user(email: str, password: str):
    user = get_user(email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Неправильный логин или пароль'
        )
    
    password_bytes = password.encode('utf-8')
    hashed_password = user.hashed_password if isinstance(user.hashed_password, bytes) else user.hashed_password.encode('utf-8')
    
    if not check_password(password_bytes, hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Неправильный логин или пароль'
        )
    return user

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise credentials_exception
        user = get_user(email)
        if user is None:
            raise credentials_exception
        return user
    except jwt.PyJWTError:
        raise credentials_exception

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def generate_access_code() -> str:
    """Генерация уникального 6-символьного кода доступа"""
    timestamp = str(time.time()).encode()
    random_bytes = secrets.token_bytes(8)
    combined = timestamp + random_bytes
    return hashlib.md5(combined).hexdigest()[:6].upper()

# Настройки для загрузки файлов
UPLOAD_DIR = Path("uploads/profile_pictures")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

def save_profile_picture(file: UploadFile, user_id: int) -> str:
    """Сохраняет загруженную картинку и возвращает URL"""
    # Проверяем тип файла
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Недопустимый тип файла. Разрешены: JPEG, PNG, GIF, WebP"
        )
    
    # Проверяем размер файла
    file.file.seek(0, 2)  # Перемещаемся в конец файла
    file_size = file.file.tell()
    file.file.seek(0)  # Возвращаемся в начало
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Файл слишком большой. Максимальный размер: 5 MB"
        )
    
    # Генерируем имя файла
    file_extension = file.filename.split('.')[-1].lower()
    filename = f"user_{user_id}_{int(time.time())}.{file_extension}"
    file_path = UPLOAD_DIR / filename
    
    # Сохраняем файл
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Возвращаем относительный URL
    return f"/static/profile_pictures/{filename}"

def delete_old_profile_picture(picture_url: str):
    """Удаляет старую картинку профиля"""
    if picture_url and picture_url.startswith("/static/profile_pictures/"):
        filename = picture_url.split("/")[-1]
        old_file = UPLOAD_DIR / filename
        if old_file.exists():
            old_file.unlink()

async def search_movie_by_keyword(keyword: str) -> Optional[MovieInfo]:
    """Поиск фильма по ключевому слову через Kinopoisk API"""
    try:
        request = SearchByKeywordRequest(keyword)
        response = kinopoisk_client.films.send_search_by_keyword_request(request)
        
        if response.items and len(response.items) > 0:
            film = response.items[0]  # Берем первый результат
            
            # Получаем детальную информацию по ID
            film_request = FilmRequest(film.kinopoisk_id)
            film_response = kinopoisk_client.films.send_film_request(film_request)
            
            # Преобразуем в нашу модель
            return MovieInfo(
                id=film_response.kinopoisk_id,
                title=film_response.name_ru or film_response.name_original or "Без названия",
                description=film_response.description or "",
                duration=film_response.film_length,
                release_year=film_response.year,
                poster_url=film_response.poster_url,
                rating_kp=film_response.rating_kinopoisk,
                votes_kp=film_response.rating_kinopoisk_vote_count,
                genres=[genre.genre for genre in film_response.genres]
            )
    except Exception as e:
        print(f"Ошибка при поиске фильма '{keyword}': {str(e)}")
    return None

def save_movie_to_db(movie: MovieInfo, suggested_by_user_id: int, room_id: int):
    """Сохраняет фильм в базу данных"""
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Проверяем, есть ли уже такой фильм в базе
                cur.execute(
                    'SELECT id FROM cinema.movie WHERE title = %s AND release_year = %s',
                    (movie.title, movie.release_year)
                )
                existing_movie = cur.fetchone()
                
                if existing_movie:
                    movie_id = existing_movie[0]
                else:
                    # Добавляем новый фильм
                    cur.execute(
                        '''
                        INSERT INTO cinema.movie 
                        (title, description, duration, release_year, poster_url)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                        ''',
                        (movie.title, movie.description, movie.duration, 
                         movie.release_year, movie.poster_url)
                    )
                    movie_id = cur.fetchone()[0]
                    
                    # Добавляем жанры
                    for genre_name in movie.genres:
                        # Проверяем/добавляем жанр
                        cur.execute(
                            'SELECT id FROM cinema.genre WHERE name = %s',
                            (genre_name,)
                        )
                        genre_result = cur.fetchone()
                        
                        if genre_result:
                            genre_id = genre_result[0]
                        else:
                            cur.execute(
                                'INSERT INTO cinema.genre (name) VALUES (%s) RETURNING id',
                                (genre_name,)
                            )
                            genre_id = cur.fetchone()[0]
                        
                        # Связываем фильм с жанром
                        cur.execute(
                            '''
                            INSERT INTO cinema.movie_genre (movie_id, genre_id)
                            VALUES (%s, %s)
                            ON CONFLICT (movie_id, genre_id) DO NOTHING
                            ''',
                            (movie_id, genre_id)
                        )
                
                # Добавляем предложение фильма
                cur.execute(
                    '''
                    INSERT INTO cinema.suggested_movie 
                    (movie_id, room_id, user_id, suggested_at, is_active, in_roulette)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    ''',
                    (movie_id, room_id, suggested_by_user_id, 
                     datetime.now(), True, True)
                )
                
                conn.commit()
                return movie_id
                
    except Exception as e:
        print(f"Ошибка при сохранении фильма в БД: {str(e)}")
        raise

async def run_roulette_selection(room_id: int, start_time: datetime):
    """Асинхронная задача для выбора фильма через рулетку"""
    await asyncio.sleep(ROULETTE_SPIN_TIME)  # Ждем 10 секунд
    
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Получаем все предложенные фильмы
                cur.execute(
                    '''
                    SELECT sm.id, sm.movie_id, sm.user_id, m.title, u.name
                    FROM cinema.suggested_movie sm
                    JOIN cinema.movie m ON sm.movie_id = m.id
                    JOIN cinema.user u ON sm.user_id = u.id
                    WHERE sm.room_id = %s AND sm.is_active = TRUE AND sm.in_roulette = TRUE
                    ''',
                    (room_id,)
                )
                suggestions = cur.fetchall()
                
                if not suggestions:
                    # Если нет предложений, сбрасываем статус
                    cur.execute(
                        '''
                        UPDATE cinema.room 
                        SET status = 'collecting', roulette_starts_at = NULL 
                        WHERE id = %s
                        ''',
                        (room_id,)
                    )
                    conn.commit()
                    return
                
                # Выбираем случайный фильм
                import random
                selected_suggestion = random.choice(suggestions)
                suggestion_id, movie_id, selected_user_id, movie_title, user_name = selected_suggestion
                
                # Получаем полную информацию о фильме
                cur.execute(
                    '''
                    SELECT title, description, duration, release_year, poster_url
                    FROM cinema.movie WHERE id = %s
                    ''',
                    (movie_id,)
                )
                movie_data = cur.fetchone()
                
                # Обновляем статус комнаты
                cur.execute(
                    '''
                    UPDATE cinema.room 
                    SET status = 'watching', selected_movie_id = %s, 
                        selected_user_id = %s, watching_starts_at = %s
                    WHERE id = %s
                    ''',
                    (movie_id, selected_user_id, datetime.now(), room_id)
                )
                
                # Помечаем выбранный фильм
                cur.execute(
                    '''
                    UPDATE cinema.suggested_movie 
                    SET in_roulette = FALSE 
                    WHERE id = %s
                    ''',
                    (suggestion_id,)
                )
                
                # Помечаем остальные фильмы как неактивные в рулетке
                cur.execute(
                    '''
                    UPDATE cinema.suggested_movie 
                    SET in_roulette = FALSE 
                    WHERE room_id = %s AND is_active = TRUE AND id != %s
                    ''',
                    (room_id, suggestion_id)
                )
                
                # Записываем результат в историю рулетки
                cur.execute(
                    '''
                    INSERT INTO cinema.roulette_history 
                    (room_id, selected_movie_id, selected_user_id, 
                     candidates_count, roulette_at, spin_duration)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ''',
                    (room_id, movie_id, selected_user_id, 
                     len(suggestions), datetime.now(), ROULETTE_SPIN_TIME)
                )
                
                # Обновляем статистику пользователя, чей фильм выбран
                cur.execute(
                    '''
                    UPDATE cinema.user_statistic 
                    SET movies_selected = movies_selected + 1 
                    WHERE user_id = %s
                    ''',
                    (selected_user_id,)
                )
                
                # Отправляем уведомления всем участникам
                cur.execute(
                    '''
                    SELECT u.id FROM cinema.user u
                    JOIN cinema.room_participant rp ON u.id = rp.user_id
                    WHERE rp.room_id = %s AND rp.is_active = TRUE
                    ''',
                    (room_id,)
                )
                participants = cur.fetchall()
                
                for participant in participants:
                    cur.execute(
                        '''
                        INSERT INTO cinema.notification 
                        (user_id, room_id, title, message, type)
                        VALUES (%s, %s, %s, %s, %s)
                        ''',
                        (participant[0], room_id, "Фильм выбран!",
                         f"Рулетка выбрала фильм '{movie_title}' от {user_name}", 
                         "movie_selected")
                    )
                
                # Создаем запись в истории сеансов
                cur.execute(
                    '''
                    INSERT INTO cinema.session_history 
                    (room_id, movie_id, suggested_by_user_id, watched_at, participants_count)
                    VALUES (%s, %s, %s, %s, 
                    (SELECT COUNT(*) FROM cinema.room_participant 
                     WHERE room_id = %s AND is_active = TRUE))
                    ''',
                    (room_id, movie_id, selected_user_id, datetime.now(), room_id)
                )
                
                conn.commit()
                
                print(f"Рулетка завершена для комнаты {room_id}. Выбран фильм: {movie_title}")
                
    except Exception as e:
        print(f"Ошибка в задаче рулетки для комнаты {room_id}: {str(e)}")

# Эндпоинты
@app.get('/ping')
def ping():
    return {'message': 'pong'}

@app.post('/auth/signup')
def signup(user_schema: UserSignUp):
    try:
        password_bytes = user_schema.password.encode('utf-8')
        hashed_password = get_hashed_password(password_bytes)
        
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'INSERT INTO "user" (name, email, password_hash) VALUES (%s, %s, %s) RETURNING id',
                    (user_schema.name, user_schema.email, hashed_password.decode('utf-8'))
                )
                user_id = cur.fetchone()[0]
                conn.commit()
        
        return {"message": "Пользователь успешно создан", "user_id": user_id}
    except psycopg2.IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь с таким email уже существует"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при создании пользователя: {str(e)}"
        )

@app.post('/auth/login')
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {'access_token': access_token, 'token_type': 'bearer'}

@app.get('/jwttest')
def jwttest(user: User = Depends(get_current_user)):
    return {'message': 'ok', 'user': user.email, 'user_id': user.id}

@app.post('/room')
def create_room(room_data: RoomCreate, user: User = Depends(get_current_user)):
    try:
        access_code = generate_access_code()
        
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Создаем комнату
                cur.execute(
                    '''
                    INSERT INTO room 
                    (name, owner_id, is_open, access_code, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    ''',
                    (room_data.name, user.id, True, access_code, 'collecting', datetime.now())
                )
                
                room_id = cur.fetchone()[0]
                
                # Добавляем владельца с ролью 'owner'
                cur.execute(
                    '''
                    INSERT INTO room_participant 
                    (room_id, user_id, role, is_active)
                    VALUES (%s, %s, %s, %s)
                    ''',
                    (room_id, user.id, 'owner', True)
                )
                
                conn.commit()
        
        return {
            "message": "Комната создана", 
            "room_id": room_id, 
            "access_code": access_code,
            "owner_role": "owner"
        }
        
    except psycopg2.errors.StringDataRightTruncation:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Название комнаты слишком длинное (максимум 15 символов)"
        )
    except psycopg2.errors.ForeignKeyViolation:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь не существует"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при создании комнаты: {str(e)}"
        )

@app.get('/rooms')
def get_rooms(user: User = Depends(get_current_user)):
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT r.id, r.name, r.owner_id, r.status, 
                           r.created_at, r.is_open, r.access_code,
                           u.id, u.name, u.email
                    FROM room r
                    JOIN room_participant rp ON r.id = rp.room_id AND rp.is_active = TRUE
                    JOIN "user" u ON u.id = rp.user_id
                    WHERE rp.user_id = %s AND r.status != 'finished'
                    ORDER BY r.created_at DESC
                """, (user.id,))
                
                rooms_data = cur.fetchall()
                
                result = {}
                for row in rooms_data:
                    room_id = row[0]
                    if room_id not in result:
                        result[room_id] = RoomResponse(
                            id=room_id,
                            name=row[1],
                            owner_id=row[2],
                            status=row[3],
                            created_at=row[4],
                            is_open=row[5],
                            access_code=row[6],
                            participants=[]
                        )
                    
                    result[room_id].participants.append(UserRead(
                        id=row[7],
                        name=row[8],
                        email=row[9]
                    ))
                
                return list(result.values())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении комнат: {str(e)}"
        )

@app.get('/rooms/{room_id}/participants')
def get_room_participants(room_id: int, user: User = Depends(get_current_user)):
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Проверяем, что пользователь является участником комнаты
                cur.execute(
                    'SELECT 1 FROM room_participant WHERE room_id = %s AND user_id = %s AND is_active = TRUE',
                    (room_id, user.id)
                )
                if not cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Вы не участник этой комнаты"
                    )
                
                # Получаем информацию о комнате
                cur.execute('SELECT id, name, owner_id FROM room WHERE id = %s', (room_id,))
                room = cur.fetchone()
                if room is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Комната не существует",
                    )
                
                room_id, room_name, owner_id = room
                
                # Получаем участников комнаты с их ролями
                cur.execute('''
                    SELECT u.id, u.name, u.email, rp.role,
                           CASE WHEN u.id = %s THEN true ELSE false END as is_current_user,
                           CASE WHEN u.id = %s THEN true ELSE false END as is_owner
                    FROM "user" u 
                    JOIN room_participant rp ON u.id = rp.user_id 
                    WHERE rp.room_id = %s AND rp.is_active = TRUE
                    ORDER BY 
                        CASE rp.role 
                            WHEN 'owner' THEN 1
                            WHEN 'moderator' THEN 2
                            WHEN 'member' THEN 3
                        END,
                        rp.joined_at
                ''', (user.id, owner_id, room_id))
                
                participants = []
                for user_data in cur.fetchall():
                    participants.append(ParticipantInfo(
                        id=user_data[0],
                        name=user_data[1],
                        email=user_data[2],
                        role=user_data[3],
                        is_current_user=user_data[4],
                        is_owner=user_data[5]
                    ))
                
                return RoomParticipantsResponse(
                    room_id=room_id,
                    room_name=room_name,
                    owner_id=owner_id,
                    participants=participants
                )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении участников: {str(e)}"
        )

@app.get("/rooms/{room_id}/access_code")
def get_room_access_code(room_id: int, user: User = Depends(get_current_user)):
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT owner_id, access_code, is_open FROM room WHERE id = %s',
                    (room_id,)
                )
                room = cur.fetchone()
                if room is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Комната не существует",
                    )
                
                owner_id, access_code, is_open = room
                if user.id != owner_id:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Отказано в доступе. Только владелец комнаты может получить код доступа.",
                    )
                
                return {
                    "room_id": room_id,
                    "access_code": access_code,
                    "is_open": is_open
                }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении кода доступа: {str(e)}"
        )

@app.post("/rooms/join/{access_code}")
def join_room_via_link(access_code: str, user: User = Depends(get_current_user)):
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Проверяем существование комнаты
                cur.execute(
                    'SELECT id, is_open, status FROM room WHERE access_code = %s',
                    (access_code.upper(),)
                )
                room = cur.fetchone()
                if room is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Комната не найдена",
                    )
                
                room_id, is_open, room_status = room
                
                # Проверяем статус комнаты
                if room_status == 'finished':
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Комната уже завершена",
                    )
                
                # Проверяем, открыта ли комната
                if not is_open:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Комната закрыта для новых участников",
                    )
                
                # Проверяем, не является ли пользователь уже участником
                cur.execute(
                    'SELECT 1 FROM room_participant WHERE room_id = %s AND user_id = %s AND is_active = TRUE',
                    (room_id, user.id)
                )
                if cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Вы уже являетесь участником этой комнаты",
                    )
                
                # Добавляем пользователя как участника с ролью 'member'
                cur.execute(
                    'INSERT INTO room_participant (room_id, user_id, role) VALUES (%s, %s, %s)',
                    (room_id, user.id, 'member')
                )
                conn.commit()
                
                return {
                    "message": "Вы успешно присоединились к комнате", 
                    "room_id": room_id,
                    "role": "member"
                }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при присоединении к комнате: {str(e)}"
        )

@app.post("/rooms/{room_id}/leave")
def leave_room(room_id: int, user: User = Depends(get_current_user)):
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Проверяем, является ли пользователь участником комнаты
                cur.execute(
                    'SELECT role FROM room_participant WHERE room_id = %s AND user_id = %s AND is_active = TRUE',
                    (room_id, user.id)
                )
                participant = cur.fetchone()
                if not participant:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Вы не являетесь участником этой комнаты",
                    )
                
                role = participant[0]
                
                # Получаем информацию о владельце комнаты
                cur.execute('SELECT owner_id FROM room WHERE id = %s', (room_id,))
                owner_id = cur.fetchone()[0]
                
                # Если пользователь владелец (owner)
                if user.id == owner_id:
                    # Проверяем, есть ли другие активные участники
                    cur.execute(
                        '''
                        SELECT user_id, role 
                        FROM room_participant 
                        WHERE room_id = %s 
                        AND user_id != %s 
                        AND is_active = TRUE 
                        ORDER BY 
                            CASE role 
                                WHEN 'owner' THEN 1
                                WHEN 'moderator' THEN 2
                                WHEN 'member' THEN 3
                            END,
                            joined_at
                        LIMIT 1
                        ''',
                        (room_id, user.id)
                    )
                    other_participant = cur.fetchone()
                    
                    if other_participant:
                        # Передаем владение другому участнику
                        new_owner_id, new_owner_role = other_participant
                        
                        # Обновляем владельца комнаты
                        cur.execute(
                            'UPDATE room SET owner_id = %s WHERE id = %s',
                            (new_owner_id, room_id)
                        )
                        
                        # Обновляем роль нового владельца
                        cur.execute(
                            'UPDATE room_participant SET role = %s WHERE room_id = %s AND user_id = %s',
                            ('owner', room_id, new_owner_id)
                        )
                        
                        message = f"Владение комнатой передано пользователю с ID {new_owner_id}"
                    else:
                        # Если нет других участников, помечаем комнату как завершенную
                        cur.execute(
                            'UPDATE room SET status = %s WHERE id = %s',
                            ('finished', room_id)
                        )
                        message = "Комната завершена, так как владелец покинул ее и нет других участников"
                else:
                    message = "Вы покинули комнату"
                
                # Обновляем запись участника
                cur.execute(
                    'UPDATE room_participant SET is_active = FALSE, left_at = %s WHERE room_id = %s AND user_id = %s',
                    (datetime.now(), room_id, user.id)
                )
                
                conn.commit()
                
                return {"message": message, "room_id": room_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при выходе из комнаты: {str(e)}"
        )

@app.get("/me")
def get_current_user_info(user: User = Depends(get_current_user)):
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Получаем статистику пользователя
                cur.execute('''
                    SELECT us.movies_suggested, us.movies_selected, us.total_ratings_received
                    FROM user_statistic us
                    WHERE us.user_id = %s
                ''', (user.id,))
                
                stats = cur.fetchone()
                if stats:
                    movies_suggested, movies_selected, total_ratings = stats
                else:
                    movies_suggested = movies_selected = total_ratings = 0
                
                return UserInfoResponse(
                    user=UserRead(id=user.id, name=user.name, email=user.email),
                    stats=UserStats(
                        movies_suggested=movies_suggested,
                        movies_selected=movies_selected,
                        total_ratings_received=total_ratings
                    )
                )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении информации о пользователе: {str(e)}"
        )

@app.get("/profile")
def get_profile(user: User = Depends(get_current_user)):
    """Получить полный профиль пользователя"""
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT id, name, email, profile_picture, 
                           registered_at, overall_rating, is_active
                    FROM "user"
                    WHERE id = %s
                ''', (user.id,))
                
                user_data = cur.fetchone()
                if not user_data:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Пользователь не найден"
                    )
                
                return UserProfileResponse(
                    id=user_data[0],
                    name=user_data[1],
                    email=user_data[2],
                    profile_picture=user_data[3] if user_data[3] else None,
                    registered_at=user_data[4],
                    overall_rating=float(user_data[5]) if user_data[5] else 0.0,
                    is_active=user_data[6]
                )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении профиля: {str(e)}"
        )

@app.patch("/profile")
def update_profile(
    user_update: UserUpdate,
    user: User = Depends(get_current_user)
):
    """Обновить профиль пользователя (имя и/или картинку)"""
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Собираем поля для обновления
                update_fields = []
                update_values = []
                
                if user_update.name is not None:
                    update_fields.append("name = %s")
                    update_values.append(user_update.name)
                
                if user_update.profile_picture is not None:
                    update_fields.append("profile_picture = %s")
                    update_values.append(user_update.profile_picture)
                
                # Если нечего обновлять
                if not update_fields:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Не указаны данные для обновления"
                    )
                
                update_values.append(user.id)
                
                # Выполняем обновление
                query = f'''
                    UPDATE "user" 
                    SET {', '.join(update_fields)}, last_updated = %s
                    WHERE id = %s
                    RETURNING id, name, email, profile_picture, 
                              registered_at, overall_rating, is_active
                '''
                update_values.append(datetime.now())
                
                cur.execute(query, tuple(update_values))
                updated_user = cur.fetchone()
                
                conn.commit()
                
                return UserProfileUpdateResponse(
                    message="Профиль успешно обновлен",
                    user=UserProfileResponse(
                        id=updated_user[0],
                        name=updated_user[1],
                        email=updated_user[2],
                        profile_picture=updated_user[3] if updated_user[3] else None,
                        registered_at=updated_user[4],
                        overall_rating=float(updated_user[5]) if updated_user[5] else 0.0,
                        is_active=updated_user[6]
                    )
                )
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь с таким именем уже существует"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обновлении профиля: {str(e)}"
        )

@app.post("/profile/picture/upload")
async def upload_profile_picture(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user)
):
    """Загрузить картинку профиля"""
    try:
        # Получаем текущую картинку для удаления
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT profile_picture FROM "user" WHERE id = %s',
                    (user.id,)
                )
                current_picture = cur.fetchone()[0]
        
        # Сохраняем новую картинку
        picture_url = save_profile_picture(file, user.id)
        
        # Обновляем профиль в БД
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''
                    UPDATE "user" 
                    SET profile_picture = %s, last_updated = %s
                    WHERE id = %s
                    RETURNING name, email, profile_picture
                    ''',
                    (picture_url, datetime.now(), user.id)
                )
                updated_user = cur.fetchone()
                
                conn.commit()
        
        # Удаляем старую картинку (если она была)
        if current_picture:
            delete_old_profile_picture(current_picture)
        
        return {
            "message": "Картинка профиля успешно обновлена",
            "profile_picture": picture_url,
            "user": {
                "id": user.id,
                "name": updated_user[0],
                "email": updated_user[1]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при загрузке картинки: {str(e)}"
        )

@app.delete("/profile/picture")
def delete_profile_picture(user: User = Depends(get_current_user)):
    """Удалить картинку профиля"""
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Получаем текущую картинку
                cur.execute(
                    'SELECT profile_picture FROM "user" WHERE id = %s',
                    (user.id,)
                )
                current_picture = cur.fetchone()[0]
                
                if not current_picture:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="У вас нет картинки профиля"
                    )
                
                # Удаляем картинку из БД
                cur.execute(
                    '''
                    UPDATE "user" 
                    SET profile_picture = '', last_updated = %s
                    WHERE id = %s
                    RETURNING name, email
                    ''',
                    (datetime.now(), user.id)
                )
                updated_user = cur.fetchone()
                
                conn.commit()
        
        # Удаляем файл
        delete_old_profile_picture(current_picture)
        
        return {
            "message": "Картинка профиля удалена",
            "user": {
                "id": user.id,
                "name": updated_user[0],
                "email": updated_user[1]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при удалении картинки: {str(e)}"
        )

@app.put("/profile/name")
def update_name(
    new_name: str = Body(..., embed=True, min_length=1, max_length=15),
    user: User = Depends(get_current_user)
):
    """Обновить только имя пользователя"""
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''
                    UPDATE "user" 
                    SET name = %s, last_updated = %s
                    WHERE id = %s
                    RETURNING id, name, email, profile_picture,
                              registered_at, overall_rating, is_active
                    ''',
                    (new_name, datetime.now(), user.id)
                )
                updated_user = cur.fetchone()
                
                conn.commit()
                
                return UserProfileUpdateResponse(
                    message="Имя успешно обновлено",
                    user=UserProfileResponse(
                        id=updated_user[0],
                        name=updated_user[1],
                        email=updated_user[2],
                        profile_picture=updated_user[3] if updated_user[3] else None,
                        registered_at=updated_user[4],
                        overall_rating=float(updated_user[5]) if updated_user[5] else 0.0,
                        is_active=updated_user[6]
                    )
                )
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь с таким именем уже существует"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обновлении имени: {str(e)}"
        )

@app.get("/profile/stats")
def get_profile_stats(user: User = Depends(get_current_user)):
    """Получить статистику профиля"""
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Основная информация
                cur.execute('''
                    SELECT u.id, u.name, u.email, u.profile_picture, 
                           u.registered_at, u.overall_rating,
                           us.movies_suggested, us.movies_selected, 
                           us.total_ratings_received, us.last_activity
                    FROM "user" u
                    LEFT JOIN user_statistic us ON u.id = us.user_id
                    WHERE u.id = %s
                ''', (user.id,))
                
                user_data = cur.fetchone()
                
                if not user_data:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Пользователь не найден"
                    )
                
                # Статистика по комнатам
                cur.execute('''
                    SELECT 
                        COUNT(DISTINCT r.id) as rooms_created,
                        COUNT(DISTINCT rp.room_id) as rooms_participated,
                        COUNT(DISTINCT CASE WHEN r.status = 'finished' THEN r.id END) as rooms_completed
                    FROM "user" u
                    LEFT JOIN room r ON u.id = r.owner_id
                    LEFT JOIN room_participant rp ON u.id = rp.user_id AND rp.is_active = TRUE
                    WHERE u.id = %s
                ''', (user.id,))
                
                room_stats = cur.fetchone()
                
                # Отзывы о пользователе
                cur.execute('''
                    SELECT 
                        COUNT(*) as total_reviews,
                        AVG(rating) as average_rating,
                        MIN(rating) as min_rating,
                        MAX(rating) as max_rating
                    FROM review
                    WHERE to_user_id = %s
                ''', (user.id,))
                
                review_stats = cur.fetchone()
                
                return {
                    "user": {
                        "id": user_data[0],
                        "name": user_data[1],
                        "email": user_data[2],
                        "profile_picture": user_data[3] if user_data[3] else None,
                        "registered_at": user_data[4],
                        "overall_rating": float(user_data[5]) if user_data[5] else 0.0,
                        "is_active": user_data[6] if len(user_data) > 6 else True
                    },
                    "statistics": {
                        "movies": {
                            "suggested": user_data[7] if user_data[7] else 0,
                            "selected": user_data[8] if user_data[8] else 0
                        },
                        "ratings_received": user_data[9] if user_data[9] else 0,
                        "last_activity": user_data[10] if len(user_data) > 10 else None,
                        "rooms": {
                            "created": room_stats[0] if room_stats and room_stats[0] else 0,
                            "participated": room_stats[1] if room_stats and room_stats[1] else 0,
                            "completed": room_stats[2] if room_stats and room_stats[2] else 0
                        },
                        "reviews": {
                            "total": review_stats[0] if review_stats and review_stats[0] else 0,
                            "average_rating": float(review_stats[1]) if review_stats and review_stats[1] else 0.0,
                            "min_rating": review_stats[2] if review_stats and review_stats[2] else 0,
                            "max_rating": review_stats[3] if review_stats and review_stats[3] else 0
                        }
                    }
                }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении статистики: {str(e)}"
        )

# Эндпоинты для администрирования (только для разработки)
@app.post("/rooms/{room_id}/promote/{user_id}")
def promote_user_to_moderator(room_id: int, user_id: int, current_user: User = Depends(get_current_user)):
    """Повысить пользователя до модератора (только владелец комнаты)"""
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Проверяем, является ли текущий пользователь владельцем комнаты
                cur.execute('SELECT owner_id FROM room WHERE id = %s', (room_id,))
                room = cur.fetchone()
                if not room:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Комната не существует",
                    )
                
                if current_user.id != room[0]:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Только владелец комнаты может повышать пользователей",
                    )
                
                # Проверяем, существует ли пользователь для повышения
                cur.execute(
                    'SELECT 1 FROM room_participant WHERE room_id = %s AND user_id = %s AND is_active = TRUE',
                    (room_id, user_id)
                )
                if not cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Пользователь не найден в этой комнате",
                    )
                
                # Повышаем пользователя до модератора
                cur.execute(
                    'UPDATE room_participant SET role = %s WHERE room_id = %s AND user_id = %s',
                    ('moderator', room_id, user_id)
                )
                conn.commit()
                
                return {"message": f"Пользователь {user_id} повышен до модератора"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при повышении пользователя: {str(e)}"
        )

@app.post("/rooms/{room_id}/demote/{user_id}")
def demote_user_to_member(room_id: int, user_id: int, current_user: User = Depends(get_current_user)):
    """Понизить модератора до участника (только владелец комнаты)"""
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Проверяем, является ли текущий пользователь владельцем комнаты
                cur.execute('SELECT owner_id FROM room WHERE id = %s', (room_id,))
                room = cur.fetchone()
                if not room:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Комната не существует",
                    )
                
                if current_user.id != room[0]:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Только владелец комнаты может понижать пользователей",
                    )
                
                # Проверяем, является ли пользователь модератором
                cur.execute(
                    'SELECT role FROM room_participant WHERE room_id = %s AND user_id = %s AND is_active = TRUE',
                    (room_id, user_id)
                )
                participant = cur.fetchone()
                if not participant:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Пользователь не найден в этой комнате",
                    )
                
                if participant[0] != 'moderator':
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Можно понижать только модераторов",
                    )
                
                # Понижаем пользователя до участника
                cur.execute(
                    'UPDATE room_participant SET role = %s WHERE room_id = %s AND user_id = %s',
                    ('member', room_id, user_id)
                )
                conn.commit()
                
                return {"message": f"Пользователь {user_id} понижен до участника"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при понижении пользователя: {str(e)}"
        )

@app.post("/rooms/{room_id}/kick/{user_id}")
def kick_user_from_room(room_id: int, user_id: int, current_user: User = Depends(get_current_user)):
    """Исключить пользователя из комнаты (владелец или модератор)"""
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Проверяем роль текущего пользователя
                cur.execute(
                    'SELECT role FROM room_participant WHERE room_id = %s AND user_id = %s AND is_active = TRUE',
                    (room_id, current_user.id)
                )
                current_user_role = cur.fetchone()
                
                if not current_user_role:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Вы не участник этой комнаты",
                    )
                
                current_user_role = current_user_role[0]
                
                # Проверяем, имеет ли пользователь право исключать
                if current_user_role not in ['owner', 'moderator']:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Только владелец или модератор может исключать пользователей",
                    )
                
                # Проверяем, существует ли пользователь для исключения
                cur.execute(
                    'SELECT role FROM room_participant WHERE room_id = %s AND user_id = %s AND is_active = TRUE',
                    (room_id, user_id)
                )
                target_user = cur.fetchone()
                
                if not target_user:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Пользователь не найден в этой комнате",
                    )
                
                target_user_role = target_user[0]
                
                # Проверяем, можно ли исключить этого пользователя
                if target_user_role == 'owner':
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Нельзя исключить владельца комнаты",
                    )
                
                if current_user_role == 'moderator' and target_user_role == 'moderator':
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Модератор не может исключить другого модератора",
                    )
                
                # Исключаем пользователя
                cur.execute(
                    'UPDATE room_participant SET is_active = FALSE, left_at = %s WHERE room_id = %s AND user_id = %s',
                    (datetime.now(), room_id, user_id)
                )
                conn.commit()
                
                return {"message": f"Пользователь {user_id} исключен из комнаты"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при исключении пользователя: {str(e)}"
        )

@app.post("/rooms/{room_id}/suggest-movie")
async def suggest_movie(
    room_id: int,
    movie_suggestion: MovieSuggestion,
    user: User = Depends(get_current_user)
):
    """Предложить фильм для рулетки"""
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Проверяем, что комната существует и находится в стадии 'collecting'
                cur.execute(
                    '''
                    SELECT status, roulette_starts_at 
                    FROM cinema.room 
                    WHERE id = %s
                    ''',
                    (room_id,)
                )
                room = cur.fetchone()
                
                if not room:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Комната не найдена"
                    )
                
                room_status, roulette_starts_at = room
                
                if room_status != 'collecting':
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Рулетка уже запущена или завершена"
                    )
                
                # Проверяем, что пользователь является участником комнаты
                cur.execute(
                    '''
                    SELECT 1 FROM cinema.room_participant 
                    WHERE room_id = %s AND user_id = %s AND is_active = TRUE
                    ''',
                    (room_id, user.id)
                )
                if not cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Вы не участник этой комнаты"
                    )
                
                # Проверяем, не предложил ли уже пользователь фильм
                cur.execute(
                    '''
                    SELECT 1 FROM cinema.suggested_movie 
                    WHERE room_id = %s AND user_id = %s AND is_active = TRUE
                    ''',
                    (room_id, user.id)
                )
                if cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Вы уже предложили фильм для этой рулетки"
                    )
                
                # Ищем фильм через Kinopoisk API
                movie_info = await search_movie_by_keyword(movie_suggestion.title)
                
                if not movie_info:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Фильм не найден. Попробуйте другое название"
                    )
                
                # Сохраняем фильм в БД
                movie_id = save_movie_to_db(movie_info, user.id, room_id)
                
                # Обновляем статистику пользователя
                cur.execute(
                    '''
                    UPDATE cinema.user_statistic 
                    SET movies_suggested = movies_suggested + 1 
                    WHERE user_id = %s
                    ''',
                    (user.id,)
                )
                
                # Отправляем уведомление о новом предложении
                cur.execute(
                    '''
                    SELECT u.id FROM cinema.user u
                    JOIN cinema.room_participant rp ON u.id = rp.user_id
                    WHERE rp.room_id = %s AND rp.is_active = TRUE AND u.id != %s
                    ''',
                    (room_id, user.id)
                )
                participants = cur.fetchall()
                
                for participant in participants:
                    cur.execute(
                        '''
                        INSERT INTO cinema.notification 
                        (user_id, room_id, title, message, type)
                        VALUES (%s, %s, %s, %s, %s)
                        ''',
                        (participant[0], room_id, "Новое предложение фильма",
                         f"{user.name} предложил(а) фильм: {movie_info.title}", 
                         "roulette_start")
                    )
                
                conn.commit()
                
                return {
                    "message": "Фильм успешно предложен",
                    "movie": movie_info,
                    "movie_id": movie_id
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при предложении фильма: {str(e)}"
        )

@app.post("/rooms/{room_id}/start-roulette")
def start_roulette(room_id: int, user: User = Depends(get_current_user)):
    """Запустить рулетку выбора фильма (только владелец комнаты)"""
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Проверяем, что пользователь является владельцем комнаты
                cur.execute(
                    '''
                    SELECT owner_id, status FROM cinema.room 
                    WHERE id = %s
                    ''',
                    (room_id,)
                )
                room = cur.fetchone()
                
                if not room:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Комната не найдена"
                    )
                
                owner_id, room_status = room
                
                if user.id != owner_id:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Только владелец комнаты может запустить рулетку"
                    )
                
                if room_status != 'collecting':
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Рулетка уже запущена или завершена"
                    )
                
                # Проверяем, что есть хотя бы 2 предложенных фильма
                cur.execute(
                    '''
                    SELECT COUNT(*) FROM cinema.suggested_movie 
                    WHERE room_id = %s AND is_active = TRUE AND in_roulette = TRUE
                    ''',
                    (room_id,)
                )
                movies_count = cur.fetchone()[0]
                
                if movies_count < 2:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Нужно как минимум 2 предложенных фильма. Сейчас: {movies_count}"
                    )
                
                # Обновляем статус комнаты и время начала рулетки
                roulette_start_time = datetime.now()
                cur.execute(
                    '''
                    UPDATE cinema.room 
                    SET status = 'roulette', roulette_starts_at = %s
                    WHERE id = %s
                    ''',
                    (roulette_start_time, room_id)
                )
                
                # Отправляем уведомления всем участникам
                cur.execute(
                    '''
                    SELECT u.id FROM cinema.user u
                    JOIN cinema.room_participant rp ON u.id = rp.user_id
                    WHERE rp.room_id = %s AND rp.is_active = TRUE
                    ''',
                    (room_id,)
                )
                participants = cur.fetchall()
                
                for participant in participants:
                    cur.execute(
                        '''
                        INSERT INTO cinema.notification 
                        (user_id, room_id, title, message, type)
                        VALUES (%s, %s, %s, %s, %s)
                        ''',
                        (participant[0], room_id, "Рулетка запущена!",
                         "Владелец комнаты запустил выбор фильма. Результат будет через 10 секунд!", 
                         "roulette_start")
                    )
                
                conn.commit()
                
                # Запускаем асинхронную задачу для выбора фильма
                asyncio.create_task(run_roulette_selection(room_id, roulette_start_time))
                
                return {
                    "message": "Рулетка запущена! Результат будет через 10 секунд",
                    "roulette_starts_at": roulette_start_time.isoformat(),
                    "movies_count": movies_count,
                    "spin_duration": ROULETTE_SPIN_TIME
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при запуске рулетки: {str(e)}"
        )

@app.get("/rooms/{room_id}/movie-suggestions")
def get_movie_suggestions(room_id: int, user: User = Depends(get_current_user)):
    """Получить все предложенные фильмы для комнаты"""
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Проверяем, что пользователь является участником комнаты
                cur.execute(
                    '''
                    SELECT 1 FROM cinema.room_participant 
                    WHERE room_id = %s AND user_id = %s AND is_active = TRUE
                    ''',
                    (room_id, user.id)
                )
                if not cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Вы не участник этой комнаты"
                    )
                
                # Получаем предложенные фильмы
                cur.execute(
                    '''
                    SELECT 
                        sm.id,
                        m.id as movie_id,
                        m.title,
                        m.description,
                        m.duration,
                        m.release_year,
                        m.poster_url,
                        u.id as user_id,
                        u.name as user_name,
                        u.profile_picture as user_avatar,
                        sm.suggested_at,
                        sm.in_roulette
                    FROM cinema.suggested_movie sm
                    JOIN cinema.movie m ON sm.movie_id = m.id
                    JOIN cinema.user u ON sm.user_id = u.id
                    WHERE sm.room_id = %s AND sm.is_active = TRUE
                    ORDER BY sm.suggested_at DESC
                    ''',
                    (room_id,)
                )
                
                suggestions = []
                for row in cur.fetchall():
                    suggestions.append({
                        "id": row[0],
                        "movie": {
                            "id": row[1],
                            "title": row[2],
                            "description": row[3],
                            "duration": row[4],
                            "release_year": row[5],
                            "poster_url": row[6]
                        },
                        "user": {
                            "id": row[7],
                            "name": row[8],
                            "profile_picture": row[9]
                        },
                        "suggested_at": row[10],
                        "in_roulette": row[11]
                    })
                
                # Получаем информацию о статусе комнаты
                cur.execute(
                    '''
                    SELECT status, roulette_starts_at, selected_movie_id
                    FROM cinema.room WHERE id = %s
                    ''',
                    (room_id,)
                )
                room_info = cur.fetchone()
                
                return {
                    "room_id": room_id,
                    "status": room_info[0],
                    "roulette_starts_at": room_info[1],
                    "selected_movie_id": room_info[2],
                    "suggestions": suggestions,
                    "count": len(suggestions),
                    "remaining_time": None  # Можно добавить расчет оставшегося времени
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении предложений: {str(e)}"
        )

@app.get("/rooms/{room_id}/roulette-result")
def get_roulette_result(room_id: int, user: User = Depends(get_current_user)):
    """Получить результат рулетки"""
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Проверяем, что пользователь является участником комнаты
                cur.execute(
                    '''
                    SELECT 1 FROM cinema.room_participant 
                    WHERE room_id = %s AND user_id = %s AND is_active = TRUE
                    ''',
                    (room_id, user.id)
                )
                if not cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Вы не участник этой комнаты"
                    )
                
                # Получаем последний результат рулетки
                cur.execute(
                    '''
                    SELECT 
                        rh.id,
                        rh.selected_movie_id,
                        rh.selected_user_id,
                        rh.candidates_count,
                        rh.roulette_at,
                        rh.spin_duration,
                        m.title as movie_title,
                        m.poster_url,
                        u.name as user_name
                    FROM cinema.roulette_history rh
                    JOIN cinema.movie m ON rh.selected_movie_id = m.id
                    JOIN cinema.user u ON rh.selected_user_id = u.id
                    WHERE rh.room_id = %s
                    ORDER BY rh.roulette_at DESC
                    LIMIT 1
                    ''',
                    (room_id,)
                )
                
                result = cur.fetchone()
                
                if not result:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Результат рулетки не найден"
                    )
                
                return {
                    "roulette_history_id": result[0],
                    "selected_movie": {
                        "id": result[1],
                        "title": result[6],
                        "poster_url": result[7]
                    },
                    "selected_user": {
                        "id": result[2],
                        "name": result[8]
                    },
                    "candidates_count": result[3],
                    "roulette_at": result[4],
                    "spin_duration": result[5]
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении результата рулетки: {str(e)}"
        )

@app.post("/rooms/{room_id}/finish-watching")
def finish_watching(room_id: int, user: User = Depends(get_current_user)):
    """Завершить просмотр и перейти к этапу оценки (только владелец)"""
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Проверяем, что пользователь является владельцем комнаты
                cur.execute(
                    '''
                    SELECT owner_id, status, selected_movie_id 
                    FROM cinema.room 
                    WHERE id = %s
                    ''',
                    (room_id,)
                )
                room = cur.fetchone()
                
                if not room:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Комната не найдена"
                    )
                
                owner_id, room_status, selected_movie_id = room
                
                if user.id != owner_id:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Только владелец комнаты может завершить просмотр"
                    )
                
                if room_status != 'watching':
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Сейчас не идет просмотр фильма"
                    )
                
                if not selected_movie_id:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Фильм не выбран для просмотра"
                    )
                
                # Обновляем статус комнаты
                cur.execute(
                    '''
                    UPDATE cinema.room 
                    SET status = 'reviewing', review_ends_at = %s
                    WHERE id = %s
                    ''',
                    (datetime.now(), room_id)
                )
                
                # Отправляем уведомления
                cur.execute(
                    '''
                    SELECT u.id FROM cinema.user u
                    JOIN cinema.room_participant rp ON u.id = rp.user_id
                    WHERE rp.room_id = %s AND rp.is_active = TRUE
                    ''',
                    (room_id,)
                )
                participants = cur.fetchall()
                
                for participant in participants:
                    cur.execute(
                        '''
                        INSERT INTO cinema.notification 
                        (user_id, room_id, title, message, type)
                        VALUES (%s, %s, %s, %s, %s)
                        ''',
                        (participant[0], room_id, "Время оценивать!",
                         "Просмотр завершен. Оцените фильм и участников!", 
                         "review_time")
                    )
                
                conn.commit()
                
                return {
                    "message": "Просмотр завершен. Начинается этап оценки",
                    "room_id": room_id,
                    "status": "reviewing",
                    "review_ends_at": datetime.now().isoformat()
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при завершении просмотра: {str(e)}"
        )

@app.post("/rooms/{room_id}/submit-review")
def submit_review(
    room_id: int,
    review: ReviewCreate,
    user: User = Depends(get_current_user)
):
    """Оценить участника, чей фильм был выбран (это и будет оценка фильму)"""
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Проверяем статус комнаты и кто предложил фильм
                cur.execute(
                    '''
                    SELECT status, selected_movie_id, selected_user_id
                    FROM cinema.room 
                    WHERE id = %s
                    ''',
                    (room_id,)
                )
                room = cur.fetchone()
                
                if not room:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Комната не найдена"
                    )
                
                room_status, selected_movie_id, selected_user_id = room
                
                if room_status != 'reviewing':
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Сейчас не время для оценок. Комната должна быть в статусе 'reviewing'"
                    )
                
                if not selected_movie_id or not selected_user_id:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="В этой комнате не было выбранного фильма или пользователя"
                    )
                
                # Проверяем, что пользователь является участником комнаты
                cur.execute(
                    '''
                    SELECT 1 FROM cinema.room_participant 
                    WHERE room_id = %s AND user_id = %s AND is_active = TRUE
                    ''',
                    (room_id, user.id)
                )
                if not cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Вы не участник этой комнаты"
                    )
                
                # Проверяем, что пользователь не оценивает сам себя
                if user.id == selected_user_id:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Нельзя оценивать самого себя"
                    )
                
                # Проверяем, не оставлял ли уже пользователь отзыв для этого фильма
                cur.execute(
                    '''
                    SELECT 1 FROM cinema.review 
                    WHERE movie_id = %s AND room_id = %s AND from_user_id = %s
                    ''',
                    (selected_movie_id, room_id, user.id)
                )
                if cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Вы уже оценили этот фильм"
                    )
                
                # Добавляем оценку в таблицу review (оценка участнику = оценка фильму)
                cur.execute(
                    '''
                    INSERT INTO cinema.review 
                    (movie_id, room_id, from_user_id, to_user_id, rating, comment, reviewed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    ''',
                    (selected_movie_id, room_id, user.id, selected_user_id, 
                     review.rating, review.comment, datetime.now())
                )
                
                review_id = cur.fetchone()[0]
                
                # 1. Обновляем общий рейтинг пользователя, чей фильм выбран
                cur.execute(
                    '''
                    UPDATE cinema.user 
                    SET overall_rating = (
                        SELECT AVG(rating)::decimal(3,2)
                        FROM cinema.review 
                        WHERE to_user_id = %s
                    )
                    WHERE id = %s
                    ''',
                    (selected_user_id, selected_user_id)
                )
                
                # 2. Обновляем статистику пользователя
                cur.execute(
                    '''
                    UPDATE cinema.user_statistic 
                    SET total_ratings_received = total_ratings_received + 1,
                        last_activity = %s
                    WHERE user_id = %s
                    ''',
                    (datetime.now(), selected_user_id)
                )
                
                # 3. Обновляем средний рейтинг фильма в session_history
                cur.execute(
                    '''
                    UPDATE cinema.session_history 
                    SET 
                        total_reviews = total_reviews + 1,
                        average_rating = (
                            SELECT AVG(rating)::decimal(3,2)
                            FROM cinema.review 
                            WHERE movie_id = %s AND room_id = %s
                        )
                    WHERE room_id = %s AND movie_id = %s
                    ''',
                    (selected_movie_id, room_id, room_id, selected_movie_id)
                )
                
                # 4. Отправляем уведомление пользователю, чей фильм выбран
                cur.execute(
                    '''
                    INSERT INTO cinema.notification 
                    (user_id, room_id, title, message, type)
                    VALUES (%s, %s, %s, %s, %s)
                    ''',
                    (selected_user_id, room_id, "Новая оценка вашего фильма",
                     f"{user.name} оценил(а) ваш фильм на {review.rating}/10", 
                     "review_time")
                )
                
                conn.commit()
                
                # Получаем полную информацию об отзыве
                cur.execute(
                    '''
                    SELECT 
                        r.id, r.movie_id, r.room_id, r.from_user_id, r.to_user_id,
                        r.rating, r.comment, r.reviewed_at,
                        fu.name as from_user_name, tu.name as to_user_name,
                        m.title as movie_title
                    FROM cinema.review r
                    JOIN cinema.user fu ON r.from_user_id = fu.id
                    JOIN cinema.user tu ON r.to_user_id = tu.id
                    JOIN cinema.movie m ON r.movie_id = m.id
                    WHERE r.id = %s
                    ''',
                    (review_id,)
                )
                
                review_data = cur.fetchone()
                
                return {
                    "message": "Оценка успешно отправлена",
                    "review": {
                        "id": review_data[0],
                        "movie": {
                            "id": review_data[1],
                            "title": review_data[10]
                        },
                        "room_id": review_data[2],
                        "from_user": {
                            "id": review_data[3],
                            "name": review_data[8]
                        },
                        "to_user": {
                            "id": review_data[4],
                            "name": review_data[9]
                        },
                        "rating": review_data[5],
                        "comment": review_data[6],
                        "reviewed_at": review_data[7]
                    }
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при отправке оценки: {str(e)}"
        )

@app.get("/rooms/{room_id}/reviews")
def get_room_reviews(room_id: int, user: User = Depends(get_current_user)):
    """Получить все оценки для выбранного фильма в комнате"""
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Проверяем, что пользователь является участником комнаты
                cur.execute(
                    '''
                    SELECT 1 FROM cinema.room_participant 
                    WHERE room_id = %s AND user_id = %s AND is_active = TRUE
                    ''',
                    (room_id, user.id)
                )
                if not cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Вы не участник этой комнаты"
                    )
                
                # Получаем выбранный фильм комнаты
                cur.execute(
                    '''
                    SELECT selected_movie_id, selected_user_id 
                    FROM cinema.room 
                    WHERE id = %s
                    ''',
                    (room_id,)
                )
                room_info = cur.fetchone()
                
                if not room_info or not room_info[0]:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="В этой комнате не было выбранного фильма"
                    )
                
                selected_movie_id, selected_user_id = room_info
                
                # Получаем информацию о фильме
                cur.execute(
                    '''
                    SELECT title, poster_url FROM cinema.movie WHERE id = %s
                    ''',
                    (selected_movie_id,)
                )
                movie_info = cur.fetchone()
                
                # Получаем информацию о пользователе, предложившем фильм
                cur.execute(
                    '''
                    SELECT name, profile_picture FROM cinema.user WHERE id = %s
                    ''',
                    (selected_user_id,)
                )
                suggested_by_user = cur.fetchone()
                
                # Получаем все оценки для этого фильма в комнате
                cur.execute(
                    '''
                    SELECT 
                        r.id, r.rating, r.comment, r.reviewed_at,
                        u.id as user_id, u.name as user_name, u.profile_picture
                    FROM cinema.review r
                    JOIN cinema.user u ON r.from_user_id = u.id
                    WHERE r.movie_id = %s AND r.room_id = %s
                    ORDER BY r.reviewed_at DESC
                    ''',
                    (selected_movie_id, room_id)
                )
                
                reviews = []
                total_rating = 0
                for row in cur.fetchall():
                    reviews.append({
                        "id": row[0],
                        "rating": row[1],
                        "comment": row[2],
                        "reviewed_at": row[3],
                        "user": {
                            "id": row[4],
                            "name": row[5],
                            "profile_picture": row[6]
                        }
                    })
                    total_rating += row[1]
                
                # Рассчитываем среднюю оценку
                average_rating = round(total_rating / len(reviews), 2) if reviews else 0
                
                # Получаем статистику из session_history
                cur.execute(
                    '''
                    SELECT average_rating, total_reviews, participants_count
                    FROM cinema.session_history
                    WHERE room_id = %s AND movie_id = %s
                    ''',
                    (room_id, selected_movie_id)
                )
                session_stats = cur.fetchone()
                
                return {
                    "movie": {
                        "id": selected_movie_id,
                        "title": movie_info[0] if movie_info else "Неизвестно",
                        "poster_url": movie_info[1] if movie_info else None
                    },
                    "suggested_by": {
                        "id": selected_user_id,
                        "name": suggested_by_user[0] if suggested_by_user else "Неизвестно",
                        "profile_picture": suggested_by_user[1] if suggested_by_user else None
                    },
                    "reviews": {
                        "list": reviews,
                        "count": len(reviews),
                        "average_rating": session_stats[0] if session_stats else average_rating,
                        "total_reviews": session_stats[1] if session_stats else len(reviews),
                        "participants_count": session_stats[2] if session_stats else 0
                    },
                    "has_user_reviewed": any(review["user"]["id"] == user.id for review in reviews)
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении оценок: {str(e)}"
        )

@app.post("/rooms/{room_id}/finish-review")
def finish_review(room_id: int, user: User = Depends(get_current_user)):
    """Завершить этап оценки и перевести комнату в статус finished (только владелец)"""
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Проверяем, что пользователь является владельцем комнаты
                cur.execute(
                    '''
                    SELECT owner_id, status, selected_movie_id 
                    FROM cinema.room 
                    WHERE id = %s
                    ''',
                    (room_id,)
                )
                room = cur.fetchone()
                
                if not room:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Комната не найдена"
                    )
                
                owner_id, room_status, selected_movie_id = room
                
                if user.id != owner_id:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Только владелец комнаты может завершить этап оценки"
                    )
                
                if room_status != 'reviewing':
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Сейчас не идет этап оценки"
                    )
                
                # Обновляем статус комнаты
                cur.execute(
                    '''
                    UPDATE cinema.room 
                    SET status = 'finished', review_ends_at = %s
                    WHERE id = %s
                    ''',
                    (datetime.now(), room_id)
                )
                
                # Отправляем уведомления всем участникам
                cur.execute(
                    '''
                    SELECT u.id FROM cinema.user u
                    JOIN cinema.room_participant rp ON u.id = rp.user_id
                    WHERE rp.room_id = %s AND rp.is_active = TRUE
                    ''',
                    (room_id,)
                )
                participants = cur.fetchall()
                
                for participant in participants:
                    cur.execute(
                        '''
                        INSERT INTO cinema.notification 
                        (user_id, room_id, title, message, type)
                        VALUES (%s, %s, %s, %s, %s)
                        ''',
                        (participant[0], room_id, "Комната завершена",
                         "Этап оценки завершен. Спасибо за участие!", 
                         "room_ended")
                    )
                
                conn.commit()
                
                return {
                    "message": "Этап оценки завершен. Комната переведена в статус 'finished'",
                    "room_id": room_id,
                    "status": "finished",
                    "review_ends_at": datetime.now().isoformat()
                }
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при завершении оценки: {str(e)}"
        )

@app.get("/rooms/{room_id}/has-reviewed")
def has_user_reviewed(room_id: int, user: User = Depends(get_current_user)):
    """Проверить, оценил ли текущий пользователь фильм в этой комнате"""
    try:
        with psycopg2.connect(**db_conn_dict) as conn:
            with conn.cursor() as cur:
                # Получаем выбранный фильм комнаты
                cur.execute(
                    '''
                    SELECT selected_movie_id FROM cinema.room WHERE id = %s
                    ''',
                    (room_id,)
                )
                room = cur.fetchone()
                
                if not room or not room[0]:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="В этой комнате не было выбранного фильма"
                    )
                
                selected_movie_id = room[0]
                
                # Проверяем, есть ли оценка от пользователя
                cur.execute(
                    '''
                    SELECT 1 FROM cinema.review 
                    WHERE movie_id = %s AND room_id = %s AND from_user_id = %s
                    ''',
                    (selected_movie_id, room_id, user.id)
                )
                
                has_reviewed = cur.fetchone() is not None
                
                return {
                    "has_reviewed": has_reviewed,
                    "room_id": room_id,
                    "movie_id": selected_movie_id,
                    "user_id": user.id
                }
                
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при проверке оценки: {str(e)}"
        )

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)