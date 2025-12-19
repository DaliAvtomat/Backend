from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
import bcrypt
import jwt
from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, field_validator, validator
from fastapi import FastAPI, HTTPException, Depends, status, Body, Request
import psycopg2
import psycopg2.extras
import uvicorn
import secrets
import sys
import httpx
from typing import Optional, Dict, Any, List
import asyncio
from contextlib import asynccontextmanager
from kinopoisk_api import kinopoisk_api
from kinopoisk_api import KinopoiskAPI

kp_api = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    global kp_api
    kp_api = KinopoiskAPI()
    print("🚀 Kinopoisk API инициализирован")
    yield
    await kp_api.close()
    print("👋 Kinopoisk API закрыт")

# Обновляем создание FastAPI с lifespan
app = FastAPI(title="MovieRatings API", version="1.0.0", lifespan=lifespan)

SECRET_KEY = "agjohuyh59i2yiq3y9iuqy34iguyaiugy349ty29h"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

db_conn_dict = {
    'database': 'postgres',
    'user': 'postgres',
    'password': 'masandra',
    'host': 'localhost',
    'port': 5432,
    'options': "-c search_path=cinema"
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
app = FastAPI(title="MovieRatings API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= МОДЕЛИ ДЛЯ ФРОНТЕНДА =================
class UserLogin(BaseModel):
    email: str
    password: str

class UserRegister(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=1, max_length=15)
    password: str

class ServerCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    icon: str

class ServerUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=50)
    icon: Optional[str] = None

class MessageCreate(BaseModel):
    text: str
    isAdmin: bool = False

class MovieRatingCreate(BaseModel):
    name: str
    rating: int = Field(..., ge=1, le=5)
    comment: str = ""

class ServerAccessCodeRequest(BaseModel):
    access_code: str = Field(..., min_length=4, max_length=7)

class ServerInviteInfo(BaseModel):
    id: int
    name: str
    admin: str
    participants_count: int
    created_at: datetime
    is_open: bool

class ServerJoinResponse(BaseModel):
    success: bool
    message: str
    server: Dict[str, Any]

class AccessCodeResponse(BaseModel):
    success: bool
    server_id: int
    server_name: str
    access_code: str
    invite_link: str
    is_open: bool

class UserUpdateRequest(BaseModel):
    username: Optional[str] = Field(None, min_length=1, max_length=15)
    avatar: Optional[str] = Field(None, max_length=255)
    
    @validator('avatar')
    def validate_avatar(cls, v):
        if v and len(v) > 255:
            raise ValueError('URL аватара слишком длинный')
        return v
    

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

class MovieSuggestion(BaseModel):
    """Модель для предложения фильма в рулетку"""
    movie_data: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    title: Optional[str] = None
    
    @validator('movie_data', always=True)
    def validate_movie_data(cls, v, values):
        """Проверяем, что есть либо movie_data, либо name/title"""
        if not v and not values.get('name') and not values.get('title'):
            raise ValueError('Должны быть предоставлены данные фильма')
        return v

# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =================
def get_db_connection():
    conn = psycopg2.connect(**db_conn_dict)
    conn.autocommit = False
    return conn

def get_hashed_password(password: str) -> bytes:
    """Хеширование пароля, возвращает bytes"""
    password_bytes = password.encode('utf-8')
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt())

def verify_password(plain_password: str, hashed_password: bytes) -> bool:
    """Проверка пароля, ожидает bytes"""
    try:
        if isinstance(hashed_password, memoryview):
            hashed_bytes = bytes(hashed_password)
        elif isinstance(hashed_password, bytes):
            hashed_bytes = hashed_password
        elif isinstance(hashed_password, str):
            # Пытаемся преобразовать строку в bytes
            try:
                hashed_bytes = hashed_password.encode('utf-8')
            except:
                try:
                    hashed_bytes = hashed_password.encode('latin-1')
                except:
                    # Если строка в hex формате
                    try:
                        if len(hashed_password) % 2 == 0:
                            hashed_bytes = bytes.fromhex(hashed_password)
                        else:
                            return False
                    except:
                        return False
        else:
            return False
            
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_bytes)
    except Exception as e:
        print(f"Password verification error: {e}")
        return False

async def search_movie_by_keyword(keyword: str) -> Optional[MovieInfo]:
    """Поиск фильма по ключевому слову через Kinopoisk API"""
    try:
        movie_data = await kinopoisk_api.search_movie_by_keyword(keyword)
        
        if movie_data:
            return MovieInfo(
                id=movie_data["id"],
                title=movie_data["title"],
                description=movie_data["description"],
                duration=movie_data["duration"],
                release_year=movie_data["release_year"],
                poster_url=movie_data["poster_url"],
                rating_kp=movie_data["rating_kp"],
                votes_kp=movie_data["votes_kp"],
                genres=movie_data["genres"],
                
            )
    except Exception as e:
        print(f"Ошибка при поиске фильма '{keyword}': {str(e)}")
    
    return None

def get_user_by_email(email: str) -> Optional[Dict]:
    """Получение пользователя по email с обработкой BYTEA поля"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    '''
                    SELECT 
                        id, 
                        name, 
                        email, 
                        password_hash,
                        profile_picture, 
                        overall_rating, 
                        is_active, 
                        registered_at 
                    FROM "user" 
                    WHERE email = %s
                    ''',
                    (email,)
                )
                user = cur.fetchone()
                if user:
                    # Преобразуем в словарь
                    user_dict = dict(user)
                    
                    # Обрабатываем password_hash
                    password_hash = user_dict['password_hash']
                    
                    # Если это memoryview (BYTEA из PostgreSQL), преобразуем в bytes
                    if isinstance(password_hash, memoryview):
                        user_dict['password_hash'] = bytes(password_hash)
                    
                    return user_dict
                return None
    except Exception as e:
        print(f"❌ Error getting user by email {email}: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_user_by_id(user_id: int) -> Optional[Dict]:
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    'SELECT id, name, email, profile_picture, overall_rating, registered_at, is_active FROM "user" WHERE id = %s',
                    (user_id,)
                )
                user = cur.fetchone()
                return dict(user) if user else None
    except Exception as e:
        print(f"Error getting user by id: {e}")
        return None

def authenticate_user(email: str, password: str) -> Dict:
    """Аутентификация пользователя"""
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Пользователь не найден'
        )
    
    # Получаем хеш пароля
    password_hash = user.get('password_hash')
    if not password_hash:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Ошибка аутентификации'
        )
    
    if not verify_password(password, password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Неверный пароль'
        )
    
    return user

def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict:
    """Получение текущего пользователя из JWT токена"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Не удалось проверить учетные данные",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise credentials_exception
        user = get_user_by_email(email)
        if user is None:
            raise credentials_exception
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Токен истек"
        )
    except jwt.InvalidTokenError:
        raise credentials_exception

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Создание JWT токена"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_movie_from_kinopoisk(query: str) -> Optional[Dict]:
    """Получение информации о фильме из Kinopoisk API"""
    try:
        if kp_api:
            movie_info = await kp_api.search_movie_by_keyword(query)
            if movie_info:
                # Сохраняем фильм в базу
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute('''
                            INSERT INTO cinema.movie 
                            (title, description, duration, release_year, poster_url, rating_kp, votes_kp)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (title) DO UPDATE SET
                            description = EXCLUDED.description,
                            rating_kp = EXCLUDED.rating_kp
                            RETURNING id
                        ''', (
                            movie_info['title'],
                            movie_info['description'],
                            movie_info['duration'],
                            movie_info['release_year'],
                            movie_info['poster_url'],
                            movie_info['rating_kp'],
                            movie_info['votes_kp']
                        ))
                        
                        movie_id = cur.fetchone()[0]
                        conn.commit()
                        
                        # Сохраняем жанры
                        if movie_info.get('genres'):
                            for genre_name in movie_info['genres']:
                                cur.execute('''
                                    INSERT INTO cinema.genre (name)
                                    VALUES (%s)
                                    ON CONFLICT (name) DO NOTHING
                                    RETURNING id
                                ''', (genre_name,))
                                
                                genre_result = cur.fetchone()
                                if genre_result:
                                    genre_id = genre_result[0]
                                    cur.execute('''
                                        INSERT INTO cinema.movie_genre (movie_id, genre_id)
                                        VALUES (%s, %s)
                                        ON CONFLICT DO NOTHING
                                    ''', (movie_id, genre_id))
                        
                        conn.commit()
                        movie_info['id'] = movie_id
                        return movie_info
    except Exception as e:
        print(f"❌ Ошибка получения фильма из Kinopoisk: {e}")
    
    return None

def suggest_movie_for_roulette(server_id: int, movie_name: str, user_id: int):
    """Добавление фильма в рулетку"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Сначала проверяем/создаем фильм
                cur.execute('''
                    SELECT id FROM cinema.movie 
                    WHERE LOWER(title) = LOWER(%s)
                ''', (movie_name,))
                
                movie = cur.fetchone()
                movie_id = None
                
                if movie:
                    movie_id = movie[0]
                else:
                    movie_info = get_movie_from_kinopoisk(movie_name)
                    cur.execute('''
                        INSERT INTO cinema.movie 
                        (title, description, duration, release_year, poster_url, rating_kp, votes_kp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (title) DO UPDATE SET
                        description = EXCLUDED.description,
                        rating_kp = EXCLUDED.rating_kp
                        RETURNING id
                    ''', (
                        movie_info['title'],
                        movie_info['description'],
                        movie_info['duration'],
                        movie_info['release_year'],
                        movie_info['poster_url'],
                        movie_info['rating_kp'],
                        movie_info['votes_kp']
                        ))
                
                # Добавляем предложение
                cur.execute('''
                    INSERT INTO cinema.suggested_movie 
                    (movie_id, room_id, user_id, is_active, in_roulette)
                    VALUES (%s, %s, %s, TRUE, TRUE)
                    ON CONFLICT (room_id, user_id) WHERE is_active = TRUE 
                    DO UPDATE SET 
                        movie_id = EXCLUDED.movie_id,
                        suggested_at = CURRENT_TIMESTAMP,
                        in_roulette = TRUE
                    RETURNING id
                ''', (movie_id, server_id, user_id))
                
                conn.commit()
                return movie_id
    except Exception as e:
        print(f"Error suggesting movie: {e}")
        raise

async def add_movie_to_in_roulette(server_id: int, movie_data: Dict, user_id: int) -> Dict:
    """Добавляет фильм в таблицу in_roulette (фильмы для рулетки)"""
    try:
        print(f"🎬 Добавляем фильм в рулетку: {movie_data.get('title')}")
        
        movie_name = movie_data.get('title', '').strip()
        if not movie_name:
            raise ValueError("Название фильма обязательно")
        
        movie_id = None
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # 1. Сохраняем/обновляем фильм в таблице movie
                cur.execute('''
                    SELECT id FROM cinema.movie 
                    WHERE LOWER(title) = LOWER(%s)
                ''', (movie_name,))
                
                existing_movie = cur.fetchone()
                
                if existing_movie:
                    movie_id = existing_movie[0]
                    print(f"✅ Фильм уже существует в базе с ID: {movie_id}")
                    
                    # Обновляем данные фильма
                    cur.execute('''
                        UPDATE cinema.movie 
                        SET 
                            title = COALESCE(%s, title),
                            description = COALESCE(%s, description),
                            duration = COALESCE(%s, duration),
                            release_year = COALESCE(%s, release_year),
                            poster_url = COALESCE(%s, poster_url),
                            rating_kp = COALESCE(%s, rating_kp),
                            votes_kp = COALESCE(%s, votes_kp),
                        WHERE id = %s
                    ''', (
                        movie_data.get('title'),
                        movie_data.get('description'),
                        movie_data.get('duration'),
                        movie_data.get('release_year'),
                        movie_data.get('poster_url'),
                        movie_data.get('rating_kp'),
                        movie_data.get('votes_kp'),
                        movie_id
                    ))
                else:
                    # Создаем новую запись фильма
                    cur.execute('''
                        INSERT INTO cinema.movie 
                        (title, description, duration, release_year, poster_url, 
                         rating_kp, votes_kp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    ''', (
                        movie_name,
                        movie_data.get('description', ''),
                        movie_data.get('duration'),
                        movie_data.get('release_year'),
                        movie_data.get('poster_url'),
                        movie_data.get('rating_kp'),
                        movie_data.get('votes_kp')
                    ))
                    
                    movie_id = cur.fetchone()[0]
                    print(f"✅ Создан фильм с ID: {movie_id}")
                
                # 2. Проверяем, не предлагал ли уже пользователь этот фильм в этой комнате
                cur.execute('''
                    SELECT id FROM cinema.in_roulette 
                    WHERE movie_id = %s AND user_id = %s AND room_id = %s
                ''', (movie_id, user_id, server_id))
                
                existing_suggestion = cur.fetchone()
                
                if existing_suggestion:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Вы уже предложили этот фильм в этой комнате"
                    )
                
                # 3. Добавляем фильм в таблицу in_roulette
                cur.execute('''
                    INSERT INTO cinema.in_roulette 
                    (movie_id, user_id, room_id)
                    VALUES (%s, %s, %s)
                    RETURNING id
                ''', (movie_id, user_id, server_id))
                
                roulette_id = cur.fetchone()[0]
                print(f"✅ Фильм добавлен в рулетку с ID: {roulette_id}")
                
                # 4. Обновляем статистику пользователя
                cur.execute('''
                    UPDATE cinema.user_statistic 
                    SET movies_suggested = movies_suggested + 1
                    WHERE user_id = %s
                ''', (user_id,))
                
                conn.commit()
                
                return {
                    "success": True,
                    "roulette_id": roulette_id,
                    "movie_id": movie_id,
                    "movie_title": movie_name,
                    "user_id": user_id,
                    "server_id": server_id
                }
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ошибка добавления фильма в рулетку: {e}")
        import traceback
        traceback.print_exc()
        raise

def get_roulette_movies(server_id: int) -> List[Dict]:
    """Получение фильмов для рулетки из таблицы in_roulette"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # Получаем фильмы из таблицы in_roulette для конкретной комнаты
                cur.execute('''
                    SELECT 
                        ir.id as roulette_id,
                        m.id as movie_id,
                        m.title as name,
                        m.description,
                        m.duration,
                        m.release_year,
                        m.poster_url,
                        m.rating_kp,
                        u.id as user_id,
                        u.name as suggested_by
                    FROM cinema.in_roulette ir
                    JOIN cinema.movie m ON ir.movie_id = m.id
                    JOIN cinema."user" u ON ir.user_id = u.id
                    WHERE ir.room_id = %s 
                    ORDER BY ir.id DESC
                ''', (server_id,))
                
                movies = cur.fetchall()
                return [dict(movie) for movie in movies]
    except Exception as e:
        print(f"Error getting roulette movies: {e}")
        return []


def spin_roulette(server_id: int) -> Optional[Dict]:
    """Запуск рулетки и выбор случайного фильма"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # Получаем все фильмы для рулетки
                movies = get_roulette_movies(server_id)
                if not movies:
                    return None
                
                # Случайный выбор
                import random
                selected_movie = random.choice(movies)
                
                # Получаем user_id пользователя, предложившего фильм
                cur.execute('''
                    SELECT user_id 
                    FROM cinema.suggested_movie 
                    WHERE movie_id = %s AND room_id = %s AND is_active = TRUE 
                    LIMIT 1
                ''', (selected_movie['movie_id'], server_id))
                
                user_result = cur.fetchone()
                selected_user_id = user_result[0] if user_result else None
                
                # Если нет user_id, используем системного пользователя (ID=1)
                #if selected_user_id is None:
                    #cur.execute('SELECT id FROM cinema."user" WHERE email = %s', ('system@movieratings.com',))
                    #system_user = cur.fetchone()
                    #selected_user_id = system_user[0] if system_user else 1
                
                # Обновляем статус комнаты
                if selected_user_id:
                    cur.execute('''
                        UPDATE cinema.room 
                        SET 
                            status = 'watching',
                            selected_movie_id = %s,
                            selected_user_id = %s,
                            watching_starts_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        RETURNING id
                    ''', (selected_movie['movie_id'], selected_user_id, server_id))
                
                    # Записываем в историю рулетки
                    cur.execute('''
                        INSERT INTO cinema.roulette_history 
                        (room_id, selected_movie_id, selected_user_id, candidates_count, spin_duration)
                        VALUES (%s, %s, %s, %s, %s)
                    ''', (server_id, selected_movie['movie_id'], selected_user_id, len(movies), 3))
                
                    cur.execute('''
                        UPDATE cinema.user_statistic 
                        SET movies_selected = movies_selected + 1
                        WHERE user_id = %s
                    ''', (selected_user_id,))
                else:
                    add_system_message(
                    server_id, 
                    f" No users)"
                )
                # Обновляем статистику пользователя, если user_id найден
                #if selected_user_id:  # Не обновляем статистику системного пользователя
                   # cur.execute('''
                       # UPDATE cinema.user_statistic 
                       # SET movies_selected = movies_selected + 1
                       # WHERE user_id = %s
                   # ''', (selected_user_id,))
                
                conn.commit()
                
                # Добавляем системное сообщение
                add_system_message(
                    server_id, 
                    f"🎲 Выбран фильм для просмотра: **{selected_movie['name']}** (предложил: {selected_movie.get('suggested_by', 'система')})"
                )
                
                return {
                    "id": selected_movie['movie_id'],
                    "name": selected_movie['name'],
                    "suggested_by": selected_movie.get('suggested_by', 'система'),
                    "suggested_by_id": selected_user_id,
                    "candidates_count": len(movies)
                }
    except Exception as e:
        print(f"Error spinning roulette: {e}")
        import traceback
        traceback.print_exc()
        return None

async def suggest_movie_for_roulette_endpoint(
    server_id: int, 
    suggestion: Dict, 
    user: Dict = Depends(get_current_user)
):
    """Предложить фильм для рулетки - добавляет в таблицу in_roulette"""
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ ПРЕДЛОЖЕНИИ ФИЛЬМА
        update_user_activity(user['id'])
        
        movie_data = suggestion.get('movie_data')
        if not movie_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Данные фильма обязательны"
            )
        
        movie_name = movie_data.get('title', '').strip()
        if not movie_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Название фильма обязательно"
            )
        
        # Проверяем, что пользователь является участником
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT 1 FROM cinema.room_participant 
                    WHERE room_id = %s AND user_id = %s AND is_active = TRUE
                ''', (server_id, user['id']))
                
                if not cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Вы не участник этого сервера"
                    )
        
        print(f"🎬 Пользователь {user['name']} предлагает фильм для рулетки: {movie_name}")
        
        # Добавляем фильм в таблицу in_roulette
        result = await add_movie_to_in_roulette(server_id, movie_data, user['id'])
        
        # Добавляем системное сообщение
        add_system_message(
            server_id, 
            f"🎬 {user['name']} предложил(а) фильм для рулетки: **{movie_name}**"
        )
        
        return {
            "success": True,
            "message": f"Фильм '{movie_name}' добавлен в рулетку",
            "roulette_id": result['roulette_id'],
            "movie_id": result['movie_id'],
            "movie_title": result['movie_title']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error suggesting movie: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при предложении фильма: {str(e)}"
        )

def add_system_message(server_id: int, text: str):
    """Добавление системного сообщения"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Используем ID системного пользователя (например, 0 или 1)
                cur.execute('''
                    INSERT INTO cinema.room_chat 
                    (room_id, user_id, message, message_type)
                    VALUES (%s, 1, %s, 'system')
                ''', (server_id, text))
                conn.commit()
    except Exception as e:
        print(f"Error adding system message: {e}")

def get_server_participants(server_id: int) -> List[Dict]:
    """Получение участников сервера с их статусами готовности"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute('''
                    SELECT 
                        u.id,
                        u.name as username,
                        u.profile_picture as avatar,
                        CASE 
                            WHEN r.owner_id = u.id THEN 'admin'
                            ELSE 'user'
                        END as role,
                        FALSE as is_ready  -- В будущем можно добавить статус готовности
                    FROM cinema.room_participant rp
                    JOIN cinema."user" u ON rp.user_id = u.id
                    JOIN cinema.room r ON rp.room_id = r.id
                    WHERE rp.room_id = %s AND rp.is_active = TRUE
                    ORDER BY 
                        CASE WHEN r.owner_id = u.id THEN 1 ELSE 2 END,
                        u.name
                ''', (server_id,))
                
                participants = cur.fetchall()
                return [dict(participant) for participant in participants]
    except Exception as e:
        print(f"Error getting participants: {e}")
        return []


# ================= ФУНКЦИИ ДЛЯ ОНЛАЙН СТАТУСОВ (НОВЫЕ) =================

def update_user_activity(user_id: int):
    """Обновить время последней активности пользователя"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    UPDATE cinema.user_statistic 
                    SET last_activity = CURRENT_TIMESTAMP
                    WHERE user_id = %s
                ''', (user_id,))
                conn.commit()
    except Exception as e:
        print(f"❌ Error updating user activity: {e}")

def _format_time_ago(td: timedelta) -> str:
    """Форматирование времени в понятный формат"""
    if td < timedelta(minutes=1):
        return "только что"
    elif td < timedelta(hours=1):
        minutes = int(td.total_seconds() // 60)
        return f"{minutes} мин. назад"
    elif td < timedelta(days=1):
        hours = int(td.total_seconds() // 3600)
        return f"{hours} ч. назад"
    else:
        days = int(td.days)
        return f"{days} дн. назад"

# ================= ЭНДПОИНТЫ API =================

@app.get("/api/health", tags=["API"])
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.post("/api/auth/login", tags=["API"])
async def login(login_data: UserLogin):
    try:
        print(f"🔑 Попытка входа: {login_data.email}")
        
        user = authenticate_user(login_data.email, login_data.password)
        
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ ВХОДЕ
        update_user_activity(user['id'])
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user['email']}, 
            expires_delta=access_token_expires
        )
        
        return {
            "success": True,
            "token": access_token,
            "user": {
                "id": user['id'],
                "username": user['name'],
                "email": user['email'],
                "avatar": user.get('profile_picture') or "👤",
                "status": "online",
                "overall_rating": float(user.get('overall_rating', 0)),
                "registered_at": user.get('registered_at', datetime.now()).isoformat()
            }
        }
    except HTTPException as e:
        return {
            "success": False,
            "message": e.detail
        }
    except Exception as e:
        print(f"Login error: {e}")
        return {
            "success": False,
            "message": "Ошибка сервера"
        }

@app.post("/api/auth/register", tags=["API"])
async def register(user_data: UserRegister):
    try:
        print(f"📝 Регистрация пользователя: {user_data.email}")
        
        # Проверяем, существует ли пользователь
        existing_user = get_user_by_email(user_data.email)
        if existing_user:
            return {
                "success": False,
                "message": "Email уже используется"
            }
        
        # Хешируем пароль (получаем bytes)
        hashed_password_bytes = get_hashed_password(user_data.password)
        print(f"🔐 Хеш пароля создан: {len(hashed_password_bytes)} байт")
        
        # Проверяем, что хеш работает
        if not verify_password(user_data.password, hashed_password_bytes):
            return {
                "success": False,
                "message": "Ошибка при создании хеша пароля"
            }
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Сохраняем bytes напрямую (PostgreSQL BYTEA)
                cur.execute(
                    '''
                    INSERT INTO "user" (name, email, password_hash, is_active, registered_at)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id, name, email, registered_at
                    ''',
                    (
                        user_data.username, 
                        user_data.email, 
                        psycopg2.Binary(hashed_password_bytes),  # Используем Binary для BYTEA
                        True, 
                        datetime.now()
                    )
                )
                
                result = cur.fetchone()
                if not result:
                    return {
                        "success": False,
                        "message": "Не удалось создать пользователя"
                    }
                    
                user_id, name, email, registered_at = result
                conn.commit()
                
                print(f"✅ Пользователь создан: ID={user_id}")
        
        # Создаем токен
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": email}, 
            expires_delta=access_token_expires
        )
        
        return {
            "success": True,
            "token": access_token,
            "user": {
                "id": user_id,
                "username": name,
                "email": email,
                "avatar": "👤",
                "status": "online",
                "overall_rating": 0.0,
                "registered_at": registered_at.isoformat() if registered_at else datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        print(f"❌ Ошибка регистрации: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": "Ошибка при регистрации"
        }

@app.get("/api/servers", tags=["API"])
async def get_user_servers(user: Dict = Depends(get_current_user)):
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ ЗАПРОСЕ СЕРВЕРОВ
        update_user_activity(user['id'])
        
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute('''
                    SELECT r.*, u.name as owner_name
                    FROM room r
                    JOIN "user" u ON r.owner_id = u.id
                    WHERE r.id IN (
                        SELECT room_id 
                        FROM room_participant 
                        WHERE user_id = %s AND is_active = TRUE
                    )
                    AND r.status != 'finished'
                    ORDER BY r.created_at DESC
                ''', (user['id'],))
                
                rooms = cur.fetchall()
                
                result = []
                for room in rooms:
                    cur.execute('''
                        SELECT u.id, u.name, u.email, rp.role
                        FROM room_participant rp
                        JOIN "user" u ON rp.user_id = u.id
                        WHERE rp.room_id = %s AND rp.is_active = TRUE
                        ORDER BY rp.role, u.name
                    ''', (room['id'],))
                    
                    participants = cur.fetchall()
                    
                    result.append({
                        "id": room['id'],
                        "name": room['name'],
                        "icon": "🎬",
                        "admin": room['owner_name'],
                        "users": [p['name'] for p in participants],
                        "createdAt": room['created_at'].isoformat() if room['created_at'] else datetime.now().isoformat(),
                        "messages": [],
                        "ratedMovies": [],
                        "status": room['status'],
                        "is_open": room['is_open']
                    })
                
                return result
    except Exception as e:
        print(f"Error getting user servers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении серверов: {str(e)}"
        )

@app.post("/api/servers", tags=["API"])
async def create_server(server_data: ServerCreate, user: Dict = Depends(get_current_user)):
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ СОЗДАНИИ СЕРВЕРА
        update_user_activity(user['id'])
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                access_code = secrets.token_hex(3).upper()
                
                cur.execute('''
                    INSERT INTO room (name, owner_id, is_open, access_code, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id, name, owner_id, created_at
                ''', (
                    server_data.name, 
                    user['id'], 
                    True, 
                    access_code, 
                    'collecting', 
                    datetime.now()
                ))
                
                room_id, room_name, owner_id, created_at = cur.fetchone()
                
                cur.execute('''
                    INSERT INTO room_participant (room_id, user_id, role, is_active)
                    VALUES (%s, %s, %s, %s)
                ''', (room_id, user['id'], 'owner', True))
                
                conn.commit()
        
        return {
            "id": room_id,
            "name": room_name,
            "icon": server_data.icon,
            "admin": user['name'],
            "users": [user['name']],
            "createdAt": created_at.isoformat(),
            "messages": [],
            "ratedMovies": [],
            "status": "collecting",
            "is_open": True
        }
    except Exception as e:
        print(f"Error creating server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при создании сервера: {str(e)}"
        )

@app.put("/api/servers/{server_id}", tags=["API"])
async def update_server(server_id: int, updates: ServerUpdate, user: Dict = Depends(get_current_user)):
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ ОБНОВЛЕНИИ СЕРВЕРА
        update_user_activity(user['id'])
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT owner_id FROM room WHERE id = %s',
                    (server_id,)
                )
                room = cur.fetchone()
                
                if not room:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Сервер не найден"
                    )
                
                if room[0] != user['id']:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Только владелец может изменять сервер"
                    )
                
                update_fields = []
                update_values = []
                
                if updates.name:
                    update_fields.append("name = %s")
                    update_values.append(updates.name)
                
                if update_fields:
                    update_values.append(server_id)
                    cur.execute(
                        f'UPDATE room SET {", ".join(update_fields)} WHERE id = %s',
                        update_values
                    )
                    conn.commit()
        
        return {"success": True, "message": "Сервер обновлен"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обновлении сервера: {str(e)}"
        )

@app.delete("/api/servers/{server_id}", tags=["API"])
async def delete_server(server_id: int, user: Dict = Depends(get_current_user)):
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ УДАЛЕНИИ СЕРВЕРА
        update_user_activity(user['id'])
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT owner_id FROM room WHERE id = %s',
                    (server_id,)
                )
                room = cur.fetchone()
                
                if not room:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Сервер не найден"
                    )
                
                if room[0] != user['id']:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Только владелец может удалить сервер"
                    )
                
                cur.execute('DELETE FROM room WHERE id = %s', (server_id,))
                conn.commit()
        
        return {"success": True, "message": "Сервер удален"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при удалении сервера: {str(e)}"
        )

@app.get("/api/messages/{server_id}", tags=["API"])
async def get_messages(server_id: int, user: Dict = Depends(get_current_user)):
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ ПОЛУЧЕНИИ СООБЩЕНИЙ
        update_user_activity(user['id'])
        
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    'SELECT 1 FROM room_participant WHERE room_id = %s AND user_id = %s AND is_active = TRUE',
                    (server_id, user['id'])
                )
                
                if not cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Вы не участник этой комнаты"
                    )
                
                cur.execute('''
                    SELECT rc.id, rc.message as text, u.name as user, 
                           rc.sent_at as time, 
                           CASE WHEN u.id = r.owner_id THEN TRUE ELSE FALSE END as is_admin
                    FROM room_chat rc
                    JOIN "user" u ON rc.user_id = u.id
                    JOIN room r ON rc.room_id = r.id
                    WHERE rc.room_id = %s
                    ORDER BY rc.sent_at ASC
                    LIMIT 100
                ''', (server_id,))
                
                messages = cur.fetchall()
                
                result = []
                for msg in messages:
                    result.append({
                        "id": msg['id'],
                        "user": msg['user'],
                        "text": msg['text'],
                        "time": msg['time'].strftime("%H:%M") if msg['time'] else "",
                        "isAdmin": msg['is_admin']
                    })
                
                return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении сообщений: {str(e)}"
        )

@app.post("/api/messages/{server_id}", tags=["API"])
async def send_message(server_id: int, message_data: MessageCreate, user: Dict = Depends(get_current_user)):
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ ОТПРАВКЕ СООБЩЕНИЯ
        update_user_activity(user['id'])
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT id, owner_id FROM room WHERE id = %s',
                    (server_id,)
                )
                room = cur.fetchone()
                
                if not room:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Комната не найдена"
                    )
                
                room_id, owner_id = room
                
                cur.execute(
                    'SELECT 1 FROM room_participant WHERE room_id = %s AND user_id = %s AND is_active = TRUE',
                    (server_id, user['id'])
                )
                
                if not cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Вы не участник этой комнаты"
                    )
                
                is_admin = (user['id'] == owner_id)
                
                cur.execute('''
                    INSERT INTO room_chat (room_id, user_id, message, message_type)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, sent_at
                ''', (server_id, user['id'], message_data.text, 'text'))
                
                message_id, sent_at = cur.fetchone()
                conn.commit()
        
        return {
            "id": message_id,
            "user": user['name'],
            "text": message_data.text,
            "time": sent_at.strftime("%H:%M"),
            "isAdmin": is_admin
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error sending message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при отправке сообщения: {str(e)}"
        )

@app.post("/api/movies/ratings/{server_id}", tags=["API"])
async def save_movie_rating_endpoint(
    server_id: int, 
    rating_data: MovieRatingCreate, 
    user: Dict = Depends(get_current_user)
):
    """Сохранить оценку фильма"""
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ ОЦЕНКЕ ФИЛЬМА
        update_user_activity(user['id'])
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Проверяем существование комнаты
                cur.execute('SELECT id, status FROM cinema.room WHERE id = %s', (server_id,))
                room = cur.fetchone()
                
                if not room:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Комната не найдена"
                    )
                
                room_id, room_status = room
                
                # Проверяем, что пользователь является участником
                cur.execute('''
                    SELECT 1 FROM cinema.room_participant 
                    WHERE room_id = %s AND user_id = %s AND is_active = TRUE
                ''', (server_id, user['id']))
                
                if not cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Вы не участник этой комнаты"
                    )
                
                # Находим или создаем фильм
                cur.execute(
                    'SELECT id FROM cinema.movie WHERE LOWER(title) = LOWER(%s)',
                    (rating_data.name,)
                )
                
                movie = cur.fetchone()
                movie_id = None
                
                if movie:
                    movie_id = movie[0]
                else:
                    cur.execute('''
                        INSERT INTO cinema.movie (title, description)
                        VALUES (%s, %s)
                        RETURNING id
                    ''', (rating_data.name, f"Фильм '{rating_data.name}', оцененный пользователем {user['name']}"))
                    
                    movie_id = cur.fetchone()[0]
                
                # Создаем запись в истории сеансов
                cur.execute('''
                    INSERT INTO cinema.session_history 
                    (room_id, movie_id, suggested_by_user_id, watched_at, participants_count)
                    VALUES (%s, %s, %s, %s, 1)
                    RETURNING id
                ''', (server_id, movie_id, user['id'], datetime.now()))
                
                session_id = cur.fetchone()[0]
                
                # Обновляем статистику пользователя
                cur.execute('''
                    UPDATE cinema.user_statistic 
                    SET movies_suggested = movies_suggested + 1,
                        last_activity = %s
                    WHERE user_id = %s
                ''', (datetime.now(), user['id']))
                
                # Создаем отзыв (если есть другие участники)
                cur.execute('''
                    SELECT user_id FROM cinema.room_participant 
                    WHERE room_id = %s AND user_id != %s AND is_active = TRUE
                ''', (server_id, user['id']))
                
                other_participants = cur.fetchall()
                
                # В будущем можно добавить отзывы для других участников
                # Пока просто сохраняем основную оценку
                
                # Добавляем сообщение в чат
                message_text = f"⭐ {user['name']} оценил(а) фильм '{rating_data.name}' на {rating_data.rating}/5"
                if rating_data.comment:
                    message_text += f": \"{rating_data.comment}\""
                
                cur.execute('''
                    INSERT INTO cinema.room_chat 
                    (room_id, user_id, message, message_type)
                    VALUES (%s, %s, %s, 'system')
                ''', (server_id, user['id'], message_text))
                
                conn.commit()
                
                # Возвращаем данные в формате, ожидаемом фронтендом
                return {
                    "success": True,
                    "id": session_id,
                    "name": rating_data.name,
                    "rating": rating_data.rating,
                    "ratedBy": user['name'],
                    "server": str(server_id),
                    "date": datetime.now().strftime("%d.%m.%Y"),
                    "comment": rating_data.comment or ""
                }
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error saving movie rating: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при сохранении оценки: {str(e)}"
        )

@app.get("/api/movies/ratings/{server_id}", tags=["API"])
async def get_movie_ratings(server_id: int, user: Dict = Depends(get_current_user)):
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ ПОЛУЧЕНИИ ОЦЕНОК
        update_user_activity(user['id'])
        
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute('''
                    SELECT sh.id, m.title as name, u.name as rated_by, 
                           sh.watched_at as date, sh.average_rating as rating
                    FROM session_history sh
                    JOIN movie m ON sh.movie_id = m.id
                    JOIN "user" u ON sh.suggested_by_user_id = u.id
                    WHERE sh.room_id = %s
                    ORDER BY sh.watched_at DESC
                ''', (server_id,))
                
                ratings = cur.fetchall()
                
                result = []
                for rating in ratings:
                    result.append({
                        "id": rating['id'],
                        "name": rating['name'],
                        "ratedBy": rating['rated_by'],
                        "date": rating['date'].strftime("%d.%m.%Y") if rating['date'] else "",
                        "rating": float(rating['rating']) if rating['rating'] else 0,
                        "comment": ""
                    })
                
                return result
    except Exception as e:
        print(f"Error getting movie ratings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении оценок: {str(e)}"
        )


@app.put("/api/users/{user_id}", tags=["API"])
async def update_user_profile(
    user_id: int, 
    # Используем Body с любой JSON структурой
    updates: Any = Body(...),
    user: Dict = Depends(get_current_user)
):
    """Обновление профиля пользователя (гибкая версия)"""
    try:
        update_user_activity(user['id'])
        
        print(f"🔍 Update user profile (flexible):")
        print(f"   - Updates received: {updates}")
        print(f"   - Updates type: {type(updates)}")
        
        # Проверяем права
        current_user_id = user.get('id')
        if isinstance(current_user_id, str):
            try:
                current_user_id = int(current_user_id)
            except ValueError:
                print(f"❌ Invalid user ID format: {current_user_id}")
        
        if current_user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Нельзя обновлять чужой профиль"
            )
        
        # Определяем, какие данные пришли
        if isinstance(updates, dict):
            update_data = updates
        else:
            # Пытаемся преобразовать в словарь
            try:
                update_data = updates.dict(exclude_unset=True)
            except AttributeError:
                update_data = {}
        
        print(f"   - Processed update data: {update_data}")
        
        # Проверяем данные
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Нет данных для обновления"
            )
        
        # Валидация username
        username = None
        if 'username' in update_data:
            username = update_data['username']
        elif 'name' in update_data:
            username = update_data['name']
        
        # Валидация avatar
        avatar = None
        if 'avatar' in update_data:
            avatar = update_data['avatar']
        elif 'profile_picture' in update_data:
            avatar = update_data['profile_picture']
        
        # Применяем валидацию как в UserUpdateRequest
        update_fields = []
        update_values = []
        
        if username:
            username = str(username).strip()
            if len(username) < 1 or len(username) > 15:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Имя должно быть от 1 до 15 символов"
                )
            update_fields.append("name = %s")
            update_values.append(username)
            print(f"   - Will update username to: {username}")
        
        if avatar:
            avatar = str(avatar)
            # Базовая проверка base64
            if avatar and not avatar.startswith('data:image/'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail='Некорректный формат аватара. Должен быть base64 изображение.'
                )
            
            # Проверка размера
            if len(avatar) > 1_000_000:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail='Аватар слишком большой. Максимальный размер: 1MB.'
                )
            
            update_fields.append("profile_picture = %s")
            update_values.append(avatar)
            print(f"   - Will update avatar (length: {len(avatar)})")
        
        if not update_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Нет валидных данных для обновления"
            )
        
        # Обновляем базу данных
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT id FROM cinema."user" WHERE id = %s', (user_id,))
                if not cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Пользователь не найден"

)
                
                update_values.append(user_id)
                query = f'UPDATE cinema."user" SET {", ".join(update_fields)} WHERE id = %s'
                print(f"   - Executing query: {query}")
                
                cur.execute(query, update_values)
                conn.commit()
        
        # Возвращаем обновленные данные
        updated_user = get_user_by_id(user_id)
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Не удалось получить обновленные данные"
            )
        
        return {
            "success": True,
            "message": "Профиль успешно обновлен",
            "user": {
                "id": updated_user['id'],
                "username": updated_user['name'],
                "email": updated_user['email'],
                "avatar": updated_user.get('profile_picture') or "👤",
                "overall_rating": float(updated_user.get('overall_rating', 0)),
                "registered_at": updated_user.get('registered_at').isoformat() if updated_user.get('registered_at') else datetime.now().isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating user profile: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обновлении профиля: {str(e)}"
        )
@app.get("/api/users/search", tags=["API"])
async def search_users(query: str, user: Dict = Depends(get_current_user)):
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ ПОИСКЕ ПОЛЬЗОВАТЕЛЕЙ
        update_user_activity(user['id'])
        
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute('''
                    SELECT id, name, email, profile_picture, overall_rating
                    FROM "user"
                    WHERE LOWER(name) LIKE LOWER(%s)
                    AND id != %s
                    LIMIT 10
                ''', (f"%{query}%", user['id']))
                
                users = cur.fetchall()
                
                result = []
                for u in users:
                    result.append({
                        "id": u['id'],
                        "username": u['name'],
                        "email": u['email'],
                        "avatar": u['profile_picture'] or "👤",
                        "overall_rating": float(u['overall_rating']) if u['overall_rating'] else 0
                    })
                
                return result
    except Exception as e:
        print(f"Error searching users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при поиске пользователей: {str(e)}"
        )

@app.get("/api/users/me", tags=["API"])
async def get_current_user_info(user: Dict = Depends(get_current_user)):
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ ПОЛУЧЕНИИ ИНФОРМАЦИИ О СЕБЕ
        update_user_activity(user['id'])
        
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute('''
                    SELECT movies_suggested, movies_selected, total_ratings_received
                    FROM user_statistic
                    WHERE user_id = %s
                ''', (user['id'],))
                
                stats = cur.fetchone()
                
                return {
                    "user": {
                        "id": user['id'],
                        "username": user['name'],
                        "email": user['email'],
                        "avatar": user.get('profile_picture') or "👤",
                        "overall_rating": float(user.get('overall_rating', 0)),
                        "registered_at": user.get('registered_at').isoformat() if user.get('registered_at') else datetime.now().isoformat()
                    },
                    "stats": {
                        "movies_suggested": stats['movies_suggested'] if stats else 0,
                        "movies_selected": stats['movies_selected'] if stats else 0,
                        "total_ratings_received": stats['total_ratings_received'] if stats else 0
                    }
                }
    except Exception as e:
        print(f"Error getting user info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении информации о пользователе: {str(e)}"
        )



@app.get("/api/servers/{server_id}", tags=["API"])
async def get_server_details(server_id: int, user: Dict = Depends(get_current_user)):
    """Получить детальную информацию о сервере"""
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ ПОЛУЧЕНИИ ДЕТАЛЕЙ СЕРВЕРА
        update_user_activity(user['id'])
        
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # Проверяем доступ
                cur.execute('''
                    SELECT 1 FROM cinema.room_participant 
                    WHERE room_id = %s AND user_id = %s AND is_active = TRUE
                ''', (server_id, user['id']))
                
                if not cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Вы не участник этого сервера"
                    )
                
                # Получаем информацию о сервере
                cur.execute('''
                    SELECT 
                        r.*,
                        u.name as owner_name,
                        u.profile_picture as owner_avatar
                    FROM cinema.room r
                    JOIN cinema."user" u ON r.owner_id = u.id
                    WHERE r.id = %s
                ''', (server_id,))
                
                server = cur.fetchone()
                
                if not server:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Сервер не найден"
                    )
                
                # Получаем участников
                participants = get_server_users(server_id)
                
                # Получаем фильмы для рулетки
                roulette_movies = get_roulette_movies(server_id)
                
                # Получаем последние сообщения
                cur.execute('''
                    SELECT 
                        rc.id,
                        rc.message as text,
                        u.name as user,
                        rc.sent_at as time,
                        CASE 
                            WHEN r.owner_id = u.id THEN TRUE
                            ELSE FALSE
                        END as is_admin
                    FROM cinema.room_chat rc
                    JOIN cinema."user" u ON rc.user_id = u.id
                    JOIN cinema.room r ON rc.room_id = r.id
                    WHERE rc.room_id = %s
                    ORDER BY rc.sent_at DESC
                    LIMIT 50
                ''', (server_id,))
                
                messages = cur.fetchall()
                
                # Получаем оценки
                cur.execute('''
                    SELECT 
                        sh.id,
                        m.title as name,
                        u.name as rated_by,
                        sh.watched_at as date,
                        sh.average_rating as rating
                    FROM cinema.session_history sh
                    JOIN cinema.movie m ON sh.movie_id = m.id
                    JOIN cinema."user" u ON sh.suggested_by_user_id = u.id
                    WHERE sh.room_id = %s
                    ORDER BY sh.watched_at DESC
                    LIMIT 20
                ''', (server_id,))
                
                ratings = cur.fetchall()
                
                return {
                    "id": server['id'],
                    "name": server['name'],
                    "icon": "🎬",  # Можно добавить поле для иконки в таблицу room
                    "admin": server['owner_name'],
                    "owner_id": server['owner_id'],
                    "status": server['status'],
                    "is_open": server['is_open'],
                    "created_at": server['created_at'].isoformat(),
                    "participants": participants,
                    "roulette_movies": roulette_movies,
                    "messages": [
                        {
                            "id": msg['id'],
                            "user": msg['user'],
                            "text": msg['text'],
                            "time": msg['time'].strftime("%H:%M"),
                            "isAdmin": msg['is_admin']
                        }
                        for msg in messages
                    ],
                    "rated_movies": [
                        {
                            "id": rating['id'],
                            "name": rating['name'],
                            "ratedBy": rating['rated_by'],
                            "date": rating['date'].strftime("%d.%m.%Y") if rating['date'] else "",
                            "rating": float(rating['rating']) if rating['rating'] else 0,
                            "comment": ""
                        }
                        for rating in ratings
                    ]
                }
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting server details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении данных сервера: {str(e)}"
        )

@app.post("/api/servers/{server_id}/roulette/suggest", tags=["API"])
async def suggest_movie_for_roulette_endpoint(
    server_id: int, 
    suggestion: MovieSuggestion,  # Используем Pydantic модель
    user: Dict = Depends(get_current_user)
):
    """Предложить фильм для рулетки - добавляет в таблицу in_roulette"""
    try:
        print(f"🎬 Начало обработки предложения фильма для сервера {server_id}")
        print(f"👤 Пользователь: {user.get('name')} (ID: {user.get('id')})")
        print(f"📦 Полученные данные (валидированные): {suggestion.dict()}")
        
        # Преобразуем Pydantic модель в dict
        suggestion_dict = suggestion.dict()
        
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ ПРЕДЛОЖЕНИИ ФИЛЬМА
        update_user_activity(user['id'])
        
        # Определяем movie_data
        movie_data = suggestion_dict.get('movie_data')
        if not movie_data:
            # Создаем movie_data из других полей
            movie_data = {}
            if suggestion_dict.get('title'):
                movie_data['title'] = suggestion_dict['title']
            elif suggestion_dict.get('name'):
                movie_data['title'] = suggestion_dict['name']
        
        movie_name = movie_data.get('title', '').strip()
        if not movie_name:
            print(f"❌ Нет названия фильма в данных: {movie_data}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Название фильма обязательно"
            )
        
        print(f"🎯 Название фильма: {movie_name}")
        
        # Проверяем, что пользователь является участником
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT 1 FROM cinema.room_participant 
                    WHERE room_id = %s AND user_id = %s AND is_active = TRUE
                ''', (server_id, user['id']))
                
                participant = cur.fetchone()
                if not participant:
                    print(f"❌ Пользователь {user['id']} не участник комнаты {server_id}")
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Вы не участник этого сервера"
                    )
                else:
                    print(f"✅ Пользователь {user['id']} является участником комнаты {server_id}")
        
        # Добавляем фильм в таблицу in_roulette
        print(f"📥 Добавляем фильм '{movie_name}' в in_roulette...")
        result = await add_movie_to_in_roulette(server_id, movie_data, user['id'])
        
        # Проверяем, добавился ли фильм
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT COUNT(*) FROM cinema.in_roulette 
                    WHERE room_id = %s
                ''', (server_id,))
                count_after = cur.fetchone()[0]
                print(f"✅ После добавления: {count_after} фильмов в рулетке для комнаты {server_id}")
        
        # Добавляем системное сообщение
        add_system_message(
            server_id, 
            f"🎬 {user['name']} предложил(а) фильм для рулетки: **{movie_name}**"
        )
        
        response = {
            "success": True,
            "message": f"Фильм '{movie_name}' добавлен в рулетку",
            "roulette_id": result['roulette_id'],
            "movie_id": result['movie_id'],
            "movie_title": result['movie_title']
        }
        
        print(f"✅ Успешно добавлен фильм. Ответ: {response}")
        return response
        
    except HTTPException as he:
        print(f"❌ HTTP Exception in suggest_movie_for_roulette_endpoint: {he.detail}")
        raise he
    except Exception as e:
        print(f"❌ Error suggesting movie: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при предложении фильма: {str(e)}"
        )


@app.get("/api/servers/{server_id}/roulette/movies", tags=["API"])
async def get_roulette_movies_endpoint(server_id: int, user: Dict = Depends(get_current_user)):
    """Получить фильмы для рулетки из таблицы in_roulette"""
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ ПОЛУЧЕНИИ ФИЛЬМОВ ДЛЯ РУЛЕТКИ
        update_user_activity(user['id'])
        
        movies = get_roulette_movies(server_id)
        return movies
    except Exception as e:
        print(f"Error getting roulette movies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении фильмов для рулетки: {str(e)}"
        )

@app.post("/api/servers/{server_id}/roulette/spin", tags=["API"])
async def spin_roulette_endpoint(server_id: int, user: Dict = Depends(get_current_user)):
    """Запустить рулетку"""
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ ЗАПУСКЕ РУЛЕТКИ
        update_user_activity(user['id'])
        
        # Проверяем, что пользователь является администратором
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT r.owner_id 
                    FROM cinema.room r
                    WHERE r.id = %s
                ''', (server_id,))
                
                room = cur.fetchone()
                if not room:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Сервер не найден"
                    )
                
                if room[0] != user['id']:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Только администратор может запустить рулетку"
                    )
        
        # Запускаем рулетку
        selected_movie = spin_roulette(server_id)
        
        if not selected_movie:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Нет фильмов для выбора в рулетке"
            )
        
        return {
            "success": True,
            "selected_movie": selected_movie,
            "message": f"Выбран фильм: {selected_movie['name']}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error spinning roulette: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при запуске рулетки: {str(e)}"
        )

@app.post("/api/servers/{server_id}/start", tags=["API"])
async def start_movie_event(server_id: int, user: Dict = Depends(get_current_user)):
    """Начать событие совместного просмотра"""
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ НАЧАЛЕ СОВМЕСТНОГО ПРОСМОТРА
        update_user_activity(user['id'])
        
        # Проверяем, что пользователь является администратором
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT r.owner_id 
                    FROM cinema.room r
                    WHERE r.id = %s
                ''', (server_id,))
                
                room = cur.fetchone()
                if not room:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Сервер не найден"
                    )
                
                if room[0] != user['id']:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Только администратор может начать совместный просмотр"
                    )
        
        # Обновляем статус комнаты
        update_server_status(server_id, "collecting")
        
        # Добавляем системное сообщение
        add_system_message(
            server_id,
            f"🎬 Администратор {user['name']} начал совместный просмотр! "
            f"Предлагайте фильмы для рулетки."
        )
        
        return {
            "success": True,
            "message": "Совместный просмотр начат",
            "status": "collecting"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error starting movie event: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при начале совместного просмотра: {str(e)}"
        )

@app.post("/api/servers/{server_id}/ready", tags=["API"])
async def confirm_ready(server_id: int, user: Dict = Depends(get_current_user)):
    """Подтвердить готовность к просмотру"""
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ ПОДТВЕРЖДЕНИИ ГОТОВНОСТИ
        update_user_activity(user['id'])
        
        # В будущем можно добавить таблицу для хранения статуса готовности
        # Пока просто отправляем сообщение в чат
        add_system_message(
            server_id,
            f"✅ {user['name']} готов к просмотру!"
        )
        
        return {
            "success": True,
            "message": "Готовность подтверждена"
        }
        
    except Exception as e:
        print(f"Error confirming ready: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при подтверждении готовности: {str(e)}"
        )

# ============== ОБНОВЛЕННЫЙ ЭНДПОИНТ ПОИСКА ФИЛЬМОВ ==============

@app.get("/api/movies/search", tags=["API"])
async def search_movies(query: str):
    """Поиск фильмов в Kinopoisk"""
    try:
        # Если запрос пустой, возвращаем пустой список
        if not query or len(query.strip()) < 2:
            return []
        
        query = query.strip()
        print(f"🔍 Поиск фильмов: '{query}'")
        
        # Ищем фильмы в локальной базе данных
        local_movies = []
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute('''
                    SELECT 
                        id,
                        title,
                        description,
                        duration,
                        release_year,
                        poster_url,
                        rating_kp,
                        votes_kp
                    FROM cinema.movie 
                    WHERE LOWER(title) LIKE LOWER(%s)
                    ORDER BY rating_kp DESC NULLS LAST
                    LIMIT 10
                ''', (f"%{query}%",))
                
                local_movies = cur.fetchall()
        
        # Если есть локальные результаты, преобразуем их и возвращаем
        if local_movies:
            print(f"✅ Найдено локально: {len(local_movies)} фильмов")
            return [dict(movie) for movie in local_movies]
        
        # Если нет локальных результатов, ищем через Kinopoisk API
        print(f"🎬 Ищем в Kinopoisk API: '{query}'")
        
        # Используем вашу функцию поиска
        movie_info = await search_movie_by_keyword(query)  
        
        print(f"✅ Результат от search_movie_by_keyword:")
        print(f"   Тип: {type(movie_info)}")
        print(f"   Данные: {movie_info}")
        
        if not movie_info:
            print("⚠️ Фильм не найден через Kinopoisk API")
            # Возвращаем пустой список
            return []
        
        # Преобразуем Movie_Info объект в словарь
        result = []
        
        # Проверяем, является ли результат списком
        if isinstance(movie_info, list):
            for movie in movie_info:
                # Если это объект Movie_Info, преобразуем в словарь
                if hasattr(movie, '__dict__'):
                    movie_dict = movie.__dict__
                    # Убираем приватные атрибуты если есть
                    movie_dict = {k: v for k, v in movie_dict.items() if not k.startswith('_')}
                    result.append(movie_dict)
                elif isinstance(movie, dict):
                    result.append(movie)
        # Если это одиночный объект Movie_Info
        elif hasattr(movie_info, '__dict__'):
            movie_dict = movie_info.__dict__
            # Убираем приватные атрибуты если есть
            movie_dict = {k: v for k, v in movie_dict.items() if not k.startswith('_')}
            result.append(movie_dict)
        # Если это словарь
        elif isinstance(movie_info, dict):
            result.append(movie_info)
        else:
            # Если непонятный формат, пытаемся преобразовать
            print(f"⚠️ Неизвестный формат данных: {type(movie_info)}")
            result = [movie_info] if movie_info else []
        
        print(f"✅ Возвращаем {len(result)} фильмов")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ошибка при поиске фильмов: {e}")
        import traceback
        traceback.print_exc()
        # Возвращаем тестовые данные в случае ошибки
        return [
            {
                "id": 1,
                "title": f"{query} (пример)",
                "description": "Пример описания фильма (ошибка поиска)",
                "duration": 120,
                "release_year": 2024,
                "poster_url": "https://via.placeholder.com/300x450?text=Error",
                "rating_kp": 7.5,
                "votes_kp": 1000,
                "genres": ["Драма", "Комедия"]
            }
        ]


# ============== ДОБАВЛЯЕМ НОВЫЕ ЭНДПОИНТЫ ДЛЯ РАБОТЫ С ACCESS CODE ==============

@app.get("/api/servers/{server_id}/access-code", tags=["API"])
async def get_server_access_code(server_id: int, user: Dict = Depends(get_current_user)):
    """Получить access code сервера (только для владельца)"""
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ ПОЛУЧЕНИИ ACCESS CODE
        update_user_activity(user['id'])
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Проверяем, что пользователь является владельцем
                cur.execute('''
                    SELECT owner_id, access_code, name, is_open
                    FROM cinema.room 
                    WHERE id = %s
                ''', (server_id,))
                
                room = cur.fetchone()
                
                if not room:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Сервер не найден"
                    )
                
                owner_id, access_code, room_name, is_open = room
                
                if owner_id != user['id']:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Только владелец может получить access code"
                    )
                
                # Если сервер закрыт, создаем новый код
                if not is_open:
                    new_code = secrets.token_hex(3).upper()
                    cur.execute('''
                        UPDATE cinema.room 
                        SET is_open = TRUE, access_code = %s
                        WHERE id = %s
                    ''', (new_code, server_id))
                    access_code = new_code
                    conn.commit()
                
                return {
                    "success": True,
                    "server_id": server_id,
                    "server_name": room_name,
                    "access_code": access_code,
                    "invite_link": f"http://localhost:3000/join/{access_code}",
                    "is_open": is_open
                }
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting access code: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении access code: {str(e)}"
        )

@app.post("/api/servers/{server_id}/regenerate-code", tags=["API"])
async def regenerate_access_code(server_id: int, user: Dict = Depends(get_current_user)):
    """Сгенерировать новый access code"""
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ РЕГЕНЕРАЦИИ КОДА
        update_user_activity(user['id'])
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Проверяем, что пользователь является владельцем
                cur.execute('''
                    SELECT owner_id, name 
                    FROM cinema.room 
                    WHERE id = %s
                ''', (server_id,))
                
                room = cur.fetchone()
                
                if not room:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Сервер не найден"
                    )
                
                owner_id, room_name = room
                
                if owner_id != user['id']:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Только владелец может сгенерировать новый код"
                    )
                
                # Генерируем новый код
                new_code = secrets.token_hex(3).upper()
                
                cur.execute('''
                    UPDATE cinema.room 
                    SET access_code = %s, is_open = TRUE
                    WHERE id = %s
                ''', (new_code, server_id))
                
                conn.commit()
                
                # Добавляем системное сообщение
                add_system_message(
                    server_id,
                    f"🔑 Администратор сгенерировал новый код доступа для сервера"
                )
                
                return {
                    "success": True,
                    "server_id": server_id,
                    "server_name": room_name,
                    "new_access_code": new_code,
                    "invite_link": f"http://localhost:3000/join/{new_code}",
                    "message": "Код доступа успешно обновлен"
                }
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error regenerating access code: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при генерации нового кода: {str(e)}"
        )

@app.post("/api/servers/join", tags=["API"])
async def join_server_by_code(join_data: Dict, user: Dict = Depends(get_current_user)):
    """Присоединиться к серверу по access code"""
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ ПРИСОЕДИНЕНИИ К СЕРВЕРУ
        update_user_activity(user['id'])
        
        access_code = join_data.get('access_code', '').strip().upper()
        
        if not access_code or len(access_code) < 4:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Неверный код доступа"
            )
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Находим сервер по коду
                cur.execute('''
                    SELECT id, name, owner_id, is_open, status
                    FROM cinema.room 
                    WHERE access_code = %s
                ''', (access_code,))
                
                room = cur.fetchone()
                
                if not room:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Сервер не найден или код неверный"
                    )
                
                room_id, room_name, owner_id, is_open, room_status = room
                
                # Проверяем, открыт ли сервер
                if not is_open:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Сервер закрыт для новых участников"
                    )
                
                # Проверяем, не присоединен ли уже пользователь
                cur.execute('''
                    SELECT id, is_active 
                    FROM cinema.room_participant 
                    WHERE room_id = %s AND user_id = %s
                ''', (room_id, user['id']))
                
                existing_participation = cur.fetchone()
                
                if existing_participation:
                    participant_id, is_active = existing_participation
                    
                    if is_active:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Вы уже являетесь участником этого сервера"
                        )
                    else:
                        # Пользователь был в сервере ранее - активируем
                        cur.execute('''
                            UPDATE cinema.room_participant 
                            SET is_active = TRUE, left_at = NULL
                            WHERE id = %s
                        ''', (participant_id,))
                        message = "Вы вернулись в сервер"
                else:
                    # Добавляем нового участника
                    cur.execute('''
                        INSERT INTO cinema.room_participant 
                        (room_id, user_id, role, is_active)
                        VALUES (%s, %s, 'member', TRUE)
                    ''', (room_id, user['id']))
                    message = "Вы успешно присоединились к серверу"
                
                # Добавляем приветственное сообщение
                cur.execute('''
                    INSERT INTO cinema.room_chat 
                    (room_id, user_id, message, message_type)
                    VALUES (%s, %s, %s, 'system')
                ''', (room_id, user['id'], f"👋 {user['name']} присоединился(ась) к серверу!"))
                
                # Обновляем статистику пользователя
                cur.execute('''
                    UPDATE cinema.user_statistic 
                    SET last_activity = CURRENT_TIMESTAMP
                    WHERE user_id = %s
                ''', (user['id'],))
                
                conn.commit()
                
                # Получаем информацию о сервере для ответа
                cur.execute('''
                    SELECT 
                        r.id,
                        r.name,
                        r.status,
                        r.created_at,
                        u.name as owner_name,
                        u.profile_picture as owner_avatar
                    FROM cinema.room r
                    JOIN cinema."user" u ON r.owner_id = u.id
                    WHERE r.id = %s
                ''', (room_id,))
                
                server_info = cur.fetchone()
                
                # Получаем количество участников
                cur.execute('''
                    SELECT COUNT(*) as participants_count
                    FROM cinema.room_participant 
                    WHERE room_id = %s AND is_active = TRUE
                ''', (room_id,))
                
                participants_count = cur.fetchone()[0]
                
                return {
                    "success": True,
                    "message": message,
                    "server": {
                        "id": server_info[0],
                        "name": server_info[1],
                        "status": server_info[2],
                        "created_at": server_info[3].isoformat(),
                        "admin": server_info[4],
                        "icon": "🎬",
                        "users": participants_count,
                        "is_open": is_open,
                        "is_admin": (owner_id == user['id'])
                    }
                }
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error joining server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при присоединении к серверу: {str(e)}"
        )

@app.post("/api/servers/{server_id}/toggle-access", tags=["API"])
async def toggle_server_access(server_id: int, user: Dict = Depends(get_current_user)):
    """Открыть/закрыть доступ к серверу"""
    try:
        # ОБНОВЛЯЕМ АКТИВНОСТЬ ПРИ ИЗМЕНЕНИИ ДОСТУПА К СЕРВЕРУ
        update_user_activity(user['id'])
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Проверяем, что пользователь является владельцем
                cur.execute('''
                    SELECT owner_id, is_open, name 
                    FROM cinema.room 
                    WHERE id = %s
                ''', (server_id,))
                
                room = cur.fetchone()
                
                if not room:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Сервер не найден"
                    )
                
                owner_id, is_open, room_name = room
                
                if owner_id != user['id']:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Только владелец может изменять доступ к серверу"
                    )
                
                # Меняем статус
                new_status = not is_open
                
                cur.execute('''
                    UPDATE cinema.room 
                    SET is_open = %s
                    WHERE id = %s
                ''', (new_status, server_id))
                
                conn.commit()
                
                # Добавляем системное сообщение
                status_text = "открыл" if new_status else "закрыл"
                add_system_message(
                    server_id,
                    f"🔒 Администратор {status_text} доступ к серверу для новых участников"
                )
                
                return {
                    "success": True,
                    "server_id": server_id,
                    "server_name": room_name,
                    "is_open": new_status,
                    "message": f"Сервер {'открыт' if new_status else 'закрыт'} для новых участников"
                }
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error toggling server access: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при изменении доступа к серверу: {str(e)}"
        )

@app.get("/api/servers/invite/{access_code}", tags=["API"])
async def get_server_by_access_code(access_code: str):
    """Получить информацию о сервере по access code (без авторизации)"""
    try:
        access_code = access_code.strip().upper()
        
        if not access_code or len(access_code) < 4:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Неверный код доступа"
            )
        
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute('''
                    SELECT 
                        r.id,
                        r.name,
                        r.status,
                        r.created_at,
                        r.is_open,
                        u.name as owner_name,
                        u.profile_picture as owner_avatar,
                        COUNT(DISTINCT rp.id) as participants_count
                    FROM cinema.room r
                    JOIN cinema."user" u ON r.owner_id = u.id
                    LEFT JOIN cinema.room_participant rp ON r.id = rp.room_id AND rp.is_active = TRUE
                    WHERE r.access_code = %s
                    GROUP BY r.id, u.name, u.profile_picture
                ''', (access_code,))
                
                room = cur.fetchone()
                
                if not room:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Сервер не найден"
                    )
                
                if not room['is_open']:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Сервер закрыт для новых участников"
                    )
                
                return {
                    "success": True,
                    "server": {
                        "id": room['id'],
                        "name": room['name'],
                        "status": room['status'],
                        "created_at": room['created_at'].isoformat(),
                        "admin": room['owner_name'],
                        "admin_avatar": room['owner_avatar'] or "👑",
                        "participants_count": room['participants_count'],
                        "icon": "🎬",
                        "is_open": room['is_open']
                    }
                }
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting server by access code: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении информации о сервере: {str(e)}"
        )

# ================= ДОПОЛНИТЕЛЬНЫЕ ЭНДПОИНТЫ =================

@app.post('/auth/login')
def login_old(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        user = authenticate_user(form_data.username, form_data.password)
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user['email']}, expires_delta=access_token_expires
        )
        return {'access_token': access_token, 'token_type': 'bearer'}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка сервера: {str(e)}"
        )

@app.get('/jwttest')
def jwttest(user: Dict = Depends(get_current_user)):
    return {'message': 'ok', 'user': user['email'], 'user_id': user['id']}

@app.get('/ping')
def ping():
    return {'message': 'pong'}

# Тестовый эндпоинт для проверки регистрации
@app.post("/api/test/create-user")
async def create_test_user():
    """Создание тестового пользователя"""
    try:
        test_email = "test@test.com"
        test_password = "test123"
        test_username = "TestUser"
        
        # Проверяем существование
        existing = get_user_by_email(test_email)
        if existing:
            return {
                "success": True,
                "message": "Пользователь уже существует",
                "user_id": existing['id']
            }
        
        # Регистрируем
        hashed = get_hashed_password(test_password)
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'INSERT INTO "user" (name, email, password_hash) VALUES (%s, %s, %s) RETURNING id',
                    (test_username, test_email, psycopg2.Binary(hashed))
                )
                user_id = cur.fetchone()[0]
                conn.commit()
        
        return {
            "success": True,
            "message": "Тестовый пользователь создан",
            "user_id": user_id,
            "credentials": {
                "email": test_email,
                "password": test_password,
                "username": test_username
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/db-check", tags=["API"])
async def check_database_connection():
    """Проверка подключения к базе данных"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Простой запрос для проверки
                cur.execute("SELECT 1 as connection_test")
                result = cur.fetchone()
                
                # Также можно проверить существование таблиц
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'cinema'
                    ) as schema_exists
                """)
                schema_check = cur.fetchone()
                
        return {
            "status": "connected",
            "database": "postgres",
            "schema": "cinema",
            "connection_test": result[0] if result else None,
            "schema_exists": schema_check[0] if schema_check else False,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return {
            "status": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }



# ================= НОВЫЕ ЭНДПОИНТЫ ДЛЯ ПРОСТОЙ РЕАЛИЗАЦИИ ОНЛАЙН СТАТУСОВ =================

@app.get("/api/users/{user_id}/online-status", tags=["Online Status"])
async def get_user_online_status(user_id: int):
    """Получить онлайн статус пользователя (простая реализация)"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute('''
                    SELECT 
                        u.id,
                        u.name as username,
                        u.profile_picture as avatar,
                        u.registered_at,
                        us.last_activity,
                        COALESCE(
                            (SELECT COUNT(*) 
                             FROM cinema.room_participant rp 
                             WHERE rp.user_id = u.id AND rp.is_active = TRUE),
                            0
                        ) as active_rooms_count
                    FROM cinema.user u
                    LEFT JOIN cinema.user_statistic us ON u.id = us.user_id
                    WHERE u.id = %s
                ''', (user_id,))
                
                user_data = cur.fetchone()
                
                if not user_data:
                    return {
                        "success": False,
                        "error": "Пользователь не найден"
                    }
                
                # Определяем статус
                now = datetime.now()
                last_activity = user_data['last_activity']
                
                status_info = {
                    "user_id": user_data['id'],
                    "username": user_data['username'],
                    "avatar": user_data['avatar'] or "👤",
                    "active_rooms_count": user_data['active_rooms_count'],
                    "registered_at": user_data['registered_at'].isoformat() if user_data['registered_at'] else None
                }
                
                if not last_activity:
                    status_info.update({
                        "is_online": False,
                        "status": "never_active",
                        "last_seen": None,
                        "time_ago": "никогда"
                    })
                else:
                    time_diff = now - last_activity
                    
                    # Определяем статус на основе времени бездействия
                    # Онлайн = активен в последние 10 минут
                    if time_diff < timedelta(minutes=10):
                        status_info.update({
                            "is_online": True,
                            "status": "online",
                            "last_seen": last_activity.isoformat(),
                            "time_ago": _format_time_ago(time_diff)
                        })
                    elif time_diff < timedelta(minutes=30):
                        status_info.update({
                            "is_online": False,
                            "status": "recently_online",
                            "last_seen": last_activity.isoformat(),
                            "time_ago": _format_time_ago(time_diff)
                        })
                    elif time_diff < timedelta(hours=1):
                        status_info.update({
                            "is_online": False,
                            "status": "away",
                            "last_seen": last_activity.isoformat(),
                            "time_ago": _format_time_ago(time_diff)
                        })
                    elif time_diff < timedelta(days=1):
                        status_info.update({
                            "is_online": False,
                            "status": "offline_today",
                            "last_seen": last_activity.isoformat(),
                            "time_ago": _format_time_ago(time_diff)
                        })
                    else:
                        status_info.update({
                            "is_online": False,
                            "status": "offline",
                            "last_seen": last_activity.isoformat(),
                            "time_ago": _format_time_ago(time_diff)
                        })
                
                return {
                    "success": True,
                    **status_info,
                    "checked_at": now.isoformat()
                }
                
    except Exception as e:
        print(f"❌ Error getting user online status: {e}")
        return {
            "success": False,
            "user_id": user_id,
            "is_online": False,
            "status": "error",
            "error": str(e)
        }

@app.get("/api/rooms/{room_id}/online-users", tags=["Online Status"])
async def get_online_users_in_room(room_id: int, user: Dict = Depends(get_current_user)):
    """Получить онлайн пользователей в комнате (простая реализация)"""
    try:
        # Проверяем, что пользователь в комнате
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT 1 FROM cinema.room_participant 
                    WHERE room_id = %s AND user_id = %s AND is_active = TRUE
                ''', (room_id, user['id']))
                
                if not cur.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Вы не участник этой комнаты"
                    )
        
        # Получаем всех участников комнаты
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute('''
                    SELECT 
                        u.id,
                        u.name as username,
                        u.profile_picture as avatar,
                        u.registered_at,
                        us.last_activity,
                        CASE 
                            WHEN r.owner_id = u.id THEN 'owner'
                            WHEN rp.role = 'moderator' THEN 'moderator'
                            ELSE 'member'
                        END as role,
                        rp.joined_at,
                        rp.role as db_role
                    FROM cinema.room_participant rp
                    JOIN cinema."user" u ON rp.user_id = u.id
                    JOIN cinema.room r ON rp.room_id = r.id
                    LEFT JOIN cinema.user_statistic us ON u.id = us.user_id
                    WHERE rp.room_id = %s AND rp.is_active = TRUE
                    ORDER BY 
                        CASE 
                            WHEN r.owner_id = u.id THEN 1
                            WHEN rp.role = 'moderator' THEN 2
                            ELSE 3
                        END,
                        u.name
                ''', (room_id,))
                
                participants = cur.fetchall()
                now = datetime.now()
                
                online_users = []
                all_users = []
                
                for participant in participants:
                    user_info = {
                        "user_id": participant['id'],
                        "username": participant['username'],
                        "avatar": participant['avatar'] or "👤",
                        "role": participant['role'],
                        "db_role": participant['db_role'],
                        "joined_at": participant['joined_at'].isoformat() if participant['joined_at'] else None,
                        "registered_at": participant['registered_at'].isoformat() if participant['registered_at'] else None
                    }
                    
                    last_activity = participant['last_activity']
                    is_online = False
                    
                    if last_activity:
                        time_diff = now - last_activity
                        # Онлайн если активен в последние 10 минут
                        if time_diff < timedelta(minutes=10):
                            is_online = True
                        
                        user_info.update({
                            "is_online": is_online,
                            "last_activity": last_activity.isoformat(),
                            "time_ago": _format_time_ago(time_diff)
                        })
                    else:
                        user_info.update({
                            "is_online": False,
                            "last_activity": None,
                            "time_ago": "никогда"
                        })
                    
                    all_users.append(user_info)
                    if is_online:
                        online_users.append(user_info)
                
                return {
                    "success": True,
                    "room_id": room_id,
                    "online_users": online_users,
                    "all_users": all_users,
                    "online_count": len(online_users),
                    "total_count": len(all_users),
                    "checked_at": now.isoformat()
                }
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting online users in room: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении онлайн пользователей: {str(e)}"
        )

@app.post("/api/users/batch-online-status", tags=["Online Status"])
async def get_batch_online_status(user_ids: List[int] = Body(...)):
    """Получить онлайн статусы нескольких пользователей за раз"""
    try:
        if not user_ids:
            return {
                "success": True,
                "statuses": {},
                "count": 0
            }
        
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # Используем ANY для получения всех пользователей за один запрос
                cur.execute('''
                    SELECT 
                        u.id,
                        u.name as username,
                        u.profile_picture as avatar,
                        u.registered_at,
                        us.last_activity
                    FROM cinema.user u
                    LEFT JOIN cinema.user_statistic us ON u.id = us.user_id
                    WHERE u.id = ANY(%s)
                ''', (user_ids,))
                
                users = cur.fetchall()
                now = datetime.now()
                
                statuses = {}
                for user in users:
                    last_activity = user['last_activity']
                    is_online = False
                    
                    if last_activity:
                        time_diff = now - last_activity
                        # Онлайн если активен в последние 10 минут
                        is_online = time_diff < timedelta(minutes=10)
                    
                    statuses[user['id']] = {
                        "user_id": user['id'],
                        "username": user['username'],
                        "avatar": user['avatar'] or "👤",
                        "is_online": is_online,
                        "last_activity": last_activity.isoformat() if last_activity else None,
                        "registered_at": user['registered_at'].isoformat() if user['registered_at'] else None,
                        "time_ago": _format_time_ago(time_diff) if last_activity else "никогда"
                    }
                
                # Добавляем отсутствующих пользователей
                for user_id in user_ids:
                    if user_id not in statuses:
                        statuses[user_id] = {
                            "user_id": user_id,
                            "is_online": False,
                            "status": "not_found",
                            "error": "Пользователь не найден"
                        }
                
                return {
                    "success": True,
                    "statuses": statuses,
                    "count": len(statuses)
                }
                
    except Exception as e:
        print(f"❌ Error getting batch online status: {e}")
        return {
            "success": False,
            "error": str(e),
            "statuses": {}
        }

@app.post("/api/users/{user_id}/update-activity", tags=["Online Status"])
async def manual_update_activity(user_id: int, user: Dict = Depends(get_current_user)):
    """Вручную обновить активность пользователя (для heartbeat)"""
    if user['id'] != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Нельзя обновлять активность другого пользователя"
        )
    
    update_user_activity(user_id)
    return {
        "success": True,
        "message": "Активность обновлена",
        "timestamp": datetime.now().isoformat()
    }

# Middleware для автоматического обновления активности
@app.middleware("http")
async def update_activity_middleware(request: Request, call_next):
    """Middleware для обновления активности пользователя при каждом запросе"""
    response = await call_next(request)
    
    # Обновляем активность, если пользователь авторизован
    try:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                email = payload.get("sub")
                if email:
                    user = get_user_by_email(email)
                    if user:
                        update_user_activity(user['id'])
            except:
                pass
    except Exception as e:
        # Игнорируем ошибки в middleware
        pass
    
    return response


if __name__ == '__main__':
    print("🚀 Запуск MovieRatings API...")
    print(f"📊 Подключение к базе данных: {db_conn_dict['host']}:{db_conn_dict['port']}")
    print(f"🔐 Секретный ключ: {'*' * 20}")
    print(f"⏰ Время жизни токена: {ACCESS_TOKEN_EXPIRE_MINUTES} минут")
    print("🎯 Онлайн статусы: активен при активности за последние 10 минут")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)