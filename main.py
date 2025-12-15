from enum import StrEnum, auto
from datetime import timedelta, datetime, timezone
from typing import Optional, List
import bcrypt
import jwt
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, validator
import psycopg2
import uvicorn
import hashlib
import secrets
import time

SECRET_KEY = ""
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

db_conn_dict = {
    'database': '',
    'user': '',
    'password': '',
    'host': 'localhost',
    'port': 5432,
    'options': "-c search_path=cinema"
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
app = FastAPI()

# Pydantic модели
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

# Вспомогательные функции
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

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)