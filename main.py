from enum import StrEnum, auto
from datetime import timedelta, datetime, timezone

import bcrypt
import jwt
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
import psycopg2
import uvicorn


SECRET_KEY = "agjohuyh59i2yiq3y9iuqy34iguyaiugy349ty29h"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
db_conn_dict = {
    'database': 'b7zrksdmhg8hfc2f6lc4',
    'user': 'uxkffrkpeuv2ohz2xq6c',
    'password': 'UbvB9wN01fNnoPsd9Orkon2ZAA3rnq',
    'host': 'b7zrksdmhg8hfc2f6lc4-postgresql.services.clever-cloud.com',
    'port': 50013,
    'options': "-c search_path=cinema,public,user"
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
app = FastAPI()

class UserBase(BaseModel):
    email: EmailStr
    name: str

class UserSignUp(UserBase):
    password: str

class UserRead(UserBase):
    id: int

class User(UserRead):
    hashed_password: str


class Room(BaseModel):
    id: int
    owner_id: int
    name: str
    status: str
    created_at: datetime
    participants: list[UserRead] = Field(default_factory=list)

class RoomStatus(StrEnum):
    CHOOSING = auto()
    WATCHING = auto()
    SLEEPING = auto()

def get_hashed_password(plain_text_password):
    return bcrypt.hashpw(plain_text_password, bcrypt.gensalt())

def check_password(plain_text_password, hashed_password):
    return bcrypt.checkpw(plain_text_password, hashed_password)

def get_user(email) -> User | None:
    with psycopg2.connect(**db_conn_dict) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'select * from "User" where "Email" = %s',
                (email, )
            )
            data = cur.fetchone()
            if not data:
                return None
            return User(
                id=data[0],
                name=data[1],
                email=data[2],
                hashed_password=data[3],
            )

def authenticate_user(email: str, password: str):
    user = get_user(email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Неправильный логин или пароль'
        )
    if not check_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Неправильный логин или пароль'
        )
    return user

def get_current_user(token = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    email = payload.get("sub")
    if email is None:
        raise credentials_exception
    user = get_user(email)
    if user is None:
        raise credentials_exception
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt



@app.get('/ping')
def ping():
    return 'pong'
    with psycopg2.connect(**db_conn_dict) as conn:
        with conn.cursor() as cur:
            cur.execute('truncate table "User", "Room", "Room_Participants" Cascade')


@app.post('/auth/signup')
def signup(
    user_schema: UserSignUp
):
    hashed_password = get_hashed_password(user_schema.password)
    with psycopg2.connect(**db_conn_dict) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'insert into "User" ("Name", "Email", "PasswordHash") values (%s, %s, %s)',
                (user_schema.name, user_schema.email, hashed_password)
            )

@app.post('/auth/login')
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    user = authenticate_user(form_data.username, form_data.password)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {'access_token': access_token}

@app.get('/jwttest')
def jwttest(
    user: User = Depends(get_current_user)
):
    return 'ok'

@app.post('/room')
def create_room(
    name: str,
    user: User = Depends(get_current_user),
):
    with psycopg2.connect(**db_conn_dict) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'insert into "Room" ("Name", "OwnerID") values (%s, %s) returning "RoomID"',
                (name, user.id)
            )
            room_id = cur.fetchone()
            if not room_id or len(room_id) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Комната не существует",
                )
                
            room_id = room_id[0]
            access_code = str(hash(room_id + 85947018))[2:9]  # TODO change hash function
            cur.execute(
                'update "Room" set "Access_code" = %s where "Room"."RoomID" = %s',
                (access_code, room_id)
            )
            cur.execute(
                'insert into "Room_Participants" ("RoomID", "UserID") values (%s, %s)',
                (room_id, user.id)
            )


@app.get('/room')
def get_rooms(
    user: User = Depends(get_current_user),
):
    with psycopg2.connect(**db_conn_dict) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            select "Room"."RoomID", "Room"."Name", "Room"."OwnerID", "Room"."Status", "Room"."Created_at", "User"."UserId", "User"."Name", "User"."Email" from "Room"
            join "Room_Participants" on "Room_Participants"."RoomID" = "Room"."RoomID"
            join "User" on "User"."UserId" = "Room_Participants"."UserID"
            """)
            from pprint import pprint
            rooms = cur.fetchall()
            pprint(rooms)
            result = {}
            for room in rooms:
                if result.get(room[0]):
                    result[room[0]].participants.append(UserRead(
                        id = room[5],
                        name = room[6],
                        email = room[7]
                    ))
                else:
                    result[room[0]] = Room(
                        id=room[0],
                        name=room[1],
                        owner_id=room[2],
                        status=room[3],
                        created_at=room[4],
                    )
                    result[room[0]].participants.append(UserRead(
                        id = room[5],
                        name = room[6],
                        email = room[7]
                    ))
            return result

@app.get('/room/participant')
def get_members(
    room_id: int,
    user: User = Depends(get_current_user),
):
    with psycopg2.connect(**db_conn_dict) as conn:
        with conn.cursor() as cur:
            cur.execute('select * from "Room" where "Room"."RoomID" = %s', (room_id, ))
            room = cur.fetchone()
            if room is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Комната не существует",
                )
            cur.execute('select * from "User" join "Room_Participants" on "User"."UserId" = "Room_Participants"."UserID" where "Room_Participants"."RoomID" = %s ', (room_id, ))
            users = cur.fetchall()
            result = []
            for user in users:
                result.append(UserRead(
                    id=user[0],
                    name=user[1],
                    email=user[2]
                ))
            return result

@app.get("/room/access_code")
def get_room_access_code(
    room_id: int,
    user: User = Depends(get_current_user),
):
    with psycopg2.connect(**db_conn_dict) as conn:
        with conn.cursor() as cur:
            cur.execute('select * from "Room" where "Room"."RoomID" = %s', (room_id, ))
            room = cur.fetchone()
            if room is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Комната не существует",
                )
            owner_id = room[2]
            if user.id != owner_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Отказано в доступе",
                )
            return room[11]


@app.get("/room/join/{access_code}")
def join_room_via_link(
    access_code: str,
    user: User = Depends(get_current_user),
):
    with psycopg2.connect(**db_conn_dict) as conn:
        with conn.cursor() as cur:
            cur.execute('select * from "Room" where "Room"."Access_code" = %s', (access_code, ))
            room = cur.fetchone()
            if room is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Комната не существует",
                )
            room_id = room[0]
            cur.execute(
                'select "UserID" from "Room_Participants" where "RoomID" = %s',
                (room_id, )
            )
            if cur.fetchone:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Уже в комнате",
                )

            cur.execute(
                'insert into "Room_Participants" ("RoomID", "UserID") values (%s, %s)',
                (room_id, user.id)
            )

if  __name__ == '__main__':
    uvicorn.run(app)
