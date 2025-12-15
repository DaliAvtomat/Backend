import httpx
from typing import Optional, Dict, Any, List
import asyncio
import json

class KinopoiskAPI:
    def __init__(self):
        self.base_url = "https://kinopoiskapiunofficial.tech/api/v2.2"
        self.headers = {
            "X-API-KEY": "",
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        self.client = httpx.AsyncClient(timeout=30.0) 
    
    async def close(self):
        await self.client.aclose()
    
    async def search_movie_by_keyword(self, keyword: str) -> Optional[Dict[str, Any]]:
        """Поиск фильма по ключевому слову"""
        try:
            url = f"{self.base_url}/films"
            params = {
                "keyword": keyword,
                "page": 1
            }
            
            response = await self.client.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("items") and len(data["items"]) > 0:
                    # Берем первый результат
                    film = data["items"][0]
                    # Получаем детальную информацию
                    detailed_info = await self.get_movie_by_id(film["kinopoiskId"])
                    if detailed_info:
                        return detailed_info
                    else:
                        # Если детальная информация недоступна, используем базовую
                        return self._format_movie_info(film)
            elif response.status_code == 404:
                print(f"Фильм '{keyword}' не найден")
            else:
                print(f"Ошибка API: {response.status_code} - {response.text}")
                
        except httpx.RequestError as e:
            print(f"Ошибка подключения: {e}")
        except Exception as e:
            print(f"Ошибка при поиске фильма: {e}")
        
        return None
    
    async def get_movie_by_id(self, film_id: int) -> Optional[Dict[str, Any]]:
        """Получение детальной информации о фильме по ID"""
        try:
            url = f"{self.base_url}/films/{film_id}"
            
            response = await self.client.get(url, headers=self.headers)
            
            if response.status_code == 200:
                film_data = response.json()
                return self._format_movie_info(film_data)
            else:
                print(f"Ошибка получения фильма {film_id}: {response.status_code}")
                
        except httpx.RequestError as e:
            print(f"Ошибка подключения: {e}")
        except Exception as e:
            print(f"Ошибка при получении фильма: {e}")
        
        return None
    
    def _format_movie_info(self, film_data: Dict[str, Any]) -> Dict[str, Any]:
        """Форматирует данные фильма в удобный вид"""
        return {
            "id": film_data.get("kinopoiskId") or film_data.get("filmId"),
            "title": film_data.get("nameRu") or film_data.get("nameOriginal") or "Без названия",
            "original_title": film_data.get("nameOriginal"),
            "description": film_data.get("description") or film_data.get("shortDescription") or "",
            "duration": film_data.get("filmLength"),
            "release_year": film_data.get("year"),
            "poster_url": film_data.get("posterUrl") or film_data.get("posterUrlPreview"),
            "rating_kp": film_data.get("ratingKinopoisk"),
            "rating_imdb": film_data.get("ratingImdb"),
            "votes_kp": film_data.get("ratingKinopoiskVoteCount"),
            "votes_imdb": film_data.get("ratingImdbVoteCount"),
            "genres": [genre["genre"] for genre in film_data.get("genres", [])],
            "countries": [country["country"] for country in film_data.get("countries", [])],
            "type": film_data.get("type"),
            "age_rating": film_data.get("ratingAgeLimits"),
            "serial": film_data.get("serial", False),
            "short_film": film_data.get("shortFilm", False)
        }
    
    async def get_film_sequels_and_prequels(self, film_id: int) -> List[Dict[str, Any]]:
        """Получение сиквелов и приквелов фильма"""
        try:
            url = f"{self.base_url}/films/{film_id}/sequels_and_prequels"
            
            response = await self.client.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Ошибка при получении сиквелов: {e}")
        
        return []
    
    async def get_film_staff(self, film_id: int) -> List[Dict[str, Any]]:
        """Получение актеров и съемочной группы"""
        try:
            url = f"{self.base_url}/films/{film_id}/staff"
            
            response = await self.client.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Ошибка при получении актеров: {e}")
        
        return []
    
    async def get_film_videos(self, film_id: int) -> Dict[str, Any]:
        """Получение видео (трейлеры, клипы)"""
        try:
            url = f"{self.base_url}/films/{film_id}/videos"
            
            response = await self.client.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Ошибка при получении видео: {e}")
        
        return {"items": []}
    
    async def search_by_filters(
        self,
        countries: Optional[List[int]] = None,
        genres: Optional[List[int]] = None,
        order: str = "RATING",
        type: str = "FILM",
        rating_from: int = 0,
        rating_to: int = 10,
        year_from: int = 1888,
        year_to: Optional[int] = None,
        page: int = 1
    ) -> Optional[Dict[str, Any]]:
        """Поиск фильмов по фильтрам"""
        try:
            url = f"{self.base_url}/films"
            params = {
                "order": order,
                "type": type,
                "ratingFrom": rating_from,
                "ratingTo": rating_to,
                "yearFrom": year_from,
                "page": page
            }
            
            if year_to:
                params["yearTo"] = year_to
            
            if countries:
                params["countries"] = ",".join(map(str, countries))
            
            if genres:
                params["genres"] = ",".join(map(str, genres))
            
            response = await self.client.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            print(f"Ошибка при поиске по фильтрам: {e}")
        
        return None

# Создаем экземпляр API клиента
kinopoisk_api = KinopoiskAPI()