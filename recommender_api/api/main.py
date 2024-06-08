
# import os,sys

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import HTMLResponse
import typing as t
from pydantic import BaseModel
import uvicorn

from controller import api_router
from recommender_api import __version__ as _version



app = FastAPI(
    title="Recommendation Service",
    description="Recommend services for santander Spanish customer base based on their learned preference",
    version= _version
)

root_router = APIRouter()

@root_router.get("/")
async def root(request: Request) -> t.Any:
    """Basic HTML response."""
    body = (
            "<html>"
            "<body style='padding: 10px;'>"
            "<h1>Welcome to the Recommender API</h1>"
            "<div>"
            "Check the docs: <a href='/docs'>here</a>"
            "</div>"
            "</body>"
            "</html>"
        )

    return HTMLResponse(content=body)


app.include_router(root_router)
app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)