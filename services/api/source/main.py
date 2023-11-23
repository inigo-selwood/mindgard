import datetime

import pydantic
import fastapi


app = fastapi.FastAPI()


class RootResponse(pydantic.BaseModel):
    time: datetime.datetime = datetime.datetime.now()


@app.get("/")
def root() -> RootResponse:
    return RootResponse()
