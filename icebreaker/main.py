from fastapi import FastAPI

from ice_breaker import ice_break_with

app = FastAPI()


@app.get("/ice_break")
def ice_break(name: str = ""):
    res, profile_pic_url = ice_break_with(name)
    return {
        "summary": res.summary,
        "facts": res.facts,
        "profile_pic_url": profile_pic_url,
    }
