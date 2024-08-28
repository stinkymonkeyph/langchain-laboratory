from langchain_community.tools.tavily_search import TavilySearchResults


def get_profile_url_travily(name: str):
    search = TavilySearchResults()
    res = search.run(f"given the name {name}, search for linkedin profile")
    return res[0]["url"]
