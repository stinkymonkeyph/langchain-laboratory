from langchain_community.tools.tavily_search import TavilySearchResults


def get_profile_url_travily(name: str):
    search = TavilySearchResults()
    res = search.run(f"given the full name {name} searched for linkedin profile")
    return res[0]["url"]
