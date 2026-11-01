"""News Synthesizer: summarizes real headlines, never anything from memory."""
from pydantic import BaseModel

from src.agents.common import explain, extract, latest_question, output
from src.data.market_data import InvalidSymbolError, normalize_symbol
from src.data.news import NewsError
from src.utils.metrics import annotate, traced


class NewsRequest(BaseModel):
    query: str = "stock market"
    symbol: str | None = None


INSTRUCTION = ("Give only the company name or topic the user's latest message wants news about, in "
               "one to three words, such as \"Nvidia\" or \"interest rates\". Leave out words like "
               "\"latest\" and \"news\". Use earlier messages only to resolve words like \"it\". Use "
               "\"stock market\" for general market news. If it is about one public company, also give "
               "its US ticker as symbol.")


def make_news_agent(llm, get_news):
    def run(state):
        request = extract(llm, NewsRequest, INSTRUCTION, state)
        query = request.query if request else latest_question(state)
        symbol = None
        if request and request.symbol:
            try:
                symbol = normalize_symbol(request.symbol)
            except InvalidSymbolError:
                pass
        try:
            with traced("news_search", has_symbol=bool(symbol)):
                articles = get_news().search(query, symbol)
                annotate(result_count=len(articles))
        except NewsError:
            return output("news", "News is unavailable right now.", error=True)
        if not articles:
            return output("news", f"I couldn't find recent headlines specifically about \"{query}\".")
        facts = "\n".join(
            f"- {a.title} ({a.publisher}" + (f", {a.published.date().isoformat()})" if a.published else ")")
            for a in articles
        )
        task = ("Summarize the themes in these headlines and give their dates. You only have the "
                "headlines, not the articles, so don't state details or causes beyond them. Headlines "
                "may be opinions or promotions, so attribute them to the publisher and don't endorse them.")
        sources = [f"{a.title} ({a.publisher}): {a.url}" for a in articles]
        return output("news", explain(llm, task, facts, state), sources, facts=facts)

    return run
