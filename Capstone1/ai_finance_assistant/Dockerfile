# The Streamlit app by default; the same image also runs the MCP server.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FINANCE_DB_PATH=/app/data/finance_assistant.db

WORKDIR /app
RUN useradd --create-home --uid 1000 app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src ./src
COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/

# Everything the app writes lives in /app/data, so one volume keeps it across restarts.
RUN mkdir /app/data && chown app /app/data \
    && ln -s /app/data/chroma_db /app/chroma_db \
    && ln -s /app/data/feedback.jsonl /app/feedback.jsonl

USER app
VOLUME /app/data
EXPOSE 8501

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["streamlit", "run", "src/web_app/app.py", "--server.address=0.0.0.0", \
     "--server.port=8501", "--server.headless=true"]
