"""Shared test setup."""
import os

# Set before any test imports the app. Load_dotenv keeps existing values, so this
# wins over .env and test questions are never sent to the real Arize project.
os.environ["ARIZE_TRACING"] = "false"
