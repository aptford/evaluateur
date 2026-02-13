"""Shared test configuration.

Load .env early so that ``skipif`` markers evaluated at collection time
can see API-key environment variables.
"""

from dotenv import load_dotenv

load_dotenv()
