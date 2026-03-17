# Fetchers Messages
# Messages for user guidance on data fetching operations

FETCHER_START = "Starting data fetching process for {source}."
"This may take a few minutes depending on the data volume."

FETCHER_SUCCESS = "Successfully fetched data from {source}."
"Retrieved {records} records. Data saved to raw directory."

FETCHER_CACHE_HIT = "Using cached data for {ticker} from {start_date} to {end_date}."
"No new API call needed."

FETCHER_RETRY = "Retrying fetch for {ticker} due to temporary error."
"Attempt {attempt}/{max_attempts}."

FETCHER_RATE_LIMIT = (
    "Rate limit reached for {source}. Waiting {wait_time} seconds before retry."
)

FETCHER_TIMEOUT = (
    "Request timed out for {ticker}. This is normal for large date ranges.Retrying..."
)

FETCHER_API_ERROR = (
    "API error from {source}: {error}. Check your API keys and network connection."
)

FETCHER_DATA_VALIDATION = (
    "Validating fetched data... Found {valid_records} valid records out of "
    "{total_records}."
)

FETCHER_COMPLETION = "Data fetching completed. Total files processed: {total_files}."
"Success rate: {success_rate}%."

# User guidance messages
FETCHER_USER_GUIDE = """
Data Fetching Guide:
- YFinance: Fetches stock data from Yahoo Finance. You'll see progress bars
and completion messages.
- FRED: Retrieves economic indicators. Data includes inflation, unemployment, etc.
- World Bank: Gets global development data. Includes GDP, population, trade statistics.
- All data is cached locally to avoid repeated API calls.
- Check the raw/ directory for downloaded files.
"""

# Output explanations
FETCHER_OUTPUT_EXPLANATION = """
What you'll see on screen:
- Progress bars showing download status
- Success messages with record counts
- Cache hit messages (faster execution)
- Error messages if APIs are unavailable
- Final summary with total files and success rate
"""
