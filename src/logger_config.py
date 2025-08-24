import logging
import sys
from streamlit.logger import get_logger

# --- Custom Log Handler for Streamlit UI ---
class StreamlitLogHandler(logging.Handler):
    """
    A custom logging handler that stores log records to be displayed in the Streamlit UI.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def emit(self, record):
        self.records.append(self.format(record))

# --- Logger Setup Function ---
def setup_logger():
    """
    Configures and returns a logger for the application.
    
    This setup includes:
    1. A standard console handler.
    2. A custom handler to capture logs for the Streamlit UI.
    """
    # Get the root logger used by Streamlit
    logger = get_logger(__name__)
    logger.setLevel(logging.INFO)

    # Prevent adding handlers multiple times in Streamlit's execution flow
    for handler in logger.handlers:
        if isinstance(handler, StreamlitLogHandler):
            return logger, handler

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Custom Streamlit Handler
    streamlit_handler = StreamlitLogHandler()
    streamlit_handler.setLevel(logging.INFO)
    streamlit_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(streamlit_handler)

    return logger, streamlit_handler

# Initialize the logger and handler for import in other modules
logger, st_log_handler = setup_logger()
# Expose the logger and handler for use in other parts of the application