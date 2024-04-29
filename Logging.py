import logging
import colorlog

# Create a StreamHandler with colorlog's ColoredFormatter
handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s: %(message)s",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)
handler.setFormatter(formatter)

# Get the root logger and add the handler
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)  # Set the logging level
