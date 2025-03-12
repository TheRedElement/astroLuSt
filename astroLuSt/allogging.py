
#%%imports
import logging
import logging.handlers

#%%definitions
def setup_logger(level:int=logging.INFO):

    format = "%(levelname)-8.8s: %(asctime)s, %(name)s [%(processName)-12.12s]: %(message)s (%(context)s)"
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(format, defaults=dict(context=""))
    handler.setFormatter(formatter)
    logging.basicConfig(
        level=level,
        handlers=[handler]
    )

    return
