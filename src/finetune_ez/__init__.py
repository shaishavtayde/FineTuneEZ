"""Package for the script that  will be executed

Initializes the package and controls dataset caching based on the ENABLE_CACHING environment variable.


"""

__author__ = "Shaishav Tayde"
__email__ = "shaishav18tayde@gmail.com"

import os
from datasets import disable_caching

if os.environ.get("ENABLE_CACHING", False):
    pass
else:
    disable_caching()