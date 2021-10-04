from zensols.util import APIError


class NLPError(APIError):
    """Raised for any errors for this library."""
    pass


class ParseError(APIError):
    """Raised for any parsing errors."""
    pass


from .norm import *
from .stemmer import *
from .feature import *
from .lang import *
from .container import *
from .docparser import *
from .combine import *
