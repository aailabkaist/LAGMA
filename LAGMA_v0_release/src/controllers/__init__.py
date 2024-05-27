REGISTRY = {}

from .basic_controller import BasicMAC
from .lagma_controller import LAGMAMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["lagma_mac"] = LAGMAMAC
