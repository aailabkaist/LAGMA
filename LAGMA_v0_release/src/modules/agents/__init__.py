REGISTRY = {}

from .rnn_agent import RNNAgent
from .lagma_agent import LAGMAAgent 

REGISTRY["rnn"] = RNNAgent
REGISTRY["lagma"] = LAGMAAgent
