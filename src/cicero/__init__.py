"""Cicero reply engine.

`.env` is loaded here rather than in `config`, so that importing any module --
`llm` on its own, for instance -- sees the same environment. Anything that reads
os.environ at import time otherwise depends on import order, which is a bug
waiting for the one caller who does it differently.
"""

from ._env import load_dotenv

load_dotenv()
