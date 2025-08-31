from typing import Dict
from .models import Job

# simple in-memory registry; swap for DB/redis later if needed
jobs: Dict[str, Job] = {}
