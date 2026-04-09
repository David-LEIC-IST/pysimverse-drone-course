# Pysimverse

A Python drone simulation framework for autonomous flight operations.

## Quick Start

```python
import time
from pysimverse import Drone

drone = Drone()
drone.connect()
drone.take_off()
time.sleep(3)
drone.land()
```

## Missions

Each Python file represents a drone mission. More missions coming soon...

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

*This README will be updated as the project grows.*
