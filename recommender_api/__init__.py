import os

api_dir = os.path.dirname(os.path.abspath(__file__))
with open (os.path.join(api_dir, 'VERSION'), 'rb') as f:
    __version__ = f.read().strip()

