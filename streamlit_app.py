"""
Root Entrypoint for Streamlit Application
Redirects execution to app/streamlit_app.py
"""

import os
import sys

app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Execute app/streamlit_app.py
streamlit_app_path = os.path.join(app_dir, "streamlit_app.py")
with open(streamlit_app_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), streamlit_app_path, 'exec')
    exec(code, globals())
