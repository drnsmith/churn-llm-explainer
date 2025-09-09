# app.py
import subprocess
subprocess.run(["streamlit", "run", "app/main.py", "--server.port=7860", "--server.address=0.0.0.0"])
# Note: This script runs the Streamlit app. Make sure you have Streamlit installed and the app files are in the correct directory.
# You can run this script from the command line to start the app.
# Usage: python app.py