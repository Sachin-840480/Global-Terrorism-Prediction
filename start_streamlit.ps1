# Activate virtual environment
cd "D:\VS Code Programs\Python"
.venv\Scripts\Activate

# Change the Directory to the Place where 'Server' file is Located.
cd "projects\gtd"

# Set Streamlit Configuration directory.
set STREAMLIT_CONFIG_DIR= "D:\VS Code Programs\Python\projects\gtd\.streamlit"

# Start the Server.
streamlit run gtd_dashboard_final.py
