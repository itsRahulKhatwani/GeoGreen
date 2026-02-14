# How to Run GeoGreen Revolution

> [!IMPORTANT]
> **Python is not installed or not found!**
> You must install Python before running this project.

## Step 1: Install Python
1.  Download Python 3.10+ from [python.org](https://www.python.org/downloads/).
2.  **CRITICAL:** During installation, check the box **"Add Python to PATH"**.
3.  Restart your computer or terminal after installation.

## Step 2: Set Up Environment
Open your terminal (Command Prompt or PowerShell) in the project folder:
`c:\Users\RAHUL\Desktop\GeoGreen Revolution\GeoGreen Revolution`

Run the following commands:
```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the environment
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install additional dashboard dependencies
pip install streamlit easyocr
```

## Step 3: Run the Project (For Presentation)
For the best presentation experience, use the **Streamlit Dashboard**:
```bash
streamlit run app.py
```
This will open a beautiful web interface in your browser.

### Alternative: Run the Standard Pipeline
If you want to show the backend processing:
```bash
python src/main.py
```

## Data Check
Your project already has data in `data/`, so you **do not** need to run `download_data.py`.
If you ever need fresh sample data, run: `python generate_dummy_data.py`
