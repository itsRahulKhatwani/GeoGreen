"""
Patches run_presentation.ipynb to add a Streamlit dashboard launch cell at the end.
Run once: python patch_notebook.py
"""
import json, os

NB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_presentation.ipynb")

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove any previously injected launch cells so this is idempotent
nb["cells"] = [
    c for c in nb["cells"]
    if not any("DASHBOARD IS LIVE" in s for s in c.get("source", []))
    and not any("Step 7" in s for s in c.get("source", []))
    and not any("Launch the Interactive Dashboard" in s for s in c.get("source", []))
]

launch_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## \U0001f680 Step 7 \u2014 Launch the Interactive Dashboard\n",
        "\n",
        "The cell below starts the **Streamlit web dashboard** in the background "
        "and opens it automatically in your browser.  \n",
        "All the analysis results above are now loaded into the dashboard \u2014 "
        "explore them interactively.\n",
        "\n",
        "> \U0001f4a1 **To analyze your own TIF files:** Go to the dashboard \u2192 "
        "**\U0001f4ca Scientific Results** tab \u2192 scroll down to "
        "**Re-run Scientific Pipeline** \u2192 upload your individual band TIF files there."
    ]
}

launch_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import subprocess, sys, time, os, webbrowser\n",
        "\n",
        "# \u2500\u2500 1. Install folium silently (enables interactive map tab) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
        "print('\U0001f4e6 Checking dependencies...')\n",
        "subprocess.run([sys.executable, '-m', 'pip', 'install', 'folium', '-q'],\n",
        "               capture_output=True)\n",
        "print('   \u2705 folium ready')\n",
        "\n",
        "# \u2500\u2500 2. Kill anything already running on port 8501 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
        "try:\n",
        "    if os.name == 'nt':  # Windows\n",
        "        subprocess.run(\n",
        "            'for /f \"tokens=5\" %a in (\"netstat -aon | findstr :8501\") '\n",
        "            'do taskkill /F /PID %a',\n",
        "            shell=True, capture_output=True)\n",
        "    else:\n",
        "        subprocess.run('fuser -k 8501/tcp', shell=True, capture_output=True)\n",
        "except Exception:\n",
        "    pass\n",
        "time.sleep(1)\n",
        "\n",
        "# \u2500\u2500 3. Launch Streamlit in the background \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
        "APP_PATH = os.path.join(NOTEBOOK_DIR, 'app.py')\n",
        "print(f'\U0001f680 Launching dashboard: {APP_PATH}')\n",
        "\n",
        "_proc = subprocess.Popen(\n",
        "    [sys.executable, '-m', 'streamlit', 'run', APP_PATH,\n",
        "     '--server.port', '8501',\n",
        "     '--server.headless', 'true',\n",
        "     '--browser.gatherUsageStats', 'false'],\n",
        "    cwd=NOTEBOOK_DIR,\n",
        "    stdout=subprocess.DEVNULL,\n",
        "    stderr=subprocess.DEVNULL\n",
        ")\n",
        "\n",
        "# \u2500\u2500 4. Wait then auto-open browser \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
        "print('\u23f3 Starting server (5 seconds)...')\n",
        "time.sleep(5)\n",
        "webbrowser.open('http://localhost:8501')\n",
        "\n",
        "print()\n",
        "print('=' * 58)\n",
        "print('  \u2705  DASHBOARD IS LIVE!')\n",
        "print('  \U0001f310  http://localhost:8501')\n",
        "print()\n",
        "print('  Navigation:')\n",
        "print('  \U0001f3e0 Executive Summary  \u2014 Full story + 3-phase roadmap')\n",
        "print('  \U0001f4ca Scientific Results \u2014 Maps + Interactive GIS map')\n",
        "print('  \U0001f4cb Recommendations    \u2014 Species, schemes, budgets')\n",
        "print('  \U0001f6f0\ufe0f  Live Demo         \u2014 Upload any image to classify')\n",
        "print()\n",
        "print('  To analyze YOUR TIF files:')\n",
        "print('  \u2192 \U0001f4ca Scientific Results > scroll down >')\n",
        "print('    \"Re-run Scientific Pipeline\" expander')\n",
        "print('=' * 58)\n",
    ]
}

# Insert before the last markdown cell (Conclusion) if possible, otherwise append
inserted = False
for i in range(len(nb["cells"]) - 1, -1, -1):
    cell = nb["cells"][i]
    if cell["cell_type"] == "markdown" and any("Conclusion" in s for s in cell.get("source", [])):
        nb["cells"].insert(i, launch_code)
        nb["cells"].insert(i, launch_md)
        inserted = True
        break

if not inserted:
    nb["cells"].extend([launch_md, launch_code])

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✅ Patched: {NB_PATH}")
print(f"   Added Step 7 launch cell ({len(nb['cells'])} total cells now)")
