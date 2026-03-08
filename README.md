🧬 GENESIS: Clinical Logic Engine
=================================

1\. Project Overview
--------------------

Genesis is an AI software that acts like a "Robotic Data Scientist." It takes raw patient data (Weight, Age, Medical History) and uses **Genetic Programming** to autonomously discover the mathematical formula required for safe drug dosage. It solves the "Black Box" problem in medical AI.

2\. Prerequisites
-----------------

To run this software, you need **Python** installed on your computer.

1.  Download Python here: [python.org/downloads](https://www.google.com/url?sa=E&q=https://www.python.org/downloads/)
    
2.  During installation, make sure to check the box: **"Add Python to PATH"**.
    

3\. Installation Guide
----------------------

1.  Unzip this project folder.
    
2.  Open your **Command Prompt** (Windows) or **Terminal** (Mac/Linux).
    
3.  Navigate to this folder using the cd command.
    
    *   _Example:_ cd Downloads/Genesis\_Project
        
4.  codeBashpip install streamlit pandas plotly graphviz sympy_(Note: For the Tree Visualization to work perfectly, you may need the Graphviz system software installed, but the app will work without it)._
    

4\. How to Run the App
----------------------

1.  codeBashstreamlit run app.py
    
2.  A new tab will automatically open in your web browser (Chrome/Edge).
    
3.  You will see the **GENESIS Dashboard**.
    

5\. How to Use the App
----------------------

1.  **Landing Page:** Click **"RUN ON KNOWN FORMULA"** to enter the Clinical Simulator.
    
2.  **Dashboard:**
    
    *   **Left Sidebar:** You can add specific patients manually or reset the database.
        
    *   **Scenario Config:** Change the "Target Formula" (e.g., switch from Linear Dose to Geriatric Decay) to test if the AI is truly smart.
        
3.  **Running the AI:**
    
    *   Click the purple **"GENERATE FORMULA"** button.
        
    *   Watch the progress bar and the logs as the AI evolves logic in real-time.
        
4.  **Verification:**
    
    *   Once finished, scroll down.
        
    *   Open the **"Evolutionary Logs"** dropdown to see which formulas survived and died.
        
    *   Open the **"3D Visuals"** to see the math surface mapped against patient data.
        

6\. Troubleshooting
-------------------

*   _Error: "Command not found":_ Make sure Python is added to your system PATH.
    
*   _Error: "Module not found":_ Run the pip install command again and check for typos.