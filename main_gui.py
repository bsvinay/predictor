import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import re
from typing import Dict, List, Tuple, Optional

from gui.advanced_opt_gui import AdvancedOptionsGUI


# Main execution
if __name__ == "__main__":
    print("Options Chain Sentiment Analyzer")
    print("================================")
    print("Features:")
    print("- Load CSV files with options chain data")
    print("- Time-based sentiment analysis")
    print("- Interactive strikes table")
    print("- Real-time charts and indicators")
    print("- ATM/ITM/OTM classification")
    print("\nStarting GUI application...")
    
    #gui = OptionsGUI()
    gui = AdvancedOptionsGUI()
    gui.run()
