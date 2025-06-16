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


class OptionsGUI:
    def __init__(self):
        self.analyzer = OptionsAnalyzer()
        self.current_timestamp_index = 0
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI"""
        self.root = tk.Tk()
        self.root.title("Options Chain Sentiment Analyzer")
        self.root.geometry("1600x1000")
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="Data Loading")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Load CSV File", command=self.load_file).pack(side=tk.LEFT, padx=5, pady=5)
        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Time slider frame
        slider_frame = ttk.LabelFrame(main_frame, text="Time Analysis")
        slider_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.time_var = tk.DoubleVar()
        self.time_slider = ttk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                    variable=self.time_var, command=self.update_analysis)
        self.time_slider.pack(fill=tk.X, padx=5, pady=5)
        
        self.time_label = ttk.Label(slider_frame, text="Select time point")
        self.time_label.pack(pady=5)
        
        # Navigation buttons frame
        nav_frame = ttk.Frame(slider_frame)
        nav_frame.pack(pady=5)
        
        # Navigation buttons
        nav_buttons = [
            ("⏮ Start", lambda: self.jump_to_position('start')),
            ("← 1hr", lambda: self.navigate_by_time(60, -1)),
            ("← 30min", lambda: self.navigate_by_time(30, -1)),
            ("← 15min", lambda: self.navigate_by_time(15, -1)),
            ("← 5min", lambda: self.navigate_by_time(5, -1)),
            ("← 1min", lambda: self.navigate_by_time(1, -1)),
            ("1min →", lambda: self.navigate_by_time(1, 1)),
            ("5min →", lambda: self.navigate_by_time(5, 1)),
            ("15min →", lambda: self.navigate_by_time(15, 1)),
            ("30min →", lambda: self.navigate_by_time(30, 1)),
            ("1hr →", lambda: self.navigate_by_time(60, 1)),
            ("End ⏭", lambda: self.jump_to_position('end'))
        ]
        
        self.nav_buttons = {}
        for i, (text, command) in enumerate(nav_buttons):
            btn = ttk.Button(nav_frame, text=text, command=command, width=8)
            btn.grid(row=0, column=i, padx=2, pady=2)
            self.nav_buttons[text] = btn
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Analysis tab
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Analysis")
        
        # Strikes table tab  
        strikes_frame = ttk.Frame(notebook)
        notebook.add(strikes_frame, text="Strikes Chain")
        
        # Setup analysis tab
        self.setup_analysis_tab(analysis_frame)
        
        # Setup strikes table tab
        self.setup_strikes_tab(strikes_frame)
        
    def setup_analysis_tab(self, parent):
        """Setup the analysis results tab"""
        # Results frame
        results_frame = ttk.LabelFrame(parent, text="Analysis Results")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for text results
        left_panel = ttk.Frame(results_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        self.results_text = tk.Text(left_panel, width=50, height=25)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Right panel for charts
        right_panel = ttk.Frame(results_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_strikes_tab(self, parent):
        """Setup the strikes table tab"""
        # Strikes table frame
        table_frame = ttk.LabelFrame(parent, text="Options Chain - Strike Wise Data")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview for strikes table
        columns = ('CE_BuildUp', 'CE_Volume', 'CE_IV', 'CE_Gamma', 'CE_Theta', 'CE_Delta', 
                  'CE_VWAP', 'CE_LTP', 'CE_OI', 'CE_OI_Change', 'Strike', 
                  'PE_OI', 'PE_OI_Change', 'PE_LTP', 'PE_VWAP', 'PE_Gamma', 
                  'PE_Theta', 'PE_Delta', 'PE_IV', 'PE_Volume', 'PE_BuildUp')
        
        self.strikes_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=20)
        
        # Define headings
        headings = {
            'CE_BuildUp': 'CE Build Up',
            'CE_Volume': 'CE Volume', 
            'CE_IV': 'CE IV',
            'CE_Gamma': 'CE Gamma',
            'CE_Theta': 'CE Theta',
            'CE_Delta': 'CE Delta',
            'CE_VWAP': 'CE VWAP',
            'CE_LTP': 'CE LTP',
            'CE_OI': 'CE OI',
            'CE_OI_Change': 'CE OI Chg',
            'Strike': 'Strike Price',
            'PE_OI': 'PE OI',
            'PE_OI_Change': 'PE OI Chg',
            'PE_LTP': 'PE LTP',
            'PE_VWAP': 'PE VWAP',
            'PE_Gamma': 'PE Gamma',
            'PE_Theta': 'PE Theta',
            'PE_Delta': 'PE Delta',
            'PE_IV': 'PE IV',
            'PE_Volume': 'PE Volume',
            'PE_BuildUp': 'PE Build Up'
        }
        
        for col, heading in headings.items():
            self.strikes_tree.heading(col, text=heading)
            if col == 'Strike':
                self.strikes_tree.column(col, width=100, anchor='center')
            else:
                self.strikes_tree.column(col, width=80, anchor='center')
        
        # Add scrollbar
        scrollbar_v = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.strikes_tree.yview)
        scrollbar_h = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.strikes_tree.xview)
        self.strikes_tree.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        # Pack treeview and scrollbars
        self.strikes_tree.grid(row=0, column=0, sticky='nsew')
        scrollbar_v.grid(row=0, column=1, sticky='ns')
        scrollbar_h.grid(row=1, column=0, sticky='ew')
        
        # Configure grid weights
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Add tags for row coloring
        self.strikes_tree.tag_configure('ATM', background='#fff3cd')
        self.strikes_tree.tag_configure('ITM', background='#d1ecf1')
        self.strikes_tree.tag_configure('OTM', background='#f8d7da')
    
    def jump_to_position(self, position):
        """Jump to start or end of timeline"""
        if len(self.analyzer.unique_timestamps) == 0:
            return
        
        if position == 'start':
            self.current_timestamp_index = 0
        elif position == 'end':
            self.current_timestamp_index = len(self.analyzer.unique_timestamps) - 1
        
        self.time_slider.set(self.current_timestamp_index)
        self.update_analysis()
    
    def navigate_by_time(self, minutes, direction):
        """Navigate by specified time interval"""
        if len(self.analyzer.unique_timestamps) == 0:
            return
        
        try:
            current_time = self.analyzer.unique_timestamps[self.current_timestamp_index]
            target_time = current_time + pd.Timedelta(minutes=direction * minutes)
            
            # Find closest timestamp to target
            closest_index = self.current_timestamp_index
            closest_diff = float('inf')
            
            for i, timestamp in enumerate(self.analyzer.unique_timestamps):
                diff = abs((timestamp - target_time).total_seconds())
                if diff < closest_diff:
                    closest_diff = diff
                    closest_index = i
            
            # Only move if we found a different timestamp
            if closest_index != self.current_timestamp_index and 0 <= closest_index < len(self.analyzer.unique_timestamps):
                self.current_timestamp_index = closest_index
                self.time_slider.set(closest_index)
                self.update_analysis()
        except Exception as e:
            print(f"Error navigating by time: {e}")
    
    def update_navigation_buttons(self):
        """Update navigation button states"""
        can_go_back = self.current_timestamp_index > 0
        can_go_forward = self.current_timestamp_index < len(self.analyzer.unique_timestamps) - 1
        is_at_start = self.current_timestamp_index == 0
        is_at_end = self.current_timestamp_index == len(self.analyzer.unique_timestamps) - 1
        
        # Enable/disable buttons based on current position
        for text, btn in self.nav_buttons.items():
            if text == "⏮ Start":
                btn.config(state='disabled' if is_at_start else 'normal')
            elif text == "End ⏭":
                btn.config(state='disabled' if is_at_end else 'normal')
            elif "←" in text:  # Back buttons
                btn.config(state='normal' if can_go_back else 'disabled')
            else:  # Forward buttons
                btn.config(state='normal' if can_go_forward else 'disabled')
    
    def load_file(self):
        """Load CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select Options Chain CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.analyzer.load_data(file_path)
                self.analyzer.process_full_data()
                
                self.file_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                
                # Setup time slider
                if len(self.analyzer.unique_timestamps) > 0:
                    self.time_slider.config(to=len(self.analyzer.unique_timestamps)-1)
                    self.time_slider.set(len(self.analyzer.unique_timestamps)-1)
                    self.current_timestamp_index = len(self.analyzer.unique_timestamps)-1
                
                self.update_analysis()
                messagebox.showinfo("Success", "Data loaded successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def update_analysis(self, *args):
        """Update analysis based on slider position"""
        if len(self.analyzer.unique_timestamps) == 0:
            return
        
        # Get current slider position
        pos = int(self.time_var.get())
        pos = min(pos, len(self.analyzer.unique_timestamps) - 1)
        self.current_timestamp_index = pos
        
        # Analyze current timestamp
        analysis = self.analyzer.analyze_timestamp(pos)
        
        if not analysis:
            return
        
        # Update time label
        timestamp = analysis['timestamp']
        self.time_label.config(text=f"Time: {timestamp.strftime('%H:%M:%S')}")
        
        # Update results text
        self.update_results_text(analysis)
        
        # Update charts
        self.update_charts(pos)
        
        # Update strikes table
        self.update_strikes_table(analysis.get('strikes_data', pd.DataFrame()))
        
        # Update navigation buttons
        self.update_navigation_buttons()
    
    def update_results_text(self, analysis):
        """Update the results text widget"""
        self.results_text.delete(1.0, tk.END)
        
        if not analysis:
            self.results_text.insert(tk.END, "No analysis data available")
            return
        
        indicators = analysis['indicators']
        range_data = analysis['range_data']
        greek_comp = range_data.get('greekComposition', {})
        
        text = f"""ADVANCED SENTIMENT ANALYSIS RESULTS
{'='*55}

Current Sentiment: {analysis['sentiment']}
Strength Indicator: {analysis['strength']}/50

ADVANCED DELTA-RANGE ANALYSIS:
ATM Strike: {range_data['atm']:.0f} (Delta: {range_data.get('atm_delta', 0):.3f})
Predicted Range: {range_data['low']:.0f} - {range_data['high']:.0f}

GREEK COMPOSITION ANALYSIS (User's Methodology):
Greek Sum (V+T+G): {range_data['greekSum']:.0f} ({range_data['confidence']} Energy)
  • Theta Dominance: {greek_comp.get('theta', 0)*100:.1f}% ({('Sideways Bias' if greek_comp.get('theta', 0) > 0.5 else 'Active')})
  • Vega Dominance: {greek_comp.get('vega', 0)*100:.1f}% ({('High Volatility' if greek_comp.get('vega', 0) > 0.6 else 'Normal')})
  • Gamma Dominance: {greek_comp.get('gamma', 0)*100:.1f}% ({('Breakout Setup' if greek_comp.get('gamma', 0) > 0.5 else 'Stable')})

MARKET PRESSURE: {range_data['pressure']}
Confidence Level: {range_data['confidence']}

TECHNICAL INDICATORS:
{'='*30}
Total OI Change: {indicators.get('total_oi_change', 0):,.0f}
Total Volume Change: {indicators.get('total_volume_change', 0):,.0f}
PCR-OI: {indicators.get('pcr_oi', 1):.3f}
PCR-Vol: {indicators.get('pcr_vol', 1):.3f}
Total Volume: {indicators.get('total_volume', 0):,.0f}
Total OI: {indicators.get('total_oi', 0):,.0f}

GREEKS ANALYSIS:
{'='*20}
Avg Delta: {indicators.get('avg_delta', 0):.4f}
Avg Gamma: {indicators.get('avg_gamma', 0):.6f}
Avg Theta: {indicators.get('avg_theta', 0):.4f}
Avg Vega: {indicators.get('avg_vega', 0):.4f}

ADVANCED ANALYSIS:
{'='*25}
{range_data['analysis']}

USER'S METHODOLOGY INSIGHTS:
{'='*35}
✓ Dynamic ATM tracking based on Delta closest to 0.5
✓ Focus on Delta range 0.05-0.6 for meaningful strikes
✓ Greek sum indicates market energy and dominant forces
✓ ITM CE buildup = Downside targets (index moves DOWN)
✓ ITM PE buildup = Upside targets (breakout levels)
✓ Gamma spikes = Directional moves incoming

INTERPRETATION:
{'='*20}
"""
        
        # Add interpretation based on market pressure
        if "SIDEWAYS" in range_data['pressure']:
            text += "➡️ Theta dominance suggests range-bound movement\n"
            text += "⏳ Time decay working, limited directional moves\n"
            text += f"🎯 Expected Range: {range_data['low']:.0f} - {range_data['high']:.0f}\n"
        elif "HIGH VOLATILITY" in range_data['pressure']:
            text += "⚡ Vega dominance indicates uncertainty\n"
            text += "🌪️ High volatility conditions, unpredictable moves\n"
            text += f"🎯 Volatile Range: {range_data['low']:.0f} - {range_data['high']:.0f}\n"
        elif "DIRECTIONAL MOVE" in range_data['pressure']:
            text += "🚀 Gamma dominance suggests breakout setup\n"
            text += "⚡ Acceleration risk high, big move coming\n"
            text += f"🎯 Breakout Range: {range_data['low']:.0f} - {range_data['high']:.0f}\n"
        elif "BEARISH BIAS" in range_data['pressure']:
            text += "🔴 ITM CE buildup indicates downside targets\n"
            text += "📉 Market trying to reach lower strike levels\n"
            text += f"🎯 Downside Target: {range_data['low']:.0f}\n"
        elif "BULLISH BIAS" in range_data['pressure']:
            text += "🟢 ITM PE buildup indicates upside targets\n"
            text += "📈 Market aiming for higher strike levels\n"
            text += f"🎯 Upside Target: {range_data['high']:.0f}\n"
        else:
            text += "⚖️ Balanced conditions\n"
            text += "📊 Monitor for Greek composition shifts\n"
            text += f"🎯 Range: {range_data['low']:.0f} - {range_data['high']:.0f}\n"
        
        text += f"\n📍 Current ATM: {range_data['atm']:.0f} (Market reference level)"
        
        self.results_text.insert(tk.END, text)
    
    def update_strikes_table(self, strikes_data):
        """Update the strikes table"""
        # Clear existing data
        for item in self.strikes_tree.get_children():
            self.strikes_tree.delete(item)
        
        if strikes_data.empty:
            return
        
        # Sort by strike price
        strikes_data = strikes_data.sort_values('Strike Price')
        
        # Find ATM strike (middle strike)
        strikes = strikes_data['Strike Price'].unique()
        if len(strikes) > 0:
            middle_index = len(strikes) // 2
            atm_strike = strikes[middle_index]
        else:
            atm_strike = 0
        
        # Helper function to safely convert values
        def safe_float(val, default=0):
            if pd.isna(val) or val == '-' or val == '':
                return default
            try:
                # Clean the value if it's a string
                if isinstance(val, str):
                    val = val.replace(',', '').replace('"', '').strip()
                return float(val)
            except:
                return default
        
        def safe_format_number(val, decimals=2):
            try:
                num_val = safe_float(val)
                if decimals == 0:
                    return f"{num_val:,.0f}"
                else:
                    return f"{num_val:.{decimals}f}"
            except:
                return "0"
        
        def safe_format_percent(val):
            try:
                num_val = safe_float(val)
                return f"{num_val:.2f}%"
            except:
                return "0.00%"
        
        for _, row in strikes_data.iterrows():
            try:
                strike = safe_float(row.get('Strike Price', 0))
                
                # CE (Call) data - columns before Strike Price
                ce_buildup = str(row.get('Build Up', '-'))
                ce_volume = safe_float(row.get('Volume', 0))
                ce_iv = safe_float(row.get('IV', 0))
                ce_gamma = safe_float(row.get('Gamma', 0))
                ce_theta = safe_float(row.get('Theta', 0))
                ce_delta = safe_float(row.get('Delta', 0))
                ce_vwap = safe_float(row.get('VWAP', 0))
                ce_ltp = safe_float(row.get('LTP', 0))
                ce_oi = safe_float(row.get('OI', 0))
                ce_oi_change = safe_float(row.get('OI Chg', 0))
                
                # PE (Put) data - need to extract from columns after Strike Price
                pe_oi = 0
                pe_oi_change = 0
                pe_ltp = 0
                pe_vwap = 0
                pe_gamma = 0
                pe_theta = 0
                pe_delta = 0
                pe_iv = 0
                pe_volume = 0
                pe_buildup = '-'
                
                # Try to extract PE data if available
                columns = strikes_data.columns.tolist()
                strike_index = -1
                try:
                    strike_index = columns.index('Strike Price')
                except ValueError:
                    pass
                    
                if strike_index > 0:
                    # Look for PE columns after Strike Price
                    pe_columns = columns[strike_index + 1:]
                    
                    # Map PE data based on known column patterns from your CSV
                    for i, col in enumerate(pe_columns):
                        try:
                            col_val = row.get(col, 0)
                            if pd.notna(col_val):
                                if i == 14:  # Approximate PE OI position
                                    pe_oi = safe_float(col_val)
                                elif i == 15:  # PE OI Change
                                    pe_oi_change = safe_float(col_val)
                                elif i == 5:   # PE LTP
                                    pe_ltp = safe_float(col_val)
                                elif i == 4:   # PE VWAP
                                    pe_vwap = safe_float(col_val)
                                elif i == 21:  # PE Gamma
                                    pe_gamma = safe_float(col_val)
                                elif i == 22:  # PE Theta
                                    pe_theta = safe_float(col_val)
                                elif i == 20:  # PE Delta
                                    pe_delta = safe_float(col_val)
                                elif i == 17:  # PE IV
                                    pe_iv = safe_float(col_val)
                                elif i == 11:  # PE Volume
                                    pe_volume = safe_float(col_val)
                                elif i == len(pe_columns) - 1:  # Last column (Build Up)
                                    pe_buildup = str(col_val) if col_val != '-' else '-'
                        except Exception as col_error:
                            continue
                
                # Determine tag for row coloring
                tag = ''
                if abs(strike - atm_strike) <= 25:
                    tag = 'ATM'
                elif strike < atm_strike:
                    tag = 'ITM'
                else:
                    tag = 'OTM'
                
                # Insert row with proper formatting
                values = (
                    ce_buildup,
                    safe_format_number(ce_volume, 0),
                    safe_format_percent(ce_iv),
                    safe_format_number(ce_gamma, 4),
                    safe_format_number(ce_theta, 2),
                    safe_format_number(ce_delta, 4),
                    safe_format_number(ce_vwap, 2),
                    safe_format_number(ce_ltp, 2),
                    safe_format_number(ce_oi, 0),
                    safe_format_number(ce_oi_change, 0),
                    safe_format_number(strike, 0),
                    safe_format_number(pe_oi, 0),
                    safe_format_number(pe_oi_change, 0),
                    safe_format_number(pe_ltp, 2),
                    safe_format_number(pe_vwap, 2),
                    safe_format_number(pe_gamma, 4),
                    safe_format_number(pe_theta, 2),
                    safe_format_number(pe_delta, 4),
                    safe_format_percent(pe_iv),
                    safe_format_number(pe_volume, 0),
                    pe_buildup
                )
                
                self.strikes_tree.insert('', tk.END, values=values, tags=(tag,))
                
            except Exception as e:
                print(f"Error processing row for strike {row.get('Strike Price', 'unknown')}: {e}")
                continue
    
    def update_charts(self, current_pos):
        """Update the charts"""
        self.fig.clear()
        
        if len(self.analyzer.strength_history) == 0:
            return
        
        # Create subplots
        ax1 = self.fig.add_subplot(3, 1, 1)
        ax2 = self.fig.add_subplot(3, 1, 2)
        ax3 = self.fig.add_subplot(3, 1, 3)
        
        # Plot 1: Strength over time
        times = range(len(self.analyzer.strength_history[:current_pos+1]))
        strengths = self.analyzer.strength_history[:current_pos+1]
        
        ax1.plot(times, strengths, 'b-', linewidth=2, label='Strength')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.axhline(y=25, color='g', linestyle='--', alpha=0.3, label='Bullish Threshold')
        ax1.axhline(y=-25, color='r', linestyle='--', alpha=0.3, label='Bearish Threshold')
        ax1.set_ylabel('Strength (-50 to 50)')
        ax1.set_title('Sentiment Strength Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        if current_pos < len(strengths):
            ax1.scatter([current_pos], [strengths[current_pos]], color='red', s=50, zorder=5)
        
        # Plot 2: Sentiment distribution
        sentiments = self.analyzer.sentiment_history[:current_pos+1]
        if sentiments:
            sentiment_counts = {}
            for sentiment in sentiments:
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            labels = list(sentiment_counts.keys())
            values = list(sentiment_counts.values())
            colors = ['green' if 'BULL' in label else 'red' if 'BEAR' in label else 'blue' for label in labels]
            
            ax2.bar(range(len(labels)), values, color=colors, alpha=0.7)
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=45, ha='right')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Sentiment Distribution')
        
        # Plot 3: PCR Analysis
        if current_pos < len(self.analyzer.unique_timestamps):
            current_analysis = self.analyzer.analyze_timestamp(current_pos)
            if current_analysis and 'indicators' in current_analysis:
                pcr_data = []
                for i in range(min(10, current_pos + 1)):  # Last 10 data points
                    analysis = self.analyzer.analyze_timestamp(current_pos - 9 + i)
                    if analysis:
                        pcr_data.append(analysis['indicators'].get('pcr_oi', 1))
                
                if pcr_data:
                    ax3.plot(range(len(pcr_data)), pcr_data, 'g-', marker='o', linewidth=2)
                    ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Neutral PCR')
                    ax3.axhline(y=1.2, color='g', linestyle='--', alpha=0.5, label='Bullish Zone')
                    ax3.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Bearish Zone')
                    ax3.set_ylabel('PCR-OI')
                    ax3.set_title('Put-Call Ratio Analysis (Last 10 Points)')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()
