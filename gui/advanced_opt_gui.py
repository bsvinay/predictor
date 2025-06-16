# Fixed AdvancedOptionsGUI that won't freeze

from analyzer.advanced_opt_analyzer import AdvancedOptionsAnalyzer
from gui.options_gui import OptionsGUI
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class AdvancedOptionsGUI(OptionsGUI):
    def __init__(self):
        self.analyzer = AdvancedOptionsAnalyzer()
        self.current_timestamp_index = 0
        self.setup_gui()
        self.setup_enhanced_vega_integration()

    def setup_enhanced_vega_integration(self):
        """Add simplified enhanced Vega analysis tab (no GUI freezing)"""
        try:
            # Import and integrate the simplified enhanced Vega tab
            from analyzer.enhanced_vega_analysis_tab import integrate_enhanced_vega_tab_simple
            integrate_enhanced_vega_tab_simple(self)
            
        except Exception as e:
            print(f"Error setting up enhanced Vega integration: {e}")
            import traceback
            traceback.print_exc()

    def update_analysis(self, *args):
        """Enhanced update method"""
        # Call parent update first
        super().update_analysis(*args)
        
        # Update enhanced Vega analysis if available
        if hasattr(self, 'enhanced_vega_tab'):
            try:
                # Import and use the simplified update function
                from analyzer.enhanced_vega_analysis_tab import update_enhanced_vega_analysis_simple
                update_enhanced_vega_analysis_simple(self, *args)
                
            except Exception as e:
                print(f"Error updating enhanced Vega analysis: {e}")

    def load_file(self):
        """Enhanced file loading - STARTS AT BEGINNING OF DAY"""
        file_path = filedialog.askopenfilename(
            title="Select Options Chain CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Load data
                self.analyzer.load_data(file_path)
                
                # Set baseline
                self.analyzer.set_baseline()
                
                # Process full data
                self.analyzer.process_full_data()
                
                # Update file label
                self.file_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                
                # Setup time slider - START AT BEGINNING OF DAY (position 0)
                if len(self.analyzer.unique_timestamps) > 0:
                    self.time_slider.config(to=len(self.analyzer.unique_timestamps)-1)
                    self.time_slider.set(0)  # ✅ START AT BEGINNING OF DAY
                    self.current_timestamp_index = 0  # ✅ START AT BEGINNING OF DAY
                
                # Update analysis
                self.update_analysis()
                
                messagebox.showinfo("Success", 
                    "Data loaded successfully!\n\n"
                    "✅ Started at beginning of day\n"
                    "✅ Baseline Greeks displayed\n" 
                    "✅ Simple verification available (no GUI freezing)\n\n"
                    "Navigate through timestamps to build prediction history.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
                import traceback
                traceback.print_exc()
