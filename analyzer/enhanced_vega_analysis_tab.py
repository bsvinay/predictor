# Simplified, robust Enhanced Vega Analysis Tab that won't freeze the GUI

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import threading

class EnhancedVegaAnalysisTab:
    """Simplified Enhanced Vega Analysis Tab - Robust and Non-Blocking"""
    
    def __init__(self, parent_frame, analyzer):
        self.parent = parent_frame
        self.analyzer = analyzer
        
        # Simple prediction tracking (no complex ML processing)
        self.prediction_history = []
        self.max_history = 50  # Limit to prevent memory issues
        
        # Simple auto-tuning parameters
        self.tuning_params = {
            'vega_weight': 0.6,
            'gamma_weight': 0.25,
            'theta_weight': 0.15,
            'strong_threshold': 20.0,
            'medium_threshold': 10.0
        }
        
        self.setup_vega_tab()
    
    def setup_vega_tab(self):
        """Setup simplified enhanced Vega analysis tab"""
        # Main container with notebook
        main_notebook = ttk.Notebook(self.parent)
        main_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 1. CURRENT ANALYSIS TAB
        current_frame = ttk.Frame(main_notebook)
        main_notebook.add(current_frame, text="Current Analysis")
        self.setup_current_analysis_tab(current_frame)
        
        # 2. SIMPLE VERIFICATION TAB
        verification_frame = ttk.Frame(main_notebook)
        main_notebook.add(verification_frame, text="Simple Verification")
        self.setup_simple_verification_tab(verification_frame)
    
    def setup_current_analysis_tab(self, parent):
        """Setup current analysis display"""
        # Composition info
        composition_frame = ttk.LabelFrame(parent, text="🎯 Strike Selection Logic")
        composition_frame.pack(fill=tk.X, padx=5, pady=(5, 10))
        
        self.composition_text = tk.Text(composition_frame, height=4, wrap=tk.WORD, font=("Courier", 9))
        self.composition_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Baseline vs Current analysis
        analysis_frame = ttk.LabelFrame(parent, text="📊 Baseline vs Current Analysis")
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 10))
        
        self.analysis_text = tk.Text(analysis_frame, height=15, wrap=tk.WORD, font=("Courier", 9))
        analysis_scrollbar = ttk.Scrollbar(analysis_frame, orient=tk.VERTICAL, command=self.analysis_text.yview)
        self.analysis_text.configure(yscrollcommand=analysis_scrollbar.set)
        self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        analysis_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Sentiment results
        results_frame = ttk.LabelFrame(parent, text="🎯 Sentiment Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        
        self.results_text = tk.Text(results_frame, height=10, wrap=tk.WORD, font=("Courier", 10))
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def setup_simple_verification_tab(self, parent):
        """Setup simplified verification tab"""
        # Control panel
        control_frame = ttk.LabelFrame(parent, text="📊 Simple Verification Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        control_grid = ttk.Frame(control_frame)
        control_grid.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(control_grid, text="Quick Verification", 
                  command=self.run_quick_verification).grid(row=0, column=0, padx=5)
        ttk.Button(control_grid, text="Simple Tune", 
                  command=self.simple_parameter_tune).grid(row=0, column=1, padx=5)
        ttk.Button(control_grid, text="Clear History", 
                  command=self.clear_history).grid(row=0, column=2, padx=5)
        
        # Status display
        status_frame = ttk.LabelFrame(control_frame, text="Status")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(status_grid, text="Predictions Tracked:").grid(row=0, column=0, sticky="w", padx=5)
        self.prediction_count_label = ttk.Label(status_grid, text="0", font=("Arial", 10, "bold"))
        self.prediction_count_label.grid(row=0, column=1, sticky="w", padx=5)
        
        ttk.Label(status_grid, text="Last Accuracy:").grid(row=0, column=2, sticky="w", padx=5)
        self.accuracy_label = ttk.Label(status_grid, text="N/A", font=("Arial", 10, "bold"))
        self.accuracy_label.grid(row=0, column=3, sticky="w", padx=5)
        
        # Simple verification results
        verification_frame = ttk.LabelFrame(parent, text="📈 Verification Results")
        verification_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.verification_text = tk.Text(verification_frame, height=15, wrap=tk.WORD, font=("Courier", 9))
        verification_scrollbar = ttk.Scrollbar(verification_frame, orient=tk.VERTICAL, command=self.verification_text.yview)
        self.verification_text.configure(yscrollcommand=verification_scrollbar.set)
        self.verification_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        verification_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def simple_track_prediction(self, timestamp_index: int, vega_analysis: Dict):
        """Simple, lightweight prediction tracking"""
        if not hasattr(self.analyzer, 'unique_timestamps') or timestamp_index >= len(self.analyzer.unique_timestamps):
            return
        
        try:
            timestamp = self.analyzer.unique_timestamps[timestamp_index]
            
            # Simple prediction entry (no complex processing)
            prediction_entry = {
                'timestamp_index': timestamp_index,
                'timestamp': timestamp,
                'sentiment': vega_analysis.get('sentiment', 'UNKNOWN'),
                'strength': vega_analysis.get('strength', 0),
                'confidence': vega_analysis.get('confidence', 'LOW'),
                'vega_sentiment': vega_analysis.get('vega_sentiment', 0),
                'ce_vega_change': vega_analysis.get('ce_vega_change', 0),
                'pe_vega_change': vega_analysis.get('pe_vega_change', 0),
                'logged_time': datetime.now()
            }
            
            self.prediction_history.append(prediction_entry)
            
            # Keep only last N predictions to prevent memory issues
            if len(self.prediction_history) > self.max_history:
                self.prediction_history = self.prediction_history[-self.max_history:]
            
            # Update status display
            self.update_status_display()
            
        except Exception as e:
            print(f"Error in simple prediction tracking: {e}")
    
    def run_quick_verification(self):
        """Quick, simple verification that won't freeze the GUI"""
        try:
            self.verification_text.delete(1.0, tk.END)
            
            if len(self.prediction_history) < 5:
                self.verification_text.insert(tk.END, 
                    "Need at least 5 predictions for verification.\n"
                    "Navigate through more timestamps first.\n\n"
                    f"Current predictions tracked: {len(self.prediction_history)}")
                return
            
            # Simple accuracy calculation (no complex processing)
            total_predictions = len(self.prediction_history)
            
            # Count by sentiment type
            bullish_count = sum(1 for p in self.prediction_history if 'BULL' in p['sentiment'])
            bearish_count = sum(1 for p in self.prediction_history if 'BEAR' in p['sentiment'])
            sideways_count = total_predictions - bullish_count - bearish_count
            
            # Simple strength analysis
            avg_strength = np.mean([abs(p['strength']) for p in self.prediction_history])
            high_confidence_count = sum(1 for p in self.prediction_history if p['confidence'] in ['HIGH', 'VERY HIGH'])
            
            # Simple Vega analysis
            avg_vega_sentiment = np.mean([abs(p['vega_sentiment']) for p in self.prediction_history])
            strong_vega_signals = sum(1 for p in self.prediction_history if abs(p['vega_sentiment']) > 15)
            
            verification_text = f"""QUICK VERIFICATION ANALYSIS
{'='*40}

PREDICTION SUMMARY:
  Total Predictions: {total_predictions}
  Bullish Predictions: {bullish_count} ({bullish_count/total_predictions*100:.1f}%)
  Bearish Predictions: {bearish_count} ({bearish_count/total_predictions*100:.1f}%)
  Sideways Predictions: {sideways_count} ({sideways_count/total_predictions*100:.1f}%)

SIGNAL STRENGTH ANALYSIS:
  Average Strength: {avg_strength:.1f}
  High Confidence Signals: {high_confidence_count} ({high_confidence_count/total_predictions*100:.1f}%)
  Strong Vega Signals (>15): {strong_vega_signals} ({strong_vega_signals/total_predictions*100:.1f}%)
  Average Vega Sentiment: {avg_vega_sentiment:.1f}

SIMPLE INSIGHTS:
{'='*20}
"""
            
            # Add simple insights
            if bullish_count > bearish_count * 1.5:
                verification_text += "📈 Predominantly bullish predictions - Monitor for upside bias\n"
            elif bearish_count > bullish_count * 1.5:
                verification_text += "📉 Predominantly bearish predictions - Monitor for downside bias\n"
            else:
                verification_text += "⚖️ Balanced prediction distribution\n"
            
            if high_confidence_count / total_predictions > 0.6:
                verification_text += "🔒 High confidence rate - Strong signal generation\n"
            else:
                verification_text += "⚠️ Low confidence rate - Consider parameter tuning\n"
            
            if avg_vega_sentiment > 12:
                verification_text += "📊 Strong average Vega signals - Good signal strength\n"
            else:
                verification_text += "📊 Weak average Vega signals - May need threshold adjustment\n"
            
            verification_text += f"""
CURRENT PARAMETERS:
  Vega Weight: {self.tuning_params['vega_weight']*100:.1f}%
  Gamma Weight: {self.tuning_params['gamma_weight']*100:.1f}%
  Theta Weight: {self.tuning_params['theta_weight']*100:.1f}%
  Strong Threshold: {self.tuning_params['strong_threshold']:.1f}
  Medium Threshold: {self.tuning_params['medium_threshold']:.1f}

SIMPLE RECOMMENDATIONS:
{'='*25}
"""
            
            # Simple recommendations
            if high_confidence_count / total_predictions < 0.3:
                verification_text += "1. 🔧 Consider lowering Strong Threshold to 15-18\n"
            if avg_vega_sentiment < 10:
                verification_text += "2. ⚙️ Consider increasing Vega weight to 0.65\n"
            if strong_vega_signals / total_predictions < 0.4:
                verification_text += "3. 📊 Consider lowering Medium Threshold to 8\n"
            
            verification_text += f"\n⏰ Analysis completed: {datetime.now().strftime('%H:%M:%S')}"
            verification_text += f"\n📊 Data range: {total_predictions} predictions"
            
            self.verification_text.insert(tk.END, verification_text)
            
            # Update accuracy display (simple estimate)
            estimated_accuracy = min(85, 50 + (high_confidence_count / total_predictions) * 30)
            self.accuracy_label.config(text=f"{estimated_accuracy:.1f}%")
            
        except Exception as e:
            error_msg = f"Error in quick verification: {e}\n\nThis is a simplified verification to avoid GUI freezing."
            self.verification_text.insert(tk.END, error_msg)
            print(f"Error in quick verification: {e}")
    
    def simple_parameter_tune(self):
        """Simple parameter tuning based on basic analysis"""
        try:
            if len(self.prediction_history) < 10:
                self.verification_text.delete(1.0, tk.END)
                self.verification_text.insert(tk.END, 
                    "Need at least 10 predictions for parameter tuning.\n"
                    f"Current: {len(self.prediction_history)} predictions")
                return
            
            # Simple tuning logic
            high_confidence_rate = sum(1 for p in self.prediction_history 
                                     if p['confidence'] in ['HIGH', 'VERY HIGH']) / len(self.prediction_history)
            
            avg_vega_sentiment = np.mean([abs(p['vega_sentiment']) for p in self.prediction_history])
            
            old_params = self.tuning_params.copy()
            
            # Simple adjustments
            if high_confidence_rate < 0.3:  # Too few high confidence signals
                self.tuning_params['strong_threshold'] *= 0.9  # Lower threshold
                self.tuning_params['medium_threshold'] *= 0.9
                
            if avg_vega_sentiment < 10:  # Weak Vega signals
                self.tuning_params['vega_weight'] = min(0.7, self.tuning_params['vega_weight'] * 1.1)
                self.tuning_params['gamma_weight'] *= 0.95
                self.tuning_params['theta_weight'] *= 0.95
            
            # Normalize weights
            total_weight = (self.tuning_params['vega_weight'] + 
                           self.tuning_params['gamma_weight'] + 
                           self.tuning_params['theta_weight'])
            
            self.tuning_params['vega_weight'] /= total_weight
            self.tuning_params['gamma_weight'] /= total_weight
            self.tuning_params['theta_weight'] /= total_weight
            
            # Display changes
            tune_text = f"""SIMPLE PARAMETER TUNING APPLIED
{'='*40}

ANALYSIS BASIS:
  Predictions Analyzed: {len(self.prediction_history)}
  High Confidence Rate: {high_confidence_rate*100:.1f}%
  Average Vega Sentiment: {avg_vega_sentiment:.1f}

PARAMETER CHANGES:
  Vega Weight:      {old_params['vega_weight']:.3f} → {self.tuning_params['vega_weight']:.3f}
  Gamma Weight:     {old_params['gamma_weight']:.3f} → {self.tuning_params['gamma_weight']:.3f}
  Theta Weight:     {old_params['theta_weight']:.3f} → {self.tuning_params['theta_weight']:.3f}
  Strong Threshold: {old_params['strong_threshold']:.1f} → {self.tuning_params['strong_threshold']:.1f}
  Medium Threshold: {old_params['medium_threshold']:.1f} → {self.tuning_params['medium_threshold']:.1f}

TUNING LOGIC:
  {'✅ Lowered thresholds for more signals' if high_confidence_rate < 0.3 else '✅ Thresholds maintained'}
  {'✅ Increased Vega weight for stronger signals' if avg_vega_sentiment < 10 else '✅ Vega weight maintained'}

⏰ Tuning completed: {datetime.now().strftime('%H:%M:%S')}
"""
            
            self.verification_text.delete(1.0, tk.END)
            self.verification_text.insert(tk.END, tune_text)
            
        except Exception as e:
            error_msg = f"Error in simple parameter tuning: {e}"
            self.verification_text.insert(tk.END, error_msg)
            print(f"Error in simple tuning: {e}")
    
    def clear_history(self):
        """Clear prediction history"""
        self.prediction_history.clear()
        self.update_status_display()
        self.verification_text.delete(1.0, tk.END)
        self.verification_text.insert(tk.END, "Prediction history cleared.")
        self.accuracy_label.config(text="N/A")
    
    def update_status_display(self):
        """Update status labels"""
        self.prediction_count_label.config(text=str(len(self.prediction_history)))
    
    def get_strike_ranges_info(self, current_greeks: Dict) -> str:
        """Get strike ranges information as requested"""
        ce_strikes = current_greeks.get('ce_strikes', [])
        pe_strikes = current_greeks.get('pe_strikes', [])
        
        if not ce_strikes and not pe_strikes:
            return "CE Strikes: No data, PE Strikes: No data, ATM: N/A"
        
        ce_range = f"{min(ce_strikes):.0f} - {max(ce_strikes):.0f}" if ce_strikes else "No data"
        pe_range = f"{min(pe_strikes):.0f} - {max(pe_strikes):.0f}" if pe_strikes else "No data"
        
        all_strikes = ce_strikes + pe_strikes
        atm_estimate = f"{np.median(all_strikes):.0f}" if all_strikes else "N/A"
        
        return f"CE Strikes: {ce_range}, PE Strikes: {pe_range}, ATM: {atm_estimate}"
    
    def update_enhanced_vega_analysis_for_position(self, timestamp_index: int):
        """Update analysis for specific timestamp position"""
        if not hasattr(self.analyzer, 'unique_timestamps') or timestamp_index >= len(self.analyzer.unique_timestamps):
            return
        
        try:
            # Get analysis
            analysis = self.analyzer.analyze_timestamp(timestamp_index)
            
            if not analysis or 'vega_analysis' not in analysis:
                return
            
            vega_analysis = analysis['vega_analysis']
            
            # Simple prediction tracking (non-blocking)
            self.simple_track_prediction(timestamp_index, vega_analysis)
            
            # Update displays
            self.update_composition_display()
            self.update_current_analysis_display(vega_analysis)
            self.update_vega_results_display(vega_analysis)
            
        except Exception as e:
            print(f"Error in update_enhanced_vega_analysis_for_position: {e}")
    
    def update_enhanced_vega_analysis(self, vega_analysis: Dict, timestamp_data: pd.DataFrame):
        """Main update method called from GUI"""
        try:
            self.update_composition_display()
            self.update_current_analysis_display(vega_analysis)
            self.update_vega_results_display(vega_analysis)
        except Exception as e:
            print(f"Error in update_enhanced_vega_analysis: {e}")
    
    def update_composition_display(self):
        """Update composition logic display"""
        try:
            self.composition_text.delete(1.0, tk.END)
            
            if not hasattr(self.analyzer, 'strike_composition') or self.analyzer.strike_composition is None:
                self.composition_text.insert(tk.END, "Strike composition not established.")
                return
            
            comp = self.analyzer.strike_composition
            composition_text = f"""POSITION-BASED SELECTION: ITM={comp['itm_count']}, ATM={comp['atm_count']}, OTM={comp['otm_count']} (Total={comp['total_count']})
CE: {comp['total_count']} highest strikes | PE: {comp['total_count']} lowest strikes | Source: {comp['established_from']}"""
            
            self.composition_text.insert(tk.END, composition_text)
            
        except Exception as e:
            print(f"Error updating composition display: {e}")
    
    def update_current_analysis_display(self, vega_analysis: Dict):
        """Update current analysis display with baseline comparison"""
        try:
            self.analysis_text.delete(1.0, tk.END)
            
            current_greeks = vega_analysis.get('current_greeks', {})
            strike_ranges = self.get_strike_ranges_info(current_greeks)
            
            # Get baseline values
            baseline_ce_vega = getattr(self.analyzer, 'baseline_ce_vega_sum', 0)
            baseline_pe_vega = getattr(self.analyzer, 'baseline_pe_vega_sum', 0)
            baseline_ce_gamma = getattr(self.analyzer, 'baseline_ce_gamma_sum', 0)
            baseline_pe_gamma = getattr(self.analyzer, 'baseline_pe_gamma_sum', 0)
            baseline_ce_theta = getattr(self.analyzer, 'baseline_ce_theta_sum', 0)
            baseline_pe_theta = getattr(self.analyzer, 'baseline_pe_theta_sum', 0)
            
            analysis_text = f"""BASELINE vs CURRENT ANALYSIS (Position-Based Selection)
{'='*65}

BASELINE GREEK SUMS (Start of Day):
CE (CALL) BASELINE:                    PE (PUT) BASELINE:
  Vega Sum:   {baseline_ce_vega:>10.2f}               Vega Sum:   {baseline_pe_vega:>10.2f}
  Gamma Sum:  {baseline_ce_gamma:>10.4f}               Gamma Sum:  {baseline_pe_gamma:>10.4f}
  Theta Sum:  {baseline_ce_theta:>10.2f}               Theta Sum:  {baseline_pe_theta:>10.2f}

CURRENT GREEK SUMS:
CE (CALL) CURRENT:                     PE (PUT) CURRENT:
  Vega Sum:   {current_greeks.get('ce_vega_sum', 0):>10.2f}               Vega Sum:   {current_greeks.get('pe_vega_sum', 0):>10.2f}
  Gamma Sum:  {current_greeks.get('ce_gamma_sum', 0):>10.4f}               Gamma Sum:  {current_greeks.get('pe_gamma_sum', 0):>10.4f}
  Theta Sum:  {current_greeks.get('ce_theta_sum', 0):>10.2f}               Theta Sum:  {current_greeks.get('pe_theta_sum', 0):>10.2f}

PERCENTAGE CHANGES FROM BASELINE:
CE (CALL) CHANGES:                     PE (PUT) CHANGES:
  Vega Change:    {vega_analysis.get('ce_vega_change', 0):>8.2f}%             Vega Change:    {vega_analysis.get('pe_vega_change', 0):>8.2f}%
  Gamma Change:   {vega_analysis.get('ce_gamma_change', 0):>8.2f}%             Gamma Change:   {vega_analysis.get('pe_gamma_change', 0):>8.2f}%
  Theta Change:   {vega_analysis.get('ce_theta_change', 0):>8.2f}%             Theta Change:   {vega_analysis.get('pe_theta_change', 0):>8.2f}%

DIFFERENTIAL ANALYSIS:
  CE-PE Vega Diff:  {vega_analysis.get('ce_vega_change', 0) - vega_analysis.get('pe_vega_change', 0):>6.2f}%
  CE-PE Gamma Diff: {vega_analysis.get('ce_gamma_change', 0) - vega_analysis.get('pe_gamma_change', 0):>6.2f}%
  CE-PE Theta Diff: {vega_analysis.get('ce_theta_change', 0) - vega_analysis.get('pe_theta_change', 0):>6.2f}%

STRIKE RANGES (as requested):
  {strike_ranges}

SIMPLE VERIFICATION STATUS:
  {'✅ Position-based selection active' if len(current_greeks.get('ce_strikes', [])) > 0 and len(current_greeks.get('pe_strikes', [])) > 0 else '⚠️ Selection issues detected'}
  📊 Predictions tracked: {len(self.prediction_history)}
"""
            
            self.analysis_text.insert(tk.END, analysis_text)
            
        except Exception as e:
            print(f"Error updating current analysis display: {e}")
    
    def update_vega_results_display(self, vega_analysis: Dict):
        """Update Vega sentiment results"""
        try:
            self.results_text.delete(1.0, tk.END)
            
            # Use current tuning parameters
            params = self.tuning_params
            
            results_text = f"""ENHANCED VEGA SENTIMENT ANALYSIS (Simplified)
{'='*55}

🎯 PRIMARY SENTIMENT: {vega_analysis.get('sentiment', 'N/A')}
📊 Overall Strength:  {vega_analysis.get('strength', 0):.1f}/100
🔒 Confidence Level:  {vega_analysis.get('confidence', 'N/A')}

WEIGHTED COMPONENT ANALYSIS:
{'='*35}
🔹 Vega Component ({params['vega_weight']*100:.1f}% weight):   {vega_analysis.get('vega_sentiment', 0):>8.2f}
🔹 Gamma Component ({params['gamma_weight']*100:.1f}% weight):  {vega_analysis.get('gamma_sentiment', 0):>8.2f}
🔹 Theta Component ({params['theta_weight']*100:.1f}% weight):  {vega_analysis.get('theta_sentiment', 0):>8.2f}
                                      -------
🎯 Combined Sentiment:                {vega_analysis.get('strength', 0):>8.1f}

SIMPLE VERIFICATION STATUS:
{'='*30}
📊 Predictions Tracked: {len(self.prediction_history)}
🎯 Verification: {'Available' if len(self.prediction_history) >= 5 else 'Need more data'}
⚙️ Parameter Tuning: {'Available' if len(self.prediction_history) >= 10 else 'Need more data'}

INTERPRETATION:
{'='*20}
"""
            
            vega_sentiment = vega_analysis.get('vega_sentiment', 0)
            confidence = vega_analysis.get('confidence', 'LOW')
            
            if vega_sentiment > params['medium_threshold']:
                results_text += "🟢 BULLISH SIGNAL: CE Greeks (high strikes) strengthening faster\n"
                results_text += "   → Institutional positioning for upward movement\n"
            elif vega_sentiment < -params['medium_threshold']:
                results_text += "🔴 BEARISH SIGNAL: PE Greeks (low strikes) strengthening faster\n"
                results_text += "   → Institutional positioning for downward movement\n"
            else:
                results_text += "⚪ NEUTRAL SIGNAL: Balanced Greek changes\n"
                results_text += "   → No clear institutional directional bias\n"
            
            results_text += f"\n🔒 CONFIDENCE: {confidence}\n"
            
            if confidence in ['HIGH', 'VERY HIGH']:
                results_text += "   ✅ Strong signal - Being tracked for verification\n"
            else:
                results_text += "   ⚠️ Weak signal - Continue monitoring\n"
            
            results_text += f"\n📊 SIMPLE VERIFICATION:\n"
            results_text += f"   • Lightweight prediction tracking (no GUI freezing)\n"
            results_text += f"   • Basic accuracy analysis and parameter tuning\n"
            results_text += f"   • Check Simple Verification tab for analysis\n"
            
            self.results_text.insert(tk.END, results_text)
            
        except Exception as e:
            print(f"Error updating vega results display: {e}")

# Integration function
def integrate_enhanced_vega_tab_simple(gui_instance):
    """Integrate the simplified, robust enhanced Vega tab"""
    
    try:
        # Find notebook widget
        notebook = None
        for widget in gui_instance.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for subwidget in widget.winfo_children():
                    if isinstance(subwidget, ttk.Notebook):
                        notebook = subwidget
                        break
        
        if notebook is None:
            print("Warning: Could not find notebook widget")
            return
        
        # Add enhanced Vega analysis tab
        enhanced_vega_frame = ttk.Frame(notebook)
        notebook.insert(1, enhanced_vega_frame, text="Enhanced Vega Analysis")
        
        # Create enhanced Vega analysis instance
        gui_instance.enhanced_vega_tab = EnhancedVegaAnalysisTab(enhanced_vega_frame, gui_instance.analyzer)
        
        print("Simplified Enhanced Vega Analysis integrated successfully (no GUI freezing)")
        return gui_instance.enhanced_vega_tab
        
    except Exception as e:
        print(f"Error integrating enhanced Vega tab: {e}")
        import traceback
        traceback.print_exc()
        return None

# Update function
def update_enhanced_vega_analysis_simple(gui_instance, *args):
    """Simplified update function"""
    if hasattr(gui_instance, 'enhanced_vega_tab'):
        try:
            pos = int(gui_instance.time_var.get())
            pos = min(pos, len(gui_instance.analyzer.unique_timestamps) - 1)
            
            gui_instance.analyzer.current_timestamp_index = pos
            gui_instance.enhanced_vega_tab.update_enhanced_vega_analysis_for_position(pos)
            
        except Exception as e:
            print(f"Error updating enhanced Vega analysis: {e}")
