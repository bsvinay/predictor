# Fixed AdvancedOptionsAnalyzer with correct strike selection logic

import pandas as pd
from typing import Dict, List, Tuple, Optional
from analyzer.options_analyzer import OptionsAnalyzer

class AdvancedOptionsAnalyzer(OptionsAnalyzer):
    def __init__(self):
        super().__init__()
        
        # Enhanced analysis tracking
        self.baseline_data = None
        self.baseline_timestamp = None
        self.baseline_ce_vega_sum = 0
        self.baseline_pe_vega_sum = 0
        self.baseline_ce_gamma_sum = 0
        self.baseline_pe_gamma_sum = 0
        self.baseline_ce_theta_sum = 0
        self.baseline_pe_theta_sum = 0
        self.baseline_strike_range = {'ce_strikes': [], 'pe_strikes': []}
        
        # FIXED: Store the established strike counts (not actual strikes)
        self.strike_composition = None  # Will store the counts: {itm_count, atm_count, otm_count, total_count}
        
        # Enhanced analysis history
        self.vega_sentiment_history = []
        self.gamma_sentiment_history = []
        self.theta_sentiment_history = []
    
    def establish_strike_composition_once(self, baseline_data: pd.DataFrame):
        """STEP 1: Use CE Delta 0.1-0.6 ONLY to establish strike counts"""
        if self.strike_composition is not None:
            return self.strike_composition
        
        try:
            # Filter baseline data using CE Delta 0.1-0.6 criteria
            ce_delta_filtered = baseline_data[
                (baseline_data['Strike Price'] > 0) & 
                (baseline_data['Delta'] >= 0.1) & 
                (baseline_data['Delta'] <= 0.6)
            ].copy()
            
            if len(ce_delta_filtered) == 0:
                print("Warning: No strikes found in CE Delta 0.1-0.6 range")
                return None
            
            # Sort by Delta to analyze the composition
            ce_delta_filtered = ce_delta_filtered.sort_values('Delta', ascending=False)
            
            # Find ATM (Delta closest to 0.5)
            ce_delta_filtered['delta_diff'] = abs(ce_delta_filtered['Delta'] - 0.5)
            atm_row = ce_delta_filtered.loc[ce_delta_filtered['delta_diff'].idxmin()]
            atm_delta = atm_row['Delta']
            
            # Classify into ITM, ATM, OTM based on Delta relative to 0.5
            itm_strikes = ce_delta_filtered[ce_delta_filtered['Delta'] > 0.55]  # ITM calls (higher delta)
            atm_strikes = ce_delta_filtered[abs(ce_delta_filtered['Delta'] - 0.5) <= 0.05]  # ATM calls
            otm_strikes = ce_delta_filtered[ce_delta_filtered['Delta'] < 0.45]  # OTM calls (lower delta)
            
            # Count the composition
            itm_count = len(itm_strikes)
            atm_count = max(1, len(atm_strikes))  # At least 1 ATM
            otm_count = len(otm_strikes)
            total_count = itm_count + atm_count + otm_count
            
            # Store the composition (COUNTS ONLY, not actual strikes)
            self.strike_composition = {
                'itm_count': itm_count,
                'atm_count': atm_count, 
                'otm_count': otm_count,
                'total_count': total_count,
                'established_from': 'CE Delta 0.1-0.6 range'
            }
            
            print(f"Strike composition established: ITM={itm_count}, ATM={atm_count}, OTM={otm_count}, Total={total_count}")
            
            return self.strike_composition
            
        except Exception as e:
            print(f"Error establishing strike composition: {e}")
            return None
    
    def select_strikes_by_position(self, timestamp_data: pd.DataFrame, option_type: str) -> pd.DataFrame:
        """STEP 2: Select strikes by position using established counts"""
        if self.strike_composition is None:
            return pd.DataFrame()
        
        if timestamp_data.empty:
            return pd.DataFrame()
        
        # Get all available strikes for this timestamp, sorted by strike price
        available_strikes = timestamp_data[
            timestamp_data['Strike Price'] > 0
        ].copy().sort_values('Strike Price')
        
        if len(available_strikes) == 0:
            return pd.DataFrame()
        
        total_needed = self.strike_composition['total_count']
        
        if option_type == 'CE':
            # For CE: Take strikes in REVERSE order (highest strikes first)
            # Higher strikes = OTM calls, Lower strikes = ITM calls
            selected_strikes = available_strikes.sort_values('Strike Price', ascending=False).head(total_needed)
            
        else:  # PE
            # For PE: Take strikes in INCREASING order (lowest strikes first) 
            # Lower strikes = OTM puts, Higher strikes = ITM puts
            selected_strikes = available_strikes.sort_values('Strike Price', ascending=True).head(total_needed)
        
        return selected_strikes
    
    def calculate_vega_sentiment_greeks(self, timestamp_data: pd.DataFrame) -> Dict:
        """Calculate Greek sums using corrected position-based selection"""
        if timestamp_data.empty:
            return {
                'ce_vega_sum': 0, 'pe_vega_sum': 0,
                'ce_gamma_sum': 0, 'pe_gamma_sum': 0, 
                'ce_theta_sum': 0, 'pe_theta_sum': 0,
                'ce_strikes': [], 'pe_strikes': []
            }
        
        try:
            # Establish composition if not done (only done once)
            if self.strike_composition is None:
                self.establish_strike_composition_once(timestamp_data)
            
            if self.strike_composition is None:
                return {
                    'ce_vega_sum': 0, 'pe_vega_sum': 0,
                    'ce_gamma_sum': 0, 'pe_gamma_sum': 0, 
                    'ce_theta_sum': 0, 'pe_theta_sum': 0,
                    'ce_strikes': [], 'pe_strikes': []
                }
            
            # Select CE strikes by position (reverse order - highest strikes first)
            ce_strikes = self.select_strikes_by_position(timestamp_data, 'CE')
            
            # Select PE strikes by position (forward order - lowest strikes first)
            pe_strikes = self.select_strikes_by_position(timestamp_data, 'PE')
            
            # Calculate Greek sums
            ce_vega_sum = ce_strikes['Vega'].sum() if not ce_strikes.empty else 0
            pe_vega_sum = pe_strikes['Vega'].sum() if not pe_strikes.empty else 0
            
            ce_gamma_sum = ce_strikes['Gamma'].sum() if not ce_strikes.empty else 0
            pe_gamma_sum = pe_strikes['Gamma'].sum() if not pe_strikes.empty else 0
            
            ce_theta_sum = ce_strikes['Theta'].sum() if not ce_strikes.empty else 0
            pe_theta_sum = pe_strikes['Theta'].sum() if not pe_strikes.empty else 0
            
            # Get strike lists
            ce_strike_prices = ce_strikes['Strike Price'].tolist() if not ce_strikes.empty else []
            pe_strike_prices = pe_strikes['Strike Price'].tolist() if not pe_strikes.empty else []
            
            return {
                'ce_vega_sum': ce_vega_sum,
                'pe_vega_sum': pe_vega_sum,
                'ce_gamma_sum': ce_gamma_sum,
                'pe_gamma_sum': pe_gamma_sum,
                'ce_theta_sum': ce_theta_sum,
                'pe_theta_sum': pe_theta_sum,
                'ce_strikes': ce_strike_prices,
                'pe_strikes': pe_strike_prices
            }
            
        except Exception as e:
            print(f"Error in calculate_vega_sentiment_greeks: {e}")
            return {
                'ce_vega_sum': 0, 'pe_vega_sum': 0,
                'ce_gamma_sum': 0, 'pe_gamma_sum': 0, 
                'ce_theta_sum': 0, 'pe_theta_sum': 0,
                'ce_strikes': [], 'pe_strikes': []
            }
    
    def set_baseline(self):
        """Set baseline data from first timestamp"""
        if len(self.unique_timestamps) == 0:
            print("Warning: No timestamps available for baseline")
            return
        
        self.baseline_timestamp = self.unique_timestamps[0]
        self.baseline_data = self.get_data_for_timestamp(self.baseline_timestamp)
        
        if self.baseline_data.empty:
            print("Warning: No baseline data found!")
            return
        
        print(f"Setting baseline at: {self.baseline_timestamp}")
        
        # STEP 1: Establish strike composition using CE Delta criteria (done once)
        self.establish_strike_composition_once(self.baseline_data)
        
        # STEP 2: Calculate baseline Greek sums using position-based selection
        baseline_greeks = self.calculate_vega_sentiment_greeks(self.baseline_data)
        
        self.baseline_ce_vega_sum = baseline_greeks['ce_vega_sum']
        self.baseline_pe_vega_sum = baseline_greeks['pe_vega_sum']
        self.baseline_ce_gamma_sum = baseline_greeks['ce_gamma_sum'] 
        self.baseline_pe_gamma_sum = baseline_greeks['pe_gamma_sum']
        self.baseline_ce_theta_sum = baseline_greeks['ce_theta_sum']
        self.baseline_pe_theta_sum = baseline_greeks['pe_theta_sum']
        
        self.baseline_strike_range = {
            'ce_strikes': baseline_greeks['ce_strikes'],
            'pe_strikes': baseline_greeks['pe_strikes']
        }
        
        print(f"Baseline established using position-based selection:")
        print(f"  CE Vega Sum: {self.baseline_ce_vega_sum:.2f} (from {len(baseline_greeks['ce_strikes'])} highest strikes)")
        print(f"  PE Vega Sum: {self.baseline_pe_vega_sum:.2f} (from {len(baseline_greeks['pe_strikes'])} lowest strikes)")
        print(f"  Composition: ITM={self.strike_composition['itm_count']}, ATM={self.strike_composition['atm_count']}, OTM={self.strike_composition['otm_count']}")
    
    def safe_percent_change(self, baseline, current):
        """Safely calculate percentage change"""
        if baseline == 0:
            return 0 if current == 0 else (100 if current > 0 else -100)
        return ((current - baseline) / baseline) * 100
    
    def calculate_vega_based_sentiment(self, timestamp_data: pd.DataFrame) -> Dict:
        """Calculate sentiment using corrected position-based methodology"""
        if self.baseline_data is None or timestamp_data.empty:
            return {
                'sentiment': 'NO DATA',
                'strength': 0,
                'vega_sentiment': 0,
                'gamma_sentiment': 0,
                'theta_sentiment': 0,
                'confidence': 'LOW',
                'ce_vega_change': 0,
                'pe_vega_change': 0,
                'current_greeks': {}
            }
        
        # Get current Greek sums using position-based selection
        current_greeks = self.calculate_vega_sentiment_greeks(timestamp_data)
        
        # Calculate percentage changes from baseline
        ce_vega_change = self.safe_percent_change(
            self.baseline_ce_vega_sum, current_greeks['ce_vega_sum']
        )
        pe_vega_change = self.safe_percent_change(
            self.baseline_pe_vega_sum, current_greeks['pe_vega_sum']
        )
        
        ce_gamma_change = self.safe_percent_change(
            self.baseline_ce_gamma_sum, current_greeks['ce_gamma_sum']
        )
        pe_gamma_change = self.safe_percent_change(
            self.baseline_pe_gamma_sum, current_greeks['pe_gamma_sum']
        )
        
        ce_theta_change = self.safe_percent_change(
            self.baseline_ce_theta_sum, current_greeks['ce_theta_sum']
        )
        pe_theta_change = self.safe_percent_change(
            self.baseline_pe_theta_sum, current_greeks['pe_theta_sum']
        )
        
        # Calculate sentiment scores with weightages
        vega_diff = ce_vega_change - pe_vega_change
        vega_sentiment = vega_diff * 0.6
        
        gamma_diff = ce_gamma_change - pe_gamma_change  
        gamma_sentiment = gamma_diff * 0.25
        
        theta_diff = ce_theta_change - pe_theta_change
        theta_sentiment = theta_diff * 0.15
        
        # Combined sentiment strength
        total_sentiment = vega_sentiment + gamma_sentiment + theta_sentiment
        strength = max(-100, min(100, total_sentiment))
        
        # Determine sentiment category
        sentiment = self.categorize_sentiment(strength, vega_sentiment)
        
        # Determine confidence level
        confidence = self.calculate_confidence(
            abs(vega_diff), abs(gamma_diff), abs(theta_diff)
        )
        
        return {
            'sentiment': sentiment,
            'strength': strength,
            'vega_sentiment': vega_sentiment,
            'gamma_sentiment': gamma_sentiment, 
            'theta_sentiment': theta_sentiment,
            'confidence': confidence,
            'ce_vega_change': ce_vega_change,
            'pe_vega_change': pe_vega_change,
            'ce_gamma_change': ce_gamma_change,
            'pe_gamma_change': pe_gamma_change,
            'ce_theta_change': ce_theta_change,
            'pe_theta_change': pe_theta_change,
            'current_greeks': current_greeks
        }
    
    def categorize_sentiment(self, strength, vega_sentiment):
        """Categorize sentiment based on strength and Vega dominance"""
        if abs(vega_sentiment) < 5:
            return "SIDEWAYS"
        elif vega_sentiment > 0:
            if vega_sentiment > 20:
                return "STRONG BULLISH"
            elif vega_sentiment > 10:
                return "BULLISH"
            else:
                return "WEAK BULLISH"
        else:
            if vega_sentiment < -20:
                return "STRONG BEARISH"
            elif vega_sentiment < -10:
                return "BEARISH"
            else:
                return "WEAK BEARISH"
    
    def calculate_confidence(self, vega_diff, gamma_diff, theta_diff):
        """Calculate confidence level based on Greek alignment"""
        alignment_score = 0
        
        if abs(vega_diff) > 10:
            alignment_score += 3
        elif abs(vega_diff) > 5:
            alignment_score += 2
        else:
            alignment_score += 1
            
        if abs(gamma_diff) > 8:
            alignment_score += 2
        elif abs(gamma_diff) > 4:
            alignment_score += 1
            
        if abs(theta_diff) > 6:
            alignment_score += 1
        
        if alignment_score >= 7:
            return "VERY HIGH"
        elif alignment_score >= 5:
            return "HIGH"
        elif alignment_score >= 3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def analyze_timestamp(self, timestamp_index: int) -> Dict:
        """Enhanced analysis: adds Vega methodology to original analysis"""
        # Get original analysis
        original_analysis = super().analyze_timestamp(timestamp_index)
        
        if timestamp_index >= len(self.unique_timestamps):
            return original_analysis
        
        timestamp = self.unique_timestamps[timestamp_index]
        timestamp_data = self.get_data_for_timestamp(timestamp)
        
        if timestamp_data.empty:
            return original_analysis
        
        # Add Vega-based analysis
        vega_analysis = self.calculate_vega_based_sentiment(timestamp_data)
        
        # Combine both analyses
        return {
            **original_analysis,  # Keep all original results
            'vega_analysis': vega_analysis,  # Add new Vega results
        }
