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

class OptionsAnalyzer:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.unique_timestamps = []
        self.sentiment_history = []
        self.strength_history = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load options chain data from CSV file"""
        try:
            # Parse filename to extract metadata
            filename = os.path.basename(file_path)
            pattern = r'([A-Z]+)_(\d{2}[A-Z]{3}\d{2})_(\d{2}-\d{2}-\d{4})_(\d+min)\.csv'
            match = re.match(pattern, filename)
            
            if match:
                index_name, expiry, date, interval = match.groups()
                print(f"Loading {index_name} data for {date}, expiry: {expiry}, interval: {interval}")
            
            # Load the CSV
            df = pd.read_csv(file_path)
            
            # Clean column names (remove extra spaces)
            df.columns = df.columns.str.strip()
            
            # Debug: print first few column names and sample data
            print("Columns:", df.columns[:10].tolist())
            print("First row sample:", df.iloc[0][:5].tolist())
            
            # Find the correct timestamp column
            timestamp_col = None
            if 'Time' in df.columns:
                # Use the 'Time' column which has full datetime
                timestamp_col = 'Time'
                print("Using 'Time' column for timestamps")
            elif 'Timestamp' in df.columns:
                # Check if Timestamp column has full datetime
                sample_ts = str(df['Timestamp'].iloc[0])
                if len(sample_ts) > 10:  # Full datetime should be longer
                    timestamp_col = 'Timestamp'
                    print("Using 'Timestamp' column for timestamps")
            
            if timestamp_col:
                # Convert timestamp to datetime with multiple format attempts
                formats_to_try = [
                    '%d-%b-%Y %H:%M:%S',  # 05-Jun-2025 09:15:59
                    '%Y-%m-%d %H:%M:%S',  # 2025-06-05 09:15:59
                    '%d/%m/%Y %H:%M:%S',  # 05/06/2025 09:15:59
                ]
                
                df['Parsed_Timestamp'] = None
                for fmt in formats_to_try:
                    try:
                        df['Parsed_Timestamp'] = pd.to_datetime(df[timestamp_col], format=fmt, errors='coerce')
                        valid_count = df['Parsed_Timestamp'].notna().sum()
                        if valid_count > 0:
                            print(f"Successfully parsed {valid_count} timestamps using format: {fmt}")
                            break
                    except:
                        continue
                
                # If none of the specific formats worked, try general parsing
                if df['Parsed_Timestamp'].isna().all():
                    print("Trying general datetime parsing...")
                    df['Parsed_Timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
                
                # Use the parsed timestamp
                df['Timestamp'] = df['Parsed_Timestamp']
                df.drop('Parsed_Timestamp', axis=1, inplace=True)
            else:
                print("Warning: No suitable timestamp column found!")
                return None
            
            # Function to clean numeric values (remove commas and quotes)
            def clean_numeric(value):
                if pd.isna(value) or value == '-' or value == '':
                    return 0
                # Convert to string and clean
                str_val = str(value).replace('"', '').replace(',', '').strip()
                try:
                    return float(str_val)
                except:
                    return 0
            
            # Clean ALL numeric columns - be more comprehensive
            numeric_patterns = ['Price', 'LTP', 'Volume', 'OI', 'IV', 'Delta', 'Gamma', 
                              'Theta', 'Vega', 'PCR', 'VWAP', 'Bid', 'Ask', 'Chg', '%']
            
            for col in df.columns:
                # Skip the timestamp column
                if col == 'Timestamp':
                    continue
                    
                # Check if column likely contains numeric data
                if any(pattern in col for pattern in numeric_patterns):
                    df[col] = df[col].apply(clean_numeric)
                # Also clean columns that look like they contain numbers
                elif df[col].dtype == 'object':
                    # Sample a few values to see if they look numeric
                    sample_vals = df[col].dropna().head(5).astype(str)
                    if len(sample_vals) > 0:
                        # Check if values contain commas and digits (Indian number format)
                        if any(',' in str(val) and any(c.isdigit() for c in str(val)) for val in sample_vals):
                            df[col] = df[col].apply(clean_numeric)
            
            # Get unique timestamps
            self.unique_timestamps = sorted(df['Timestamp'].dropna().unique())
            
            self.data = df
            print(f"Loaded {len(df)} rows with {len(self.unique_timestamps)} unique timestamps")
            
            if len(self.unique_timestamps) > 0:
                print(f"First timestamp: {self.unique_timestamps[0]}")
                print(f"Last timestamp: {self.unique_timestamps[-1]}")
            else:
                print("ERROR: No valid timestamps found!")
                print("Sample timestamp values:", df[timestamp_col].head().tolist())
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_data_for_timestamp(self, timestamp) -> pd.DataFrame:
        """Get all strikes data for a specific timestamp"""
        if self.data is None:
            return pd.DataFrame()
        
        return self.data[self.data['Timestamp'] == timestamp].copy()
    
    def identify_strike_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify ATM, ITM, OTM strikes"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Find ATM strike (middle strike in the chain)
        strikes = sorted(df['Strike Price'].dropna().unique())
        if len(strikes) == 0:
            return df
        
        middle_index = len(strikes) // 2
        atm_strike = strikes[middle_index]
        
        def classify_strike(strike):
            if abs(strike - atm_strike) <= 25:  # ATM range (within 25 points)
                return 'ATM'
            elif strike < atm_strike:
                return 'ITM-Call/OTM-Put'
            else:
                return 'OTM-Call/ITM-Put'
        
        df['Strike_Type'] = df['Strike Price'].apply(classify_strike)
        
        return df
    
    def calculate_sentiment_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate sentiment indicators for a timestamp"""
        if df.empty:
            return {}
        
        indicators = {}
        
        # Aggregate metrics across all strikes for this timestamp
        indicators['total_oi_change'] = df['OI Chg'].sum()
        indicators['total_volume_change'] = df['Volume Chg'].sum() if 'Volume Chg' in df.columns else 0
        indicators['avg_delta'] = df['Delta'].mean()
        indicators['avg_gamma'] = df['Gamma'].mean()
        indicators['avg_theta'] = df['Theta'].mean()
        indicators['avg_vega'] = df['Vega'].mean()
        
        # PCR values (same for all rows in timestamp)
        indicators['pcr_oi'] = df['PCR-OI'].iloc[0] if len(df) > 0 else 1
        indicators['pcr_vol'] = df['PCR-Vol'].iloc[0] if len(df) > 0 else 1
        
        # Volume and OI totals
        indicators['total_volume'] = df['Volume'].sum()
        indicators['total_oi'] = df['OI'].sum()
        
        return indicators
    
    def determine_sentiment(self, indicators: Dict) -> Tuple[str, int]:
        """Determine market sentiment and strength based on indicators"""
        sentiment_score = 0
        sentiment = "SIDEWAYS"
        
        # PCR Analysis (Put-Call Ratio)
        pcr_oi = indicators.get('pcr_oi', 1)
        if pcr_oi > 1.2:
            sentiment_score += 15  # Bullish (more puts = bearish expectation, contrarian bullish)
        elif pcr_oi < 0.8:
            sentiment_score -= 15  # Bearish
        
        # OI Change Analysis
        oi_change = indicators.get('total_oi_change', 0)
        if oi_change > 5000000:  # Large positive OI change
            sentiment_score += 10
        elif oi_change < -5000000:  # Large negative OI change
            sentiment_score -= 10
        
        # Volume Analysis
        volume_change = indicators.get('total_volume_change', 0)
        if volume_change > 0:
            sentiment_score += 5
        else:
            sentiment_score -= 5
        
        # Greeks Analysis
        avg_delta = indicators.get('avg_delta', 0)
        if avg_delta > 0.5:
            sentiment_score += 10
        elif avg_delta < 0.3:
            sentiment_score -= 10
        
        avg_gamma = indicators.get('avg_gamma', 0)
        if abs(avg_gamma) > 0.002:
            sentiment_score += 5  # High gamma suggests volatility
        
        # Determine sentiment category
        if sentiment_score > 25:
            sentiment = "STRONG BULLISH"
        elif sentiment_score > 15:
            sentiment = "BULLISH"
        elif sentiment_score > 5:
            sentiment = "WEAK BULLISH"
        elif sentiment_score < -25:
            sentiment = "STRONG BEARISH"
        elif sentiment_score < -15:
            sentiment = "BEARISH"
        elif sentiment_score < -5:
            sentiment = "WEAK BEARISH"
        else:
            sentiment = "SIDEWAYS"
        
        # Additional pattern detection
        if abs(sentiment_score) > 30:
            if sentiment_score > 0:
                sentiment = "BREAKOUT"
            else:
                sentiment = "BREAKDOWN"
        
        # Normalize strength to -50 to 50 range
        strength = max(-50, min(50, sentiment_score))
        
        return sentiment, strength
    
    def calculate_advanced_range(self, timestamp_data):
        """Calculate range using advanced Delta-Range methodology (User's approach)"""
        if timestamp_data.empty:
            return {'low': 0, 'high': 0, 'current': 0, 'atm': 0, 'analysis': "No data", 
                   'greekSum': 0, 'pressure': 'BALANCED', 'confidence': 'LOW', 'greekComposition': {}}
        
        # Step 1: Extract and clean data with proper Delta scaling
        valid_strikes = []
        for _, row in timestamp_data.iterrows():
            strike = row.get('Strike Price', 0)
            delta = row.get('Delta', 0)
            
            # Scale Delta to 0-1 range if needed (handle various scaling)
            if delta > 1:
                if delta > 100:
                    delta = delta / 1000  # Scale from 1000
                else:
                    delta = delta / 100   # Scale from 100
            
            if strike > 0 and delta > 0:
                vega = abs(row.get('Vega', 0))
                theta = abs(row.get('Theta', 0))
                gamma = abs(row.get('Gamma', 0))
                ce_oi = row.get('OI', 0)
                ce_oi_chg = row.get('OI Chg', 0)
                
                valid_strikes.append({
                    'strike': strike,
                    'delta': delta,
                    'vega': vega,
                    'theta': theta,
                    'gamma': gamma,
                    'ce_oi': ce_oi,
                    'ce_oi_chg': ce_oi_chg
                })
        
        if not valid_strikes:
            return {'low': 0, 'high': 0, 'current': 0, 'atm': 0, 'analysis': "No valid strikes",
                   'greekSum': 0, 'pressure': 'BALANCED', 'confidence': 'LOW', 'greekComposition': {}}
        
        # Step 2: Find ATM based on Delta closest to 0.5 (User's method)
        atm_strike = valid_strikes[0]['strike']
        closest_delta_to_atm = 1
        atm_delta = 0
        
        for strike_data in valid_strikes:
            if abs(strike_data['delta'] - 0.5) < closest_delta_to_atm:
                closest_delta_to_atm = abs(strike_data['delta'] - 0.5)
                atm_strike = strike_data['strike']
                atm_delta = strike_data['delta']
        
        # Step 3: Filter Delta range 0.05 to 0.6 (User's methodology)
        delta_range_strikes = [s for s in valid_strikes if 0.05 <= s['delta'] <= 0.6]
        
        # Step 4: Calculate Greek sum for Delta range strikes (User's key insight)
        vega_sum = sum(s['vega'] for s in delta_range_strikes)
        theta_sum = sum(s['theta'] for s in delta_range_strikes)
        gamma_sum = sum(s['gamma'] for s in delta_range_strikes)
        greek_sum = vega_sum + theta_sum + gamma_sum
        
        # Step 5: Analyze ITM buildup for targets (User's directional logic)
        itm_ce_strikes = [s for s in valid_strikes if s['strike'] < atm_strike and s['delta'] > 0.3]
        itm_pe_strikes = [s for s in valid_strikes if s['strike'] > atm_strike]
        
        significant_ce = [s for s in itm_ce_strikes if abs(s['ce_oi_chg']) > 100000]
        significant_ce.sort(key=lambda x: abs(x['ce_oi_chg']), reverse=True)
        
        # Step 6: Determine market pressure using Greek composition (User's approach)
        market_pressure = "BALANCED"
        confidence = "LOW"
        analysis = ""
        
        # Initialize dominance variables to avoid reference errors
        theta_dominance = theta_sum / greek_sum if greek_sum > 0 else 0
        vega_dominance = vega_sum / greek_sum if greek_sum > 0 else 0
        gamma_dominance = gamma_sum / greek_sum if greek_sum > 0 else 0
        
        if greek_sum > 500:
            confidence = "HIGH" if greek_sum > 1000 else "MEDIUM"
            
            # User's observations: Greek dominance determines market behavior
            if theta_dominance > 0.5:
                market_pressure = "SIDEWAYS"
                analysis = "Theta dominant suggests time decay pressure, range-bound movement"
            elif vega_dominance > 0.6:
                market_pressure = "HIGH VOLATILITY"
                analysis = "Vega dominant indicates uncertainty, volatile conditions"
            elif gamma_dominance > 0.5:
                market_pressure = "DIRECTIONAL MOVE"
                analysis = "Gamma dominant suggests acceleration risk, breakout imminent"
            else:
                market_pressure = "MIXED SIGNALS"
                analysis = "Balanced Greek composition, monitor for shifts"
            
            # ITM CE buildup adds downside bias (User's target logic)
            if significant_ce:
                target_strike = significant_ce[0]['strike']
                analysis += f" | ITM CE buildup at {target_strike:.0f} suggests downside target"
                market_pressure += " → BEARISH BIAS"
        
        # Calculate range based on significant levels
        strikes = [s['strike'] for s in valid_strikes]
        predicted_low = min(strikes) if strikes else atm_strike
        predicted_high = max(strikes) if strikes else atm_strike
        
        if significant_ce:
            predicted_low = min(predicted_low, significant_ce[0]['strike'])
        
        return {
            'low': predicted_low,
            'high': predicted_high,
            'current': atm_strike,
            'atm': atm_strike,
            'atm_delta': atm_delta,
            'analysis': analysis or "Standard range calculation",
            'greekSum': greek_sum,
            'pressure': market_pressure,
            'confidence': confidence,
            'greekComposition': {
                'theta': theta_dominance if greek_sum > 0 else 0,
                'vega': vega_dominance if greek_sum > 0 else 0,
                'gamma': gamma_dominance if greek_sum > 0 else 0
            }
        }
    
    def analyze_timestamp(self, timestamp_index: int) -> Dict:
        """Analyze sentiment for a specific timestamp index"""
        if timestamp_index >= len(self.unique_timestamps):
            return {}
        
        timestamp = self.unique_timestamps[timestamp_index]
        timestamp_data = self.get_data_for_timestamp(timestamp)
        
        if timestamp_data.empty:
            return {}
        
        # Identify strike types
        timestamp_data = self.identify_strike_types(timestamp_data)
        
        # Calculate indicators
        indicators = self.calculate_sentiment_indicators(timestamp_data)
        
        # Determine sentiment
        sentiment, strength = self.determine_sentiment(indicators)
        
        # Calculate advanced range using user's methodology
        range_data = self.calculate_advanced_range(timestamp_data)
        
        return {
            'sentiment': sentiment,
            'strength': strength,
            'indicators': indicators,
            'range_data': range_data,
            'timestamp': timestamp,
            'strikes_data': timestamp_data
        }
    
    def process_full_data(self):
        """Process the entire dataset and create sentiment timeline"""
        if self.data is None or len(self.unique_timestamps) == 0:
            return
        
        self.sentiment_history = []
        self.strength_history = []
        
        for i in range(len(self.unique_timestamps)):
            analysis = self.analyze_timestamp(i)
            if analysis:
                self.sentiment_history.append(analysis['sentiment'])
                self.strength_history.append(analysis['strength'])

