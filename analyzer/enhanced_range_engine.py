import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

class EnhancedRangePredictor:
    """
    Advanced range prediction engine using multi-factor analysis
    Combines Vega momentum, Gamma walls, OI flow, and historical patterns
    """
    
    def __init__(self):
        self.vega_sensitivity = 0.4    # How much Vega affects range
        self.gamma_sensitivity = 0.3   # How much Gamma affects breakouts
        self.oi_sensitivity = 0.2      # How much OI flow affects bias
        self.historical_weight = 0.1   # Historical pattern influence
        
        # Range prediction history for learning
        self.range_predictions = []
        self.actual_ranges = []
        self.accuracy_history = []
        
    def predict_daily_range(self, current_data: pd.DataFrame, 
                           baseline_data: pd.DataFrame,
                           current_spot: float,
                           historical_data: Optional[List] = None) -> Dict:
        """
        Main method to predict daily range using multi-factor analysis
        """
        try:
            # 1. Calculate Vega-based volatility expansion
            vega_range = self.calculate_vega_range_expansion(current_data, baseline_data, current_spot)
            
            # 2. Detect Gamma walls for breakout levels
            gamma_levels = self.detect_gamma_walls(current_data, current_spot)
            
            # 3. Analyze OI flow for directional bias
            oi_bias = self.analyze_oi_momentum(current_data, baseline_data)
            
            # 4. Calculate base statistical range
            base_range = self.calculate_base_range(current_data, current_spot)
            
            # 5. Synthesize all factors into final prediction
            range_prediction = self.synthesize_range_prediction(
                vega_range, gamma_levels, oi_bias, base_range, current_spot
            )
            
            # 6. Add confidence scoring
            confidence = self.calculate_range_confidence(current_data, range_prediction)
            
            # 7. Store prediction for learning
            self.store_prediction(range_prediction, confidence)
            
            return {
                'predicted_low': range_prediction['low'],
                'predicted_high': range_prediction['high'],
                'current_spot': current_spot,
                'range_width': range_prediction['high'] - range_prediction['low'],
                'confidence': confidence,
                'factors': {
                    'vega_expansion': vega_range,
                    'gamma_walls': gamma_levels,
                    'oi_bias': oi_bias,
                    'base_range': base_range
                },
                'breakout_levels': gamma_levels.get('breakout_levels', []),
                'key_levels': self.identify_key_levels(current_data, range_prediction),
                'prediction_timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error in range prediction: {e}")
            return self.get_fallback_range(current_spot)
    
    def calculate_vega_range_expansion(self, current_data: pd.DataFrame, 
                                     baseline_data: pd.DataFrame, 
                                     current_spot: float) -> Dict:
        """
        Calculate range expansion based on Vega momentum
        High Vega increase = Wider expected range
        """
        try:
            # Get current and baseline Vega sums
            current_vega = self.get_vega_sum(current_data)
            baseline_vega = self.get_vega_sum(baseline_data)
            
            if baseline_vega == 0:
                return {'expansion_factor': 1.0, 'vega_momentum': 0}
            
            # Calculate Vega momentum (% change from baseline)
            vega_momentum = ((current_vega - baseline_vega) / baseline_vega) * 100
            
            # Convert Vega momentum to range expansion factor
            # High Vega momentum = Wider range expectation
            if abs(vega_momentum) > 20:  # Strong Vega movement
                expansion_factor = 1.0 + (abs(vega_momentum) / 200)  # 20% Vega = 10% wider range
            elif abs(vega_momentum) > 10:  # Moderate Vega movement
                expansion_factor = 1.0 + (abs(vega_momentum) / 400)  # 10% Vega = 2.5% wider range
            else:  # Low Vega movement
                expansion_factor = 1.0 + (abs(vega_momentum) / 800)  # Minimal expansion
            
            # Calculate Vega-based range
            base_range_width = current_spot * 0.02  # 2% base range
            vega_range_width = base_range_width * expansion_factor
            
            return {
                'expansion_factor': expansion_factor,
                'vega_momentum': vega_momentum,
                'vega_range_width': vega_range_width,
                'current_vega': current_vega,
                'baseline_vega': baseline_vega
            }
            
        except Exception as e:
            print(f"Error in Vega range calculation: {e}")
            return {'expansion_factor': 1.0, 'vega_momentum': 0}
    
    def detect_gamma_walls(self, current_data: pd.DataFrame, current_spot: float) -> Dict:
        """
        Detect Gamma walls that act as breakout resistance levels
        High Gamma concentration = Breakout resistance
        """
        try:
            # Calculate Gamma concentration at each strike
            gamma_by_strike = {}
            for _, row in current_data.iterrows():
                strike = row.get('Strike Price', 0)
                gamma = abs(row.get('Gamma', 0))
                
                if strike > 0:
                    if strike not in gamma_by_strike:
                        gamma_by_strike[strike] = 0
                    gamma_by_strike[strike] += gamma
            
            if not gamma_by_strike:
                return {'gamma_walls': [], 'breakout_levels': []}
            
            # Find strikes with highest Gamma concentration
            sorted_gamma = sorted(gamma_by_strike.items(), key=lambda x: x[1], reverse=True)
            
            # Identify potential Gamma walls (top 20% of Gamma)
            total_gamma = sum(gamma_by_strike.values())
            gamma_threshold = total_gamma * 0.1  # Top strikes with >10% of total Gamma
            
            gamma_walls = []
            for strike, gamma in sorted_gamma:
                if gamma > gamma_threshold:
                    distance_from_spot = abs(strike - current_spot)
                    gamma_strength = gamma / total_gamma
                    
                    gamma_walls.append({
                        'strike': strike,
                        'gamma': gamma,
                        'gamma_strength': gamma_strength,
                        'distance_from_spot': distance_from_spot,
                        'resistance_level': 'HIGH' if gamma_strength > 0.15 else 'MEDIUM'
                    })
            
            # Sort by distance from current spot
            gamma_walls.sort(key=lambda x: x['distance_from_spot'])
            
            # Find breakout levels (Gamma walls that could act as resistance)
            breakout_levels = []
            for wall in gamma_walls[:6]:  # Top 6 closest Gamma walls
                if wall['distance_from_spot'] < current_spot * 0.05:  # Within 5% of spot
                    breakout_levels.append({
                        'level': wall['strike'],
                        'type': 'RESISTANCE' if wall['strike'] > current_spot else 'SUPPORT',
                        'strength': wall['resistance_level'],
                        'gamma_concentration': wall['gamma_strength']
                    })
            
            return {
                'gamma_walls': gamma_walls,
                'breakout_levels': breakout_levels,
                'total_gamma': total_gamma
            }
            
        except Exception as e:
            print(f"Error in Gamma wall detection: {e}")
            return {'gamma_walls': [], 'breakout_levels': []}
    
    def analyze_oi_momentum(self, current_data: pd.DataFrame, 
                           baseline_data: pd.DataFrame) -> Dict:
        """
        Analyze OI changes for directional bias
        Strong OI changes indicate institutional positioning
        """
        try:
            # Calculate OI changes from baseline
            current_oi_total = current_data['OI'].sum()
            baseline_oi_total = baseline_data['OI'].sum()
            
            if baseline_oi_total == 0:
                return {'oi_momentum': 0, 'bias': 'NEUTRAL'}
            
            oi_change_pct = ((current_oi_total - baseline_oi_total) / baseline_oi_total) * 100
            
            # Analyze OI changes by strike position
            oi_changes_by_position = {'ITM': 0, 'ATM': 0, 'OTM': 0}
            
            # Find ATM strike
            strikes = current_data['Strike Price'].dropna().unique()
            atm_strike = strikes[len(strikes) // 2] if len(strikes) > 0 else 0
            
            for _, row in current_data.iterrows():
                strike = row.get('Strike Price', 0)
                oi = row.get('OI', 0)
                
                if strike > 0:
                    if abs(strike - atm_strike) <= 50:  # ATM range
                        oi_changes_by_position['ATM'] += oi
                    elif strike < atm_strike:  # ITM calls/OTM puts
                        oi_changes_by_position['ITM'] += oi
                    else:  # OTM calls/ITM puts
                        oi_changes_by_position['OTM'] += oi
            
            # Determine bias based on OI positioning
            total_oi = sum(oi_changes_by_position.values())
            if total_oi > 0:
                itm_ratio = oi_changes_by_position['ITM'] / total_oi
                otm_ratio = oi_changes_by_position['OTM'] / total_oi
                atm_ratio = oi_changes_by_position['ATM'] / total_oi
                
                # Determine bias
                if itm_ratio > 0.5:
                    bias = 'BEARISH'  # Heavy ITM positioning
                elif otm_ratio > 0.5:
                    bias = 'BULLISH'  # Heavy OTM positioning
                elif atm_ratio > 0.6:
                    bias = 'SIDEWAYS'  # Heavy ATM positioning
                else:
                    bias = 'NEUTRAL'
            else:
                bias = 'NEUTRAL'
            
            return {
                'oi_momentum': oi_change_pct,
                'bias': bias,
                'oi_distribution': oi_changes_by_position,
                'total_oi_change': current_oi_total - baseline_oi_total
            }
            
        except Exception as e:
            print(f"Error in OI momentum analysis: {e}")
            return {'oi_momentum': 0, 'bias': 'NEUTRAL'}
    
    def calculate_base_range(self, current_data: pd.DataFrame, current_spot: float) -> Dict:
        """
        Calculate base statistical range from current options data
        """
        try:
            # Get all strikes and their distances from spot
            strikes = current_data['Strike Price'].dropna().unique()
            strikes = sorted([s for s in strikes if s > 0])
            
            if len(strikes) < 3:
                # Fallback to percentage-based range
                range_width = current_spot * 0.03  # 3% range
                return {
                    'low': current_spot - range_width,
                    'high': current_spot + range_width,
                    'width': range_width * 2,
                    'method': 'percentage_fallback'
                }
            
            # Calculate weighted range based on OI and Volume
            weighted_strikes = []
            for strike in strikes:
                strike_data = current_data[current_data['Strike Price'] == strike]
                if not strike_data.empty:
                    oi = strike_data['OI'].sum()
                    volume = strike_data['Volume'].sum()
                    weight = oi + volume
                    
                    if weight > 0:
                        weighted_strikes.append({
                            'strike': strike,
                            'weight': weight,
                            'distance': abs(strike - current_spot)
                        })
            
            if not weighted_strikes:
                range_width = current_spot * 0.03
                return {
                    'low': current_spot - range_width,
                    'high': current_spot + range_width,
                    'width': range_width * 2,
                    'method': 'percentage_fallback'
                }
            
            # Calculate weighted average distance
            total_weight = sum(s['weight'] for s in weighted_strikes)
            weighted_distance = sum(s['distance'] * s['weight'] for s in weighted_strikes) / total_weight
            
            # Use weighted distance as base range
            base_range_width = weighted_distance * 1.5  # 1.5x weighted distance
            
            return {
                'low': current_spot - base_range_width,
                'high': current_spot + base_range_width,
                'width': base_range_width * 2,
                'weighted_distance': weighted_distance,
                'method': 'weighted_strikes'
            }
            
        except Exception as e:
            print(f"Error in base range calculation: {e}")
            range_width = current_spot * 0.03
            return {
                'low': current_spot - range_width,
                'high': current_spot + range_width,
                'width': range_width * 2,
                'method': 'error_fallback'
            }
    
    def synthesize_range_prediction(self, vega_range: Dict, gamma_levels: Dict, 
                                  oi_bias: Dict, base_range: Dict, current_spot: float) -> Dict:
        """
        Combine all factors into final range prediction
        """
        try:
            # Start with base range
            base_low = base_range['low']
            base_high = base_range['high']
            base_width = base_high - base_low
            
            # Apply Vega expansion
            vega_expansion = vega_range.get('expansion_factor', 1.0)
            expanded_width = base_width * vega_expansion
            
            # Adjust for OI bias
            oi_bias_type = oi_bias.get('bias', 'NEUTRAL')
            if oi_bias_type == 'BULLISH':
                # Bias range upward
                final_low = current_spot - (expanded_width * 0.4)
                final_high = current_spot + (expanded_width * 0.6)
            elif oi_bias_type == 'BEARISH':
                # Bias range downward
                final_low = current_spot - (expanded_width * 0.6)
                final_high = current_spot + (expanded_width * 0.4)
            else:
                # Neutral bias
                final_low = current_spot - (expanded_width * 0.5)
                final_high = current_spot + (expanded_width * 0.5)
            
            # Adjust for Gamma walls
            gamma_walls = gamma_levels.get('breakout_levels', [])
            for wall in gamma_walls:
                wall_level = wall['level']
                wall_strength = wall.get('gamma_concentration', 0)
                
                # Strong Gamma walls can limit range
                if wall_strength > 0.15:  # Strong wall
                    if wall['type'] == 'RESISTANCE' and wall_level < final_high:
                        final_high = min(final_high, wall_level + (current_spot * 0.01))
                    elif wall['type'] == 'SUPPORT' and wall_level > final_low:
                        final_low = max(final_low, wall_level - (current_spot * 0.01))
            
            return {
                'low': final_low,
                'high': final_high,
                'width': final_high - final_low,
                'center': (final_high + final_low) / 2,
                'upside_potential': final_high - current_spot,
                'downside_risk': current_spot - final_low,
                'synthesis_factors': {
                    'base_range': base_range,
                    'vega_expansion': vega_expansion,
                    'oi_bias': oi_bias_type,
                    'gamma_adjustments': len(gamma_walls)
                }
            }
            
        except Exception as e:
            print(f"Error in range synthesis: {e}")
            fallback_range = current_spot * 0.04
            return {
                'low': current_spot - fallback_range,
                'high': current_spot + fallback_range,
                'width': fallback_range * 2,
                'center': current_spot
            }
    
    def calculate_range_confidence(self, current_data: pd.DataFrame, range_prediction: Dict) -> str:
        """
        Calculate confidence level for range prediction
        """
        try:
            confidence_score = 0
            
            # Factor 1: Data quality
            if len(current_data) > 20:
                confidence_score += 20
            elif len(current_data) > 10:
                confidence_score += 10
            
            # Factor 2: Vega consistency
            vega_values = current_data['Vega'].dropna()
            if len(vega_values) > 0:
                vega_std = vega_values.std()
                vega_mean = abs(vega_values.mean())
                if vega_mean > 0 and (vega_std / vega_mean) < 0.5:  # Low relative std
                    confidence_score += 20
                elif vega_mean > 0:
                    confidence_score += 10
            
            # Factor 3: OI data availability
            oi_values = current_data['OI'].dropna()
            if len(oi_values) > 0 and oi_values.sum() > 1000000:  # Significant OI
                confidence_score += 20
            elif len(oi_values) > 0:
                confidence_score += 10
            
            # Factor 4: Gamma concentration
            gamma_values = current_data['Gamma'].dropna()
            if len(gamma_values) > 0:
                gamma_sum = abs(gamma_values.sum())
                if gamma_sum > 100:  # Significant Gamma
                    confidence_score += 20
                elif gamma_sum > 50:
                    confidence_score += 10
            
            # Factor 5: Range width reasonableness
            range_width = range_prediction.get('width', 0)
            current_spot = range_prediction.get('center', 0)
            if current_spot > 0:
                range_pct = (range_width / current_spot) * 100
                if 2 <= range_pct <= 8:  # Reasonable range (2-8%)
                    confidence_score += 20
                elif range_pct <= 15:  # Acceptable range
                    confidence_score += 10
            
            # Convert to confidence level
            if confidence_score >= 80:
                return 'VERY HIGH'
            elif confidence_score >= 60:
                return 'HIGH'
            elif confidence_score >= 40:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception as e:
            print(f"Error in confidence calculation: {e}")
            return 'LOW'
    
    def identify_key_levels(self, current_data: pd.DataFrame, range_prediction: Dict) -> List[Dict]:
        """
        Identify key support/resistance levels within the predicted range
        """
        try:
            key_levels = []
            
            # Add range boundaries as key levels
            key_levels.append({
                'level': range_prediction['low'],
                'type': 'SUPPORT',
                'strength': 'HIGH',
                'source': 'PREDICTED_RANGE_LOW'
            })
            
            key_levels.append({
                'level': range_prediction['high'],
                'type': 'RESISTANCE',
                'strength': 'HIGH',
                'source': 'PREDICTED_RANGE_HIGH'
            })
            
            # Add significant OI levels
            oi_by_strike = {}
            for _, row in current_data.iterrows():
                strike = row.get('Strike Price', 0)
                oi = row.get('OI', 0)
                
                if strike > 0:
                    if strike not in oi_by_strike:
                        oi_by_strike[strike] = 0
                    oi_by_strike[strike] += oi
            
            if oi_by_strike:
                total_oi = sum(oi_by_strike.values())
                for strike, oi in oi_by_strike.items():
                    oi_concentration = oi / total_oi
                    
                    if oi_concentration > 0.1:  # >10% of total OI
                        # Determine if it's within our range
                        if range_prediction['low'] <= strike <= range_prediction['high']:
                            key_levels.append({
                                'level': strike,
                                'type': 'PIVOT',
                                'strength': 'HIGH' if oi_concentration > 0.15 else 'MEDIUM',
                                'source': 'OI_CONCENTRATION',
                                'oi_percentage': oi_concentration * 100
                            })
            
            # Sort by level
            key_levels.sort(key=lambda x: x['level'])
            
            return key_levels
            
        except Exception as e:
            print(f"Error identifying key levels: {e}")
            return []
    
    def get_vega_sum(self, data: pd.DataFrame) -> float:
        """Helper method to calculate total Vega sum"""
        try:
            return abs(data['Vega'].sum())
        except:
            return 0
    
    def store_prediction(self, range_prediction: Dict, confidence: str):
        """Store prediction for later accuracy analysis"""
        try:
            self.range_predictions.append({
                'timestamp': datetime.now(),
                'predicted_low': range_prediction['low'],
                'predicted_high': range_prediction['high'],
                'predicted_width': range_prediction['width'],
                'confidence': confidence
            })
        except Exception as e:
            print(f"Error storing prediction: {e}")
    
    def get_fallback_range(self, current_spot: float) -> Dict:
        """Fallback range prediction when main calculation fails"""
        fallback_range = current_spot * 0.03  # 3% range
        return {
            'predicted_low': current_spot - fallback_range,
            'predicted_high': current_spot + fallback_range,
            'current_spot': current_spot,
            'range_width': fallback_range * 2,
            'confidence': 'LOW',
            'factors': {},
            'breakout_levels': [],
            'key_levels': [],
            'prediction_timestamp': datetime.now(),
            'note': 'Fallback prediction due to calculation error'
        }


# Integration with your existing analyzer
class AdvancedOptionsAnalyzer:
    """Enhanced version of your existing analyzer with range prediction"""
    
    def __init__(self):
        # Your existing initialization code...
        self.range_predictor = EnhancedRangePredictor()
        self.range_prediction_history = []
    
    def analyze_timestamp_with_range_prediction(self, timestamp_index: int) -> Dict:
        """Enhanced analysis that includes advanced range prediction"""
        
        # Get your existing analysis
        analysis = self.analyze_timestamp(timestamp_index)
        
        if not analysis:
            return analysis
        
        # Get current and baseline data
        current_data = analysis.get('strikes_data', pd.DataFrame())
        baseline_data = self.baseline_data if hasattr(self, 'baseline_data') else pd.DataFrame()
        
        # Estimate current spot price
        current_spot = self.estimate_current_spot(current_data)
        
        # Get enhanced range prediction
        range_prediction = self.range_predictor.predict_daily_range(
            current_data, baseline_data, current_spot
        )
        
        # Add range prediction to analysis
        analysis['enhanced_range_prediction'] = range_prediction
        
        # Store for history
        self.range_prediction_history.append(range_prediction)
        
        return analysis
    
    def estimate_current_spot(self, current_data: pd.DataFrame) -> float:
        """Estimate current spot price from options data"""
        try:
            # Find ATM strike (Delta closest to 0.5)
            closest_to_atm = float('inf')
            atm_strike = 0
            
            for _, row in current_data.iterrows():
                strike = row.get('Strike Price', 0)
                delta = row.get('Delta', 0)
                
                if strike > 0 and delta > 0:
                    # Handle Delta scaling
                    if delta > 1:
                        delta = delta / 100 if delta <= 100 else delta / 1000
                    
                    delta_diff = abs(delta - 0.5)
                    if delta_diff < closest_to_atm:
                        closest_to_atm = delta_diff
                        atm_strike = strike
            
            return atm_strike if atm_strike > 0 else 24000  # Fallback
            
        except Exception as e:
            print(f"Error estimating spot: {e}")
            return 24000  # Fallback spot price