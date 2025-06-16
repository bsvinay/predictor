import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import statistics

class KeyLevelDetector:
    """
    Advanced key level detection system for identifying:
    - Support/Resistance levels
    - Breakout/Breakdown levels  
    - Reversal levels
    - Max Pain levels
    - Gamma walls
    """
    
    def __init__(self):
        self.level_history = []  # Track level performance over time
        self.level_strength_weights = {
            'oi_concentration': 0.3,    # OI concentration importance
            'gamma_concentration': 0.25, # Gamma wall importance  
            'volume_activity': 0.2,     # Volume significance
            'price_proximity': 0.15,    # Distance from current price
            'historical_respect': 0.1   # How well level held in past
        }
    
    def detect_all_key_levels(self, current_data: pd.DataFrame, 
                             current_spot: float,
                             historical_data: Optional[List] = None) -> Dict:
        """
        Main method to detect all types of key levels
        """
        try:
            # 1. OI Concentration Levels (Support/Resistance)
            oi_levels = self.detect_oi_concentration_levels(current_data, current_spot)
            
            # 2. Gamma Walls (Breakout resistance)
            gamma_walls = self.detect_gamma_walls(current_data, current_spot)
            
            # 3. Max Pain Level (Gravitational center)
            max_pain = self.calculate_max_pain_level(current_data)
            
            # 4. Volume Activity Levels
            volume_levels = self.detect_volume_activity_levels(current_data, current_spot)
            
            # 5. Delta Neutral Levels
            delta_neutral = self.find_delta_neutral_levels(current_data, current_spot)
            
            # 6. Combine and rank all levels
            all_levels = self.combine_and_rank_levels(
                oi_levels, gamma_walls, max_pain, volume_levels, delta_neutral, current_spot
            )
            
            # 7. Identify breakout/breakdown scenarios
            breakout_analysis = self.analyze_breakout_scenarios(all_levels, current_spot)
            
            # 8. Classify levels by importance
            classified_levels = self.classify_level_importance(all_levels, current_spot)
            
            return {
                'key_levels': classified_levels,
                'breakout_analysis': breakout_analysis,
                'max_pain_level': max_pain,
                'level_details': {
                    'oi_levels': oi_levels,
                    'gamma_walls': gamma_walls,
                    'volume_levels': volume_levels,
                    'delta_neutral': delta_neutral
                },
                'current_spot': current_spot,
                'analysis_timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error in key level detection: {e}")
            return self.get_fallback_levels(current_spot)
    
    def detect_oi_concentration_levels(self, current_data: pd.DataFrame, current_spot: float) -> List[Dict]:
        """
        Detect levels with high OI concentration (Support/Resistance)
        """
        try:
            oi_by_strike = {}
            
            # Aggregate OI by strike
            for _, row in current_data.iterrows():
                strike = row.get('Strike Price', 0)
                oi = row.get('OI', 0)
                oi_change = row.get('OI Chg', 0)
                
                if strike > 0:
                    if strike not in oi_by_strike:
                        oi_by_strike[strike] = {'oi': 0, 'oi_change': 0}
                    oi_by_strike[strike]['oi'] += oi
                    oi_by_strike[strike]['oi_change'] += oi_change
            
            if not oi_by_strike:
                return []
            
            # Calculate total OI for percentage calculation
            total_oi = sum(data['oi'] for data in oi_by_strike.values())
            total_oi_change = sum(data['oi_change'] for data in oi_by_strike.values())
            
            oi_levels = []
            for strike, data in oi_by_strike.items():
                oi_concentration = (data['oi'] / total_oi) * 100 if total_oi > 0 else 0
                oi_change_impact = abs(data['oi_change'] / total_oi_change) * 100 if total_oi_change != 0 else 0
                
                # Only consider strikes with significant OI concentration
                if oi_concentration > 5:  # >5% of total OI
                    distance_from_spot = abs(strike - current_spot)
                    distance_pct = (distance_from_spot / current_spot) * 100
                    
                    # Determine level type based on position relative to spot
                    if strike > current_spot + (current_spot * 0.01):  # Above spot
                        level_type = 'RESISTANCE'
                    elif strike < current_spot - (current_spot * 0.01):  # Below spot
                        level_type = 'SUPPORT'
                    else:
                        level_type = 'PIVOT'  # Near current spot
                    
                    # Calculate strength based on OI concentration and activity
                    strength_score = oi_concentration + (oi_change_impact * 0.5)
                    
                    if strength_score > 20:
                        strength = 'VERY_STRONG'
                    elif strength_score > 15:
                        strength = 'STRONG'
                    elif strength_score > 10:
                        strength = 'MODERATE'
                    else:
                        strength = 'WEAK'
                    
                    oi_levels.append({
                        'strike': strike,
                        'level_type': level_type,
                        'strength': strength,
                        'strength_score': strength_score,
                        'oi_concentration': oi_concentration,
                        'oi_change_impact': oi_change_impact,
                        'distance_from_spot': distance_from_spot,
                        'distance_pct': distance_pct,
                        'total_oi': data['oi'],
                        'oi_change': data['oi_change'],
                        'source': 'OI_CONCENTRATION'
                    })
            
            # Sort by strength score
            oi_levels.sort(key=lambda x: x['strength_score'], reverse=True)
            
            return oi_levels[:10]  # Return top 10 OI levels
            
        except Exception as e:
            print(f"Error detecting OI levels: {e}")
            return []
    
    def detect_gamma_walls(self, current_data: pd.DataFrame, current_spot: float) -> List[Dict]:
        """
        Detect Gamma walls that act as breakout resistance
        """
        try:
            gamma_by_strike = {}
            
            # Aggregate Gamma by strike  
            for _, row in current_data.iterrows():
                strike = row.get('Strike Price', 0)
                gamma = abs(row.get('Gamma', 0))  # Use absolute Gamma
                
                if strike > 0 and gamma > 0:
                    if strike not in gamma_by_strike:
                        gamma_by_strike[strike] = 0
                    gamma_by_strike[strike] += gamma
            
            if not gamma_by_strike:
                return []
            
            total_gamma = sum(gamma_by_strike.values())
            gamma_walls = []
            
            for strike, gamma in gamma_by_strike.items():
                gamma_concentration = (gamma / total_gamma) * 100 if total_gamma > 0 else 0
                
                # Only consider strikes with significant Gamma concentration
                if gamma_concentration > 8:  # >8% of total Gamma
                    distance_from_spot = abs(strike - current_spot)
                    distance_pct = (distance_from_spot / current_spot) * 100
                    
                    # Gamma walls are typically closer to current price
                    if distance_pct <= 10:  # Within 10% of current spot
                        
                        # Determine resistance type
                        if strike > current_spot:
                            resistance_type = 'UPSIDE_RESISTANCE'
                        else:
                            resistance_type = 'DOWNSIDE_SUPPORT'
                        
                        # Calculate wall strength
                        if gamma_concentration > 20:
                            wall_strength = 'VERY_STRONG'
                        elif gamma_concentration > 15:
                            wall_strength = 'STRONG'  
                        elif gamma_concentration > 12:
                            wall_strength = 'MODERATE'
                        else:
                            wall_strength = 'WEAK'
                        
                        gamma_walls.append({
                            'strike': strike,
                            'resistance_type': resistance_type,
                            'wall_strength': wall_strength,
                            'gamma_concentration': gamma_concentration,
                            'total_gamma': gamma,
                            'distance_from_spot': distance_from_spot,
                            'distance_pct': distance_pct,
                            'breakout_probability': self.calculate_breakout_probability(
                                gamma_concentration, distance_pct
                            ),
                            'source': 'GAMMA_WALL'
                        })
            
            # Sort by Gamma concentration
            gamma_walls.sort(key=lambda x: x['gamma_concentration'], reverse=True)
            
            return gamma_walls[:8]  # Return top 8 Gamma walls
            
        except Exception as e:
            print(f"Error detecting Gamma walls: {e}")
            return []
    
    def calculate_max_pain_level(self, current_data: pd.DataFrame) -> Dict:
        """
        Calculate Max Pain level where maximum option premium decays
        """
        try:
            # Group data by strike to calculate max pain
            strikes_data = {}
            
            for _, row in current_data.iterrows():
                strike = row.get('Strike Price', 0)
                
                if strike > 0:
                    if strike not in strikes_data:
                        strikes_data[strike] = {
                            'call_oi': 0,
                            'put_oi': 0,
                            'total_oi': 0
                        }
                    
                    # Assume current data has both call and put info
                    # This might need adjustment based on your data structure
                    oi = row.get('OI', 0)
                    delta = row.get('Delta', 0)
                    
                    # Simple heuristic: positive delta = calls, negative = puts
                    if delta > 0:
                        strikes_data[strike]['call_oi'] += oi
                    else:
                        strikes_data[strike]['put_oi'] += oi
                    
                    strikes_data[strike]['total_oi'] += oi
            
            if not strikes_data:
                return {'max_pain_strike': 0, 'confidence': 'LOW'}
            
            # Calculate pain (loss) for option writers at each strike
            pain_by_strike = {}
            
            for test_strike in strikes_data.keys():
                total_pain = 0
                
                for strike, data in strikes_data.items():
                    call_oi = data['call_oi']
                    put_oi = data['put_oi']
                    
                    # Calculate pain for call writers
                    if test_strike > strike:  # Calls are ITM
                        call_pain = call_oi * (test_strike - strike)
                        total_pain += call_pain
                    
                    # Calculate pain for put writers  
                    if test_strike < strike:  # Puts are ITM
                        put_pain = put_oi * (strike - test_strike)
                        total_pain += put_pain
                
                pain_by_strike[test_strike] = total_pain
            
            # Find strike with maximum pain (where writers lose most)
            max_pain_strike = max(pain_by_strike.keys(), key=lambda x: pain_by_strike[x])
            max_pain_value = pain_by_strike[max_pain_strike]
            
            # Calculate confidence based on how much pain is concentrated at this level
            total_pain = sum(pain_by_strike.values())
            pain_concentration = (max_pain_value / total_pain) * 100 if total_pain > 0 else 0
            
            if pain_concentration > 25:
                confidence = 'HIGH'
            elif pain_concentration > 15:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
            
            return {
                'max_pain_strike': max_pain_strike,
                'max_pain_value': max_pain_value,
                'pain_concentration': pain_concentration,
                'confidence': confidence,
                'total_pain': total_pain,
                'pain_distribution': pain_by_strike,
                'source': 'MAX_PAIN_CALCULATION'
            }
            
        except Exception as e:
            print(f"Error calculating max pain: {e}")
            return {'max_pain_strike': 0, 'confidence': 'LOW'}
    
    def detect_volume_activity_levels(self, current_data: pd.DataFrame, current_spot: float) -> List[Dict]:
        """
        Detect levels with high volume activity
        """
        try:
            volume_by_strike = {}
            
            for _, row in current_data.iterrows():
                strike = row.get('Strike Price', 0)
                volume = row.get('Volume', 0)
                
                if strike > 0:
                    if strike not in volume_by_strike:
                        volume_by_strike[strike] = 0
                    volume_by_strike[strike] += volume
            
            if not volume_by_strike:
                return []
            
            total_volume = sum(volume_by_strike.values())
            volume_levels = []
            
            for strike, volume in volume_by_strike.items():
                volume_concentration = (volume / total_volume) * 100 if total_volume > 0 else 0
                
                if volume_concentration > 8:  # >8% of total volume
                    distance_from_spot = abs(strike - current_spot)
                    distance_pct = (distance_from_spot / current_spot) * 100
                    
                    # Classify volume level
                    if volume_concentration > 20:
                        activity_level = 'VERY_HIGH'
                    elif volume_concentration > 15:
                        activity_level = 'HIGH'
                    elif volume_concentration > 12:
                        activity_level = 'MODERATE'
                    else:
                        activity_level = 'LOW'
                    
                    volume_levels.append({
                        'strike': strike,
                        'activity_level': activity_level,
                        'volume_concentration': volume_concentration,
                        'total_volume': volume,
                        'distance_from_spot': distance_from_spot,
                        'distance_pct': distance_pct,
                        'source': 'VOLUME_ACTIVITY'
                    })
            
            # Sort by volume concentration
            volume_levels.sort(key=lambda x: x['volume_concentration'], reverse=True)
            
            return volume_levels[:6]  # Return top 6 volume levels
            
        except Exception as e:
            print(f"Error detecting volume levels: {e}")
            return []
    
    def find_delta_neutral_levels(self, current_data: pd.DataFrame, current_spot: float) -> List[Dict]:
        """
        Find levels where call and put deltas balance (delta neutral)
        """
        try:
            delta_by_strike = {}
            
            for _, row in current_data.iterrows():
                strike = row.get('Strike Price', 0)
                delta = row.get('Delta', 0)
                oi = row.get('OI', 0)
                
                if strike > 0:
                    if strike not in delta_by_strike:
                        delta_by_strike[strike] = {'weighted_delta': 0, 'total_oi': 0}
                    
                    # Weight delta by OI
                    delta_by_strike[strike]['weighted_delta'] += delta * oi
                    delta_by_strike[strike]['total_oi'] += oi
            
            delta_neutral_levels = []
            
            for strike, data in delta_by_strike.items():
                if data['total_oi'] > 0:
                    avg_delta = data['weighted_delta'] / data['total_oi']
                    
                    # Look for strikes close to delta neutral (delta ≈ 0)
                    if abs(avg_delta) < 0.1:  # Close to delta neutral
                        distance_from_spot = abs(strike - current_spot)
                        distance_pct = (distance_from_spot / current_spot) * 100
                        
                        delta_neutral_levels.append({
                            'strike': strike,
                            'avg_delta': avg_delta,
                            'total_oi': data['total_oi'],
                            'distance_from_spot': distance_from_spot,
                            'distance_pct': distance_pct,
                            'neutrality_score': 1 - abs(avg_delta),  # Higher = more neutral
                            'source': 'DELTA_NEUTRAL'
                        })
            
            # Sort by neutrality score
            delta_neutral_levels.sort(key=lambda x: x['neutrality_score'], reverse=True)
            
            return delta_neutral_levels[:4]  # Return top 4 delta neutral levels
            
        except Exception as e:
            print(f"Error finding delta neutral levels: {e}")
            return []
    
    def combine_and_rank_levels(self, oi_levels: List, gamma_walls: List, 
                               max_pain: Dict, volume_levels: List, 
                               delta_neutral: List, current_spot: float) -> List[Dict]:
        """
        Combine all level types and rank by importance
        """
        try:
            all_levels = []
            
            # Add OI levels
            for level in oi_levels:
                all_levels.append({
                    'strike': level['strike'],
                    'level_type': level['level_type'],
                    'strength': level['strength'],
                    'importance_score': self.calculate_importance_score(level, current_spot),
                    'sources': ['OI_CONCENTRATION'],
                    'details': level
                })
            
            # Add Gamma walls
            for wall in gamma_walls:
                # Check if strike already exists
                existing = next((l for l in all_levels if abs(l['strike'] - wall['strike']) < 25), None)
                if existing:
                    # Combine with existing level
                    existing['sources'].append('GAMMA_WALL')
                    existing['importance_score'] += wall['gamma_concentration'] * 0.5
                    existing['details']['gamma_info'] = wall
                else:
                    # Add as new level
                    level_type = 'RESISTANCE' if wall['strike'] > current_spot else 'SUPPORT'
                    all_levels.append({
                        'strike': wall['strike'],
                        'level_type': level_type,
                        'strength': wall['wall_strength'],
                        'importance_score': wall['gamma_concentration'] * 0.8,
                        'sources': ['GAMMA_WALL'],
                        'details': wall
                    })
            
            # Add Max Pain level
            if max_pain.get('max_pain_strike', 0) > 0:
                max_pain_strike = max_pain['max_pain_strike']
                existing = next((l for l in all_levels if abs(l['strike'] - max_pain_strike) < 25), None)
                if existing:
                    existing['sources'].append('MAX_PAIN')
                    existing['importance_score'] += max_pain.get('pain_concentration', 0) * 0.6
                    existing['details']['max_pain_info'] = max_pain
                else:
                    level_type = 'PIVOT'  # Max pain is typically a pivot level
                    all_levels.append({
                        'strike': max_pain_strike,
                        'level_type': level_type,
                        'strength': max_pain.get('confidence', 'MEDIUM'),
                        'importance_score': max_pain.get('pain_concentration', 0) * 0.6,
                        'sources': ['MAX_PAIN'],
                        'details': max_pain
                    })
            
            # Add Volume levels
            for vol_level in volume_levels:
                existing = next((l for l in all_levels if abs(l['strike'] - vol_level['strike']) < 25), None)
                if existing:
                    existing['sources'].append('VOLUME_ACTIVITY')
                    existing['importance_score'] += vol_level['volume_concentration'] * 0.4
                    existing['details']['volume_info'] = vol_level
                else:
                    level_type = 'RESISTANCE' if vol_level['strike'] > current_spot else 'SUPPORT'
                    all_levels.append({
                        'strike': vol_level['strike'],
                        'level_type': level_type,
                        'strength': vol_level['activity_level'],
                        'importance_score': vol_level['volume_concentration'] * 0.4,
                        'sources': ['VOLUME_ACTIVITY'],
                        'details': vol_level
                    })
            
            # Sort by importance score
            all_levels.sort(key=lambda x: x['importance_score'], reverse=True)
            
            return all_levels
            
        except Exception as e:
            print(f"Error combining levels: {e}")
            return []
    
    def calculate_importance_score(self, level: Dict, current_spot: float) -> float:
        """
        Calculate importance score for a level based on multiple factors
        """
        try:
            score = 0
            
            # Base score from strength
            strength_scores = {
                'VERY_STRONG': 50, 'STRONG': 40, 'MODERATE': 25, 'WEAK': 10
            }
            score += strength_scores.get(level.get('strength', 'WEAK'), 10)
            
            # Distance penalty (closer levels are more important)
            distance_pct = level.get('distance_pct', 50)
            distance_penalty = min(distance_pct * 2, 30)  # Max 30 point penalty
            score -= distance_penalty
            
            # OI concentration bonus
            oi_concentration = level.get('oi_concentration', 0)
            score += oi_concentration * 1.5
            
            # Ensure positive score
            return max(score, 1)
            
        except Exception as e:
            print(f"Error calculating importance score: {e}")
            return 1
    
    def analyze_breakout_scenarios(self, all_levels: List[Dict], current_spot: float) -> Dict:
        """
        Analyze potential breakout/breakdown scenarios
        """
        try:
            # Find immediate resistance and support levels
            immediate_resistance = []
            immediate_support = []
            
            for level in all_levels:
                if level['strike'] > current_spot and level['level_type'] in ['RESISTANCE', 'PIVOT']:
                    immediate_resistance.append(level)
                elif level['strike'] < current_spot and level['level_type'] in ['SUPPORT', 'PIVOT']:
                    immediate_support.append(level)
            
            # Sort by distance from current spot
            immediate_resistance.sort(key=lambda x: x['strike'])
            immediate_support.sort(key=lambda x: x['strike'], reverse=True)
            
            # Get closest levels
            next_resistance = immediate_resistance[0] if immediate_resistance else None
            next_support = immediate_support[0] if immediate_support else None
            
            # Analyze breakout probability
            breakout_analysis = {
                'next_resistance': next_resistance,
                'next_support': next_support,
                'upside_breakout': self.analyze_upside_breakout(immediate_resistance, current_spot),
                'downside_breakdown': self.analyze_downside_breakdown(immediate_support, current_spot),
                'range_bound_probability': self.calculate_range_bound_probability(
                    next_resistance, next_support, current_spot
                )
            }
            
            return breakout_analysis
            
        except Exception as e:
            print(f"Error analyzing breakout scenarios: {e}")
            return {}
    
    def analyze_upside_breakout(self, resistance_levels: List, current_spot: float) -> Dict:
        """
        Analyze upside breakout potential
        """
        try:
            if not resistance_levels:
                return {'probability': 'LOW', 'target': 0}
            
            first_resistance = resistance_levels[0]
            resistance_strength = first_resistance.get('importance_score', 0)
            distance_to_resistance = first_resistance['strike'] - current_spot
            distance_pct = (distance_to_resistance / current_spot) * 100
            
            # Calculate breakout probability
            if resistance_strength > 40 and distance_pct < 2:
                probability = 'LOW'  # Strong nearby resistance
            elif resistance_strength > 30:
                probability = 'MEDIUM'
            elif distance_pct > 5:
                probability = 'MEDIUM'  # Far resistance, easier to break
            else:
                probability = 'HIGH'
            
            # Find next target after breakout
            next_target = resistance_levels[1]['strike'] if len(resistance_levels) > 1 else first_resistance['strike'] * 1.02
            
            return {
                'probability': probability,
                'first_resistance': first_resistance['strike'],
                'resistance_strength': resistance_strength,
                'distance_pct': distance_pct,
                'next_target': next_target
            }
            
        except Exception as e:
            print(f"Error analyzing upside breakout: {e}")
            return {'probability': 'LOW', 'target': 0}
    
    def analyze_downside_breakdown(self, support_levels: List, current_spot: float) -> Dict:
        """
        Analyze downside breakdown potential
        """
        try:
            if not support_levels:
                return {'probability': 'LOW', 'target': 0}
            
            first_support = support_levels[0]
            support_strength = first_support.get('importance_score', 0)
            distance_to_support = current_spot - first_support['strike']
            distance_pct = (distance_to_support / current_spot) * 100
            
            # Calculate breakdown probability
            if support_strength > 40 and distance_pct < 2:
                probability = 'LOW'  # Strong nearby support
            elif support_strength > 30:
                probability = 'MEDIUM'
            elif distance_pct > 5:
                probability = 'MEDIUM'  # Far support, easier to break
            else:
                probability = 'HIGH'
            
            # Find next target after breakdown
            next_target = support_levels[1]['strike'] if len(support_levels) > 1 else first_support['strike'] * 0.98
            
            return {
                'probability': probability,
                'first_support': first_support['strike'],
                'support_strength': support_strength,
                'distance_pct': distance_pct,
                'next_target': next_target
            }
            
        except Exception as e:
            print(f"Error analyzing downside breakdown: {e}")
            return {'probability': 'LOW', 'target': 0}
    
    def calculate_range_bound_probability(self, next_resistance: Dict, next_support: Dict, current_spot: float) -> Dict:
        """
        Calculate probability of range-bound movement
        """
        try:
            if not next_resistance or not next_support:
                return {'probability': 'LOW', 'range_width': 0}
            
            resistance_level = next_resistance['strike']
            support_level = next_support['strike']
            range_width = resistance_level - support_level
            range_width_pct = (range_width / current_spot) * 100
            
            resistance_strength = next_resistance.get('importance_score', 0)
            support_strength = next_support.get('importance_score', 0)
            combined_strength = (resistance_strength + support_strength) / 2
            
            # Range-bound probability based on range width and level strength
            if range_width_pct < 3 and combined_strength > 35:
                probability = 'HIGH'  # Tight range with strong levels
            elif range_width_pct < 5 and combined_strength > 25:
                probability = 'MEDIUM'
            else:
                probability = 'LOW'
            
            return {
                'probability': probability,
                'range_width': range_width,
                'range_width_pct': range_width_pct,
                'resistance_level': resistance_level,
                'support_level': support_level,
                'combined_strength': combined_strength
            }
            
        except Exception as e:
            print(f"Error calculating range-bound probability: {e}")
            return {'probability': 'LOW', 'range_width': 0}
    
    def classify_level_importance(self, all_levels: List[Dict], current_spot: float) -> Dict:
        """
        Classify levels by importance categories
        """
        try:
            classified = {
                'critical_levels': [],    # Levels with very high importance
                'major_levels': [],       # Important levels to watch
                'minor_levels': [],       # Secondary levels
                'pivot_levels': []        # Special pivot/max pain levels
            }
            
            for level in all_levels:
                importance_score = level.get('importance_score', 0)
                level_type = level.get('level_type', '')
                
                if importance_score > 50:
                    classified['critical_levels'].append(level)
                elif importance_score > 30:
                    classified['major_levels'].append(level)
                elif importance_score > 15:
                    classified['minor_levels'].append(level)
                
                if level_type == 'PIVOT' or 'MAX_PAIN' in level.get('sources', []):
                    classified['pivot_levels'].append(level)
            
            return classified
            
        except Exception as e:
            print(f"Error classifying level importance: {e}")
            return {'critical_levels': [], 'major_levels': [], 'minor_levels': [], 'pivot_levels': []}
    
    def calculate_breakout_probability(self, gamma_concentration: float, distance_pct: float) -> str:
        """
        Calculate probability of breaking through a Gamma wall
        """
        try:
            # High Gamma concentration = harder to break
            # Close distance = easier to test
            
            if gamma_concentration > 20 and distance_pct < 3:
                return 'LOW'  # Strong wall, close to price
            elif gamma_concentration > 15:
                return 'MEDIUM'
            elif distance_pct > 8:
                return 'MEDIUM'  # Far from price, less immediate impact
            else:
                return 'HIGH'
                
        except Exception as e:
            return 'LOW'
    
    def get_fallback_levels(self, current_spot: float) -> Dict:
        """
        Fallback levels when detection fails
        """
        fallback_range = current_spot * 0.02  # 2% range
        
        return {
            'key_levels': {
                'critical_levels': [
                    {
                        'strike': current_spot + fallback_range,
                        'level_type': 'RESISTANCE',
                        'strength': 'MEDIUM',
                        'importance_score': 25,
                        'sources': ['FALLBACK']
                    },
                    {
                        'strike': current_spot - fallback_range,
                        'level_type': 'SUPPORT', 
                        'strength': 'MEDIUM',
                        'importance_score': 25,
                        'sources': ['FALLBACK']
                    }
                ],
                'major_levels': [],
                'minor_levels': [],
                'pivot_levels': []
            },
            'breakout_analysis': {
                'next_resistance': {'strike': current_spot + fallback_range},
                'next_support': {'strike': current_spot - fallback_range},
                'upside_breakout': {'probability': 'MEDIUM'},
                'downside_breakdown': {'probability': 'MEDIUM'},
                'range_bound_probability': {'probability': 'MEDIUM'}
            },
            'max_pain_level': {'max_pain_strike': current_spot, 'confidence': 'LOW'},
            'current_spot': current_spot,
            'analysis_timestamp': datetime.now(),
            'note': 'Fallback levels due to detection error'
        }