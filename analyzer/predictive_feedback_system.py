import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import json
import sqlite3
from dataclasses import dataclass, asdict
import statistics

@dataclass
class Prediction:
    """Structure for storing predictions"""
    timestamp: datetime
    prediction_type: str  # 'DIRECTION', 'RANGE', 'BREAKOUT', 'LEVEL'
    predicted_value: Union[str, float, Dict]
    confidence: str
    factors: Dict
    timeframe: str  # '30min', '1hr', '3hr', 'EOD'
    current_spot: float
    prediction_id: str

@dataclass 
class PredictionResult:
    """Structure for storing prediction outcomes"""
    prediction_id: str
    actual_value: Union[str, float, Dict]
    accuracy: float
    outcome_timestamp: datetime
    success: bool

class PredictiveModel:
    """
    Advanced predictive model with learning capabilities
    Makes predictions and learns from outcomes to improve accuracy
    """
    
    def __init__(self, db_path: str = "predictions.db"):
        self.db_path = db_path
        self.initialize_database()
        
        # Model parameters (will be adjusted based on performance)
        self.weights = {
            'vega_momentum': 0.35,      # Vega change importance
            'gamma_acceleration': 0.25,  # Gamma spike importance  
            'oi_flow': 0.20,            # OI change importance
            'level_proximity': 0.15,     # Distance to key levels
            'historical_pattern': 0.05   # Pattern matching weight
        }
        
        # Learning parameters
        self.learning_rate = 0.1
        self.min_predictions_for_learning = 10
        self.weight_adjustment_threshold = 0.6  # Adjust if accuracy < 60%
        
        # Prediction history
        self.predictions = []
        self.results = []
        self.accuracy_history = {}
        
    def initialize_database(self):
        """Initialize SQLite database for prediction storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    prediction_type TEXT,
                    predicted_value TEXT,
                    confidence TEXT,
                    factors TEXT,
                    timeframe TEXT,
                    current_spot REAL
                )
            ''')
            
            # Results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    prediction_id TEXT,
                    actual_value TEXT,
                    accuracy REAL,
                    outcome_timestamp TEXT,
                    success INTEGER,
                    FOREIGN KEY (prediction_id) REFERENCES predictions (id)
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    timestamp TEXT,
                    prediction_type TEXT,
                    timeframe TEXT,
                    accuracy REAL,
                    total_predictions INTEGER,
                    successful_predictions INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def make_comprehensive_prediction(self, current_data: pd.DataFrame,
                                    baseline_data: pd.DataFrame,
                                    key_levels: Dict,
                                    range_prediction: Dict,
                                    current_spot: float,
                                    historical_data: Optional[List] = None) -> Dict:
        """
        Make comprehensive predictions across multiple categories
        """
        try:
            predictions = {}
            current_time = datetime.now()
            
            # 1. Direction Prediction (Bull/Bear/Sideways)
            direction_pred = self.predict_direction(
                current_data, baseline_data, key_levels, current_spot
            )
            predictions['direction'] = direction_pred
            
            # 2. Range Predictions (multiple timeframes)
            range_predictions = self.predict_ranges(
                current_data, range_prediction, key_levels, current_spot
            )
            predictions['ranges'] = range_predictions
            
            # 3. Breakout/Breakdown Predictions
            breakout_pred = self.predict_breakouts(
                key_levels, current_data, current_spot
            )
            predictions['breakouts'] = breakout_pred
            
            # 4. Level Test Predictions (will key levels hold?)
            level_pred = self.predict_level_behavior(
                key_levels, current_data, current_spot
            )
            predictions['levels'] = level_pred
            
            # 5. Volatility Predictions
            volatility_pred = self.predict_volatility_changes(
                current_data, baseline_data
            )
            predictions['volatility'] = volatility_pred
            
            # Store all predictions
            for pred_type, pred_data in predictions.items():
                if isinstance(pred_data, dict) and 'predictions' in pred_data:
                    for individual_pred in pred_data['predictions']:
                        self.store_prediction(individual_pred)
            
            # Calculate overall market sentiment
            overall_sentiment = self.synthesize_overall_sentiment(predictions, current_spot)
            predictions['overall_sentiment'] = overall_sentiment
            
            return {
                'predictions': predictions,
                'prediction_timestamp': current_time,
                'current_spot': current_spot,
                'model_confidence': self.calculate_model_confidence(predictions),
                'next_review_time': current_time + timedelta(minutes=30)
            }
            
        except Exception as e:
            print(f"Error making comprehensive prediction: {e}")
            return self.get_fallback_prediction(current_spot)
    
    def predict_direction(self, current_data: pd.DataFrame, baseline_data: pd.DataFrame,
                         key_levels: Dict, current_spot: float) -> Dict:
        """
        Predict market direction (BULLISH/BEARISH/SIDEWAYS)
        """
        try:
            factors = {}
            direction_score = 0
            
            # Factor 1: Vega momentum analysis
            vega_factor = self.analyze_vega_momentum(current_data, baseline_data)
            factors['vega_momentum'] = vega_factor
            direction_score += vega_factor['score'] * self.weights['vega_momentum']
            
            # Factor 2: Gamma acceleration analysis
            gamma_factor = self.analyze_gamma_acceleration(current_data, baseline_data)
            factors['gamma_acceleration'] = gamma_factor  
            direction_score += gamma_factor['score'] * self.weights['gamma_acceleration']
            
            # Factor 3: OI flow analysis
            oi_factor = self.analyze_oi_flow_direction(current_data, baseline_data)
            factors['oi_flow'] = oi_factor
            direction_score += oi_factor['score'] * self.weights['oi_flow']
            
            # Factor 4: Level proximity bias
            level_factor = self.analyze_level_proximity_bias(key_levels, current_spot)
            factors['level_proximity'] = level_factor
            direction_score += level_factor['score'] * self.weights['level_proximity']
            
            # Determine direction and confidence
            if direction_score > 0.6:
                direction = 'BULLISH'
                confidence = 'HIGH' if direction_score > 0.8 else 'MEDIUM'
            elif direction_score < -0.6:
                direction = 'BEARISH' 
                confidence = 'HIGH' if direction_score < -0.8 else 'MEDIUM'
            else:
                direction = 'SIDEWAYS'
                confidence = 'MEDIUM' if abs(direction_score) < 0.3 else 'LOW'
            
            # Create predictions for different timeframes
            predictions = []
            for timeframe in ['30min', '1hr', '3hr']:
                pred_id = f"DIR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{timeframe}"
                
                pred = Prediction(
                    timestamp=datetime.now(),
                    prediction_type='DIRECTION',
                    predicted_value=direction,
                    confidence=confidence,
                    factors=factors,
                    timeframe=timeframe,
                    current_spot=current_spot,
                    prediction_id=pred_id
                )
                predictions.append(pred)
            
            return {
                'direction': direction,
                'confidence': confidence,
                'direction_score': direction_score,
                'factors': factors,
                'predictions': predictions
            }
            
        except Exception as e:
            print(f"Error predicting direction: {e}")
            return {'direction': 'SIDEWAYS', 'confidence': 'LOW'}
    
    def predict_ranges(self, current_data: pd.DataFrame, range_prediction: Dict,
                      key_levels: Dict, current_spot: float) -> Dict:
        """
        Predict price ranges for different timeframes
        """
        try:
            base_range = range_prediction
            range_predictions = []
            
            # Adjust ranges based on timeframe and volatility
            volatility_factor = self.calculate_current_volatility_factor(current_data)
            
            timeframe_multipliers = {
                '30min': 0.3,
                '1hr': 0.5,
                '3hr': 0.8,
                'EOD': 1.0
            }
            
            for timeframe, multiplier in timeframe_multipliers.items():
                # Adjust range based on timeframe
                base_width = base_range.get('range_width', current_spot * 0.04)
                adjusted_width = base_width * multiplier * volatility_factor
                
                predicted_low = current_spot - (adjusted_width * 0.5)
                predicted_high = current_spot + (adjusted_width * 0.5)
                
                # Adjust for nearby key levels
                nearby_levels = self.find_nearby_levels(key_levels, current_spot, adjusted_width)
                if nearby_levels['support']:
                    predicted_low = max(predicted_low, nearby_levels['support']['strike'] - 20)
                if nearby_levels['resistance']:
                    predicted_high = min(predicted_high, nearby_levels['resistance']['strike'] + 20)
                
                # Calculate confidence based on level alignment
                confidence = self.calculate_range_confidence(
                    predicted_low, predicted_high, nearby_levels, volatility_factor
                )
                
                pred_id = f"RANGE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{timeframe}"
                
                pred = Prediction(
                    timestamp=datetime.now(),
                    prediction_type='RANGE',
                    predicted_value={
                        'low': predicted_low,
                        'high': predicted_high,
                        'width': predicted_high - predicted_low
                    },
                    confidence=confidence,
                    factors={
                        'base_range': base_range,
                        'volatility_factor': volatility_factor,
                        'nearby_levels': nearby_levels,
                        'timeframe_multiplier': multiplier
                    },
                    timeframe=timeframe,
                    current_spot=current_spot,
                    prediction_id=pred_id
                )
                range_predictions.append(pred)
            
            return {
                'range_predictions': range_predictions,
                'volatility_factor': volatility_factor
            }
            
        except Exception as e:
            print(f"Error predicting ranges: {e}")
            return {'range_predictions': []}
    
    def predict_breakouts(self, key_levels: Dict, current_data: pd.DataFrame,
                         current_spot: float) -> Dict:
        """
        Predict breakout/breakdown probabilities
        """
        try:
            breakout_analysis = key_levels.get('breakout_analysis', {})
            breakout_predictions = []
            
            # Upside breakout prediction
            if breakout_analysis.get('next_resistance'):
                resistance = breakout_analysis['next_resistance']
                upside_prob = self.calculate_breakout_probability(
                    current_data, resistance, current_spot, 'UPSIDE'
                )
                
                pred_id = f"BREAKOUT_UP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                pred = Prediction(
                    timestamp=datetime.now(),
                    prediction_type='BREAKOUT',
                    predicted_value={
                        'direction': 'UPSIDE',
                        'level': resistance['strike'],
                        'probability': upside_prob['probability'],
                        'target_after_break': upside_prob['target']
                    },
                    confidence=upside_prob['confidence'],
                    factors=upside_prob['factors'],
                    timeframe='1hr',
                    current_spot=current_spot,
                    prediction_id=pred_id
                )
                breakout_predictions.append(pred)
            
            # Downside breakdown prediction
            if breakout_analysis.get('next_support'):
                support = breakout_analysis['next_support']
                downside_prob = self.calculate_breakout_probability(
                    current_data, support, current_spot, 'DOWNSIDE'
                )
                
                pred_id = f"BREAKOUT_DOWN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                pred = Prediction(
                    timestamp=datetime.now(),
                    prediction_type='BREAKOUT',
                    predicted_value={
                        'direction': 'DOWNSIDE',
                        'level': support['strike'],
                        'probability': downside_prob['probability'],
                        'target_after_break': downside_prob['target']
                    },
                    confidence=downside_prob['confidence'],
                    factors=downside_prob['factors'],
                    timeframe='1hr',
                    current_spot=current_spot,
                    prediction_id=pred_id
                )
                breakout_predictions.append(pred)
            
            return {
                'breakout_predictions': breakout_predictions,
                'breakout_summary': self.summarize_breakout_potential(breakout_predictions)
            }
            
        except Exception as e:
            print(f"Error predicting breakouts: {e}")
            return {'breakout_predictions': []}
    
    def predict_level_behavior(self, key_levels: Dict, current_data: pd.DataFrame,
                              current_spot: float) -> Dict:
        """
        Predict how key levels will behave (HOLD/BREAK)
        """
        try:
            level_predictions = []
            classified_levels = key_levels.get('key_levels', {})
            
            # Analyze critical and major levels
            levels_to_analyze = (
                classified_levels.get('critical_levels', []) +
                classified_levels.get('major_levels', [])
            )
            
            for level in levels_to_analyze[:5]:  # Top 5 most important levels
                level_strike = level['strike']
                distance_from_spot = abs(level_strike - current_spot)
                distance_pct = (distance_from_spot / current_spot) * 100
                
                # Only predict for levels within reasonable distance
                if distance_pct <= 5:  # Within 5% of current spot
                    hold_probability = self.calculate_level_hold_probability(
                        level, current_data, current_spot
                    )
                    
                    pred_id = f"LEVEL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(level_strike)}"
                    
                    pred = Prediction(
                        timestamp=datetime.now(),
                        prediction_type='LEVEL',
                        predicted_value={
                            'level_strike': level_strike,
                            'behavior': 'HOLD' if hold_probability['probability'] > 0.6 else 'BREAK',
                            'hold_probability': hold_probability['probability'],
                            'level_type': level['level_type']
                        },
                        confidence=hold_probability['confidence'],
                        factors=hold_probability['factors'],
                        timeframe='1hr',
                        current_spot=current_spot,
                        prediction_id=pred_id
                    )
                    level_predictions.append(pred)
            
            return {
                'level_predictions': level_predictions,
                'levels_analyzed': len(level_predictions)
            }
            
        except Exception as e:
            print(f"Error predicting level behavior: {e}")
            return {'level_predictions': []}
    
    def predict_volatility_changes(self, current_data: pd.DataFrame,
                                 baseline_data: pd.DataFrame) -> Dict:
        """
        Predict changes in implied volatility
        """
        try:
            # Calculate current volatility metrics
            current_vega_sum = abs(current_data['Vega'].sum())
            baseline_vega_sum = abs(baseline_data['Vega'].sum()) if not baseline_data.empty else current_vega_sum
            
            vega_change_pct = ((current_vega_sum - baseline_vega_sum) / baseline_vega_sum * 100) if baseline_vega_sum > 0 else 0
            
            # Predict volatility direction
            if vega_change_pct > 15:
                vol_direction = 'INCREASING'
                confidence = 'HIGH' if vega_change_pct > 25 else 'MEDIUM'
            elif vega_change_pct < -15:
                vol_direction = 'DECREASING'
                confidence = 'HIGH' if vega_change_pct < -25 else 'MEDIUM'
            else:
                vol_direction = 'STABLE'
                confidence = 'MEDIUM'
            
            pred_id = f"VOL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            pred = Prediction(
                timestamp=datetime.now(),
                prediction_type='VOLATILITY',
                predicted_value={
                    'direction': vol_direction,
                    'vega_change_pct': vega_change_pct,
                    'expected_iv_change': vega_change_pct * 0.1  # Rough IV change estimate
                },
                confidence=confidence,
                factors={
                    'current_vega_sum': current_vega_sum,
                    'baseline_vega_sum': baseline_vega_sum,
                    'vega_momentum': vega_change_pct
                },
                timeframe='1hr',
                current_spot=0,  # Volatility prediction doesn't need spot
                prediction_id=pred_id
            )
            
            return {
                'volatility_prediction': pred,
                'vega_analysis': {
                    'current_sum': current_vega_sum,
                    'baseline_sum': baseline_vega_sum,
                    'change_pct': vega_change_pct
                }
            }
            
        except Exception as e:
            print(f"Error predicting volatility: {e}")
            return {'volatility_prediction': None}
    
    def record_actual_outcome(self, prediction_id: str, actual_value: Union[str, float, Dict],
                            outcome_timestamp: datetime = None) -> bool:
        """
        Record actual outcome for a prediction to calculate accuracy
        """
        try:
            if outcome_timestamp is None:
                outcome_timestamp = datetime.now()
            
            # Find the original prediction
            prediction = self.find_prediction_by_id(prediction_id)
            if not prediction:
                print(f"Prediction {prediction_id} not found")
                return False
            
            # Calculate accuracy based on prediction type
            accuracy = self.calculate_prediction_accuracy(prediction, actual_value)
            success = accuracy > 0.5  # Consider >50% accuracy as success
            
            # Store result
            result = PredictionResult(
                prediction_id=prediction_id,
                actual_value=actual_value,
                accuracy=accuracy,
                outcome_timestamp=outcome_timestamp,
                success=success
            )
            
            self.store_result(result)
            self.results.append(result)
            
            # Update accuracy history
            self.update_accuracy_history(prediction.prediction_type, accuracy)
            
            # Trigger learning if enough data
            if len(self.results) >= self.min_predictions_for_learning:
                self.trigger_learning_update()
            
            return True
            
        except Exception as e:
            print(f"Error recording outcome: {e}")
            return False
    
    def calculate_prediction_accuracy(self, prediction: Prediction, actual_value: Union[str, float, Dict]) -> float:
        """
        Calculate accuracy score for different prediction types
        """
        try:
            pred_type = prediction.prediction_type
            predicted = prediction.predicted_value
            
            if pred_type == 'DIRECTION':
                # Direction accuracy: exact match = 1.0, partial match = 0.5
                if predicted == actual_value:
                    return 1.0
                elif (predicted in ['BULLISH', 'BEARISH'] and actual_value == 'SIDEWAYS') or \
                     (predicted == 'SIDEWAYS' and actual_value in ['BULLISH', 'BEARISH']):
                    return 0.3  # Partial credit for sideways vs directional
                else:
                    return 0.0
            
            elif pred_type == 'RANGE':
                # Range accuracy: how much of actual movement was within predicted range
                if isinstance(actual_value, dict) and 'actual_low' in actual_value:
                    pred_low = predicted['low']
                    pred_high = predicted['high']
                    actual_low = actual_value['actual_low']
                    actual_high = actual_value['actual_high']
                    
                    # Calculate overlap percentage
                    overlap_low = max(pred_low, actual_low)
                    overlap_high = min(pred_high, actual_high)
                    
                    if overlap_high > overlap_low:
                        overlap_range = overlap_high - overlap_low
                        actual_range = actual_high - actual_low
                        accuracy = min(overlap_range / actual_range, 1.0) if actual_range > 0 else 0.0
                    else:
                        accuracy = 0.0
                    
                    return accuracy
                else:
                    return 0.0
            
            elif pred_type == 'BREAKOUT':
                # Breakout accuracy: did the predicted breakout occur?
                if isinstance(actual_value, dict):
                    predicted_direction = predicted['direction']
                    actual_broke = actual_value.get('broke', False)
                    actual_direction = actual_value.get('direction', '')
                    
                    if actual_broke and predicted_direction == actual_direction:
                        return 1.0
                    elif not actual_broke and predicted.get('probability', 'LOW') == 'LOW':
                        return 0.7  # Correctly predicted no breakout
                    else:
                        return 0.0
                else:
                    return 0.0
            
            elif pred_type == 'LEVEL':
                # Level behavior accuracy
                predicted_behavior = predicted['behavior']
                actual_behavior = actual_value
                
                return 1.0 if predicted_behavior == actual_behavior else 0.0
            
            elif pred_type == 'VOLATILITY':
                # Volatility direction accuracy
                predicted_direction = predicted['direction']
                actual_direction = actual_value
                
                return 1.0 if predicted_direction == actual_direction else 0.0
            
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error calculating accuracy: {e}")
            return 0.0
    
    def trigger_learning_update(self):
        """
        Trigger model learning based on recent performance
        """
        try:
            # Calculate recent accuracy by prediction type
            recent_results = self.results[-20:]  # Last 20 predictions
            accuracy_by_type = {}
            
            for result in recent_results:
                prediction = self.find_prediction_by_id(result.prediction_id)
                if prediction:
                    pred_type = prediction.prediction_type
                    if pred_type not in accuracy_by_type:
                        accuracy_by_type[pred_type] = []
                    accuracy_by_type[pred_type].append(result.accuracy)
            
            # Adjust weights for underperforming factors
            adjustments_made = False
            
            for pred_type, accuracies in accuracy_by_type.items():
                avg_accuracy = statistics.mean(accuracies)
                
                if avg_accuracy < self.weight_adjustment_threshold:
                    # Reduce weights of factors in poor predictions
                    poor_predictions = [r for r in recent_results 
                                     if self.find_prediction_by_id(r.prediction_id).prediction_type == pred_type 
                                     and r.accuracy < 0.5]
                    
                    self.adjust_weights_from_poor_predictions(poor_predictions)
                    adjustments_made = True
            
            if adjustments_made:
                print("Model weights adjusted based on recent performance")
                self.save_model_state()
            
        except Exception as e:
            print(f"Error in learning update: {e}")
    
    def adjust_weights_from_poor_predictions(self, poor_predictions: List[PredictionResult]):
        """
        Adjust model weights based on poor predictions
        """
        try:
            for result in poor_predictions:
                prediction = self.find_prediction_by_id(result.prediction_id)
                if prediction and hasattr(prediction, 'factors'):
                    factors = prediction.factors
                    
                    # Identify which factors were most confident in wrong predictions
                    for factor_name, factor_data in factors.items():
                        if isinstance(factor_data, dict) and 'confidence' in factor_data:
                            if factor_data['confidence'] in ['HIGH', 'VERY_HIGH']:
                                # Reduce weight of overconfident factors
                                if factor_name in self.weights:
                                    old_weight = self.weights[factor_name]
                                    self.weights[factor_name] *= (1 - self.learning_rate)
                                    print(f"Reduced {factor_name} weight from {old_weight:.3f} to {self.weights[factor_name]:.3f}")
            
            # Normalize weights to sum to 1.0
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                for key in self.weights:
                    self.weights[key] /= total_weight
                    
        except Exception as e:
            print(f"Error adjusting weights: {e}")
    
    def generate_performance_report(self, days_back: int = 7) -> Dict:
        """
        Generate comprehensive performance report
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_results = [r for r in self.results if r.outcome_timestamp >= cutoff_date]
            
            if not recent_results:
                return {'error': 'No recent results available'}
            
            # Overall metrics
            total_predictions = len(recent_results)
            successful_predictions = sum(1 for r in recent_results if r.success)
            overall_accuracy = statistics.mean([r.accuracy for r in recent_results])
            
            # Accuracy by prediction type
            accuracy_by_type = {}
            for result in recent_results:
                prediction = self.find_prediction_by_id(result.prediction_id)
                if prediction:
                    pred_type = prediction.prediction_type
                    if pred_type not in accuracy_by_type:
                        accuracy_by_type[pred_type] = []
                    accuracy_by_type[pred_type].append(result.accuracy)
            
            # Calculate averages
            type_averages = {}
            for pred_type, accuracies in accuracy_by_type.items():
                type_averages[pred_type] = {
                    'average_accuracy': statistics.mean(accuracies),
                    'best_accuracy': max(accuracies),
                    'worst_accuracy': min(accuracies),
                    'total_predictions': len(accuracies),
                    'successful_predictions': sum(1 for a in accuracies if a > 0.5)
                }
            
            # Identify best and worst performing factors
            factor_performance = self.analyze_factor_performance(recent_results)
            
            return {
                'report_period': f"Last {days_back} days",
                'overall_metrics': {
                    'total_predictions': total_predictions,
                    'successful_predictions': successful_predictions,
                    'success_rate': successful_predictions / total_predictions,
                    'average_accuracy': overall_accuracy
                },
                'accuracy_by_type': type_averages,
                'factor_performance': factor_performance,
                'current_weights': self.weights.copy(),
                'recommendations': self.generate_improvement_recommendations(type_averages)
            }
            
        except Exception as e:
            print(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def store_prediction(self, prediction: Prediction):
        """Store prediction in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions 
                (id, timestamp, prediction_type, predicted_value, confidence, factors, timeframe, current_spot)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction.prediction_id,
                prediction.timestamp.isoformat(),
                prediction.prediction_type,
                json.dumps(prediction.predicted_value),
                prediction.confidence,
                json.dumps(prediction.factors),
                prediction.timeframe,
                prediction.current_spot
            ))
            
            conn.commit()
            conn.close()
            
            self.predictions.append(prediction)
            
        except Exception as e:
            print(f"Error storing prediction: {e}")
    
    def store_result(self, result: PredictionResult):
        """Store prediction result in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO results 
                (prediction_id, actual_value, accuracy, outcome_timestamp, success)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                result.prediction_id,
                json.dumps(result.actual_value),
                result.accuracy,
                result.outcome_timestamp.isoformat(),
                1 if result.success else 0
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error storing result: {e}")
    
    def find_prediction_by_id(self, prediction_id: str) -> Optional[Prediction]:
        """Find prediction by ID"""
        for pred in self.predictions:
            if pred.prediction_id == prediction_id:
                return pred
        return None
    
    # Helper methods for analysis components
    def analyze_vega_momentum(self, current_data: pd.DataFrame, baseline_data: pd.DataFrame) -> Dict:
        """Analyze Vega momentum factor"""
        try:
            current_vega = abs(current_data['Vega'].sum())
            baseline_vega = abs(baseline_data['Vega'].sum()) if not baseline_data.empty else current_vega
            
            if baseline_vega > 0:
                vega_change_pct = ((current_vega - baseline_vega) / baseline_vega) * 100
            else:
                vega_change_pct = 0
            
            # Convert to directional score (-1 to +1)
            if abs(vega_change_pct) > 20:
                score = np.sign(vega_change_pct) * 0.8
                confidence = 'HIGH'
            elif abs(vega_change_pct) > 10:
                score = np.sign(vega_change_pct) * 0.5
                confidence = 'MEDIUM'
            else:
                score = 0
                confidence = 'LOW'
            
            return {
                'score': score,
                'confidence': confidence,
                'vega_change_pct': vega_change_pct,
                'current_vega': current_vega,
                'baseline_vega': baseline_vega
            }
            
        except Exception as e:
            return {'score': 0, 'confidence': 'LOW', 'error': str(e)}
    
    def analyze_gamma_acceleration(self, current_data: pd.DataFrame, baseline_data: pd.DataFrame) -> Dict:
        """Analyze Gamma acceleration factor"""
        try:
            current_gamma = abs(current_data['Gamma'].sum())
            baseline_gamma = abs(baseline_data['Gamma'].sum()) if not baseline_data.empty else current_gamma
            
            if baseline_gamma > 0:
                gamma_change_pct = ((current_gamma - baseline_gamma) / baseline_gamma) * 100
            else:
                gamma_change_pct = 0
            
            # Gamma acceleration suggests directional move
            if abs(gamma_change_pct) > 25:
                score = 0.6 if gamma_change_pct > 0 else -0.6  # High gamma = potential volatility
                confidence = 'HIGH'
            elif abs(gamma_change_pct) > 15:
                score = 0.4 if gamma_change_pct > 0 else -0.4
                confidence = 'MEDIUM'
            else:
                score = 0
                confidence = 'LOW'
            
            return {
                'score': score,
                'confidence': confidence,
                'gamma_change_pct': gamma_change_pct,
                'current_gamma': current_gamma,
                'baseline_gamma': baseline_gamma
            }
            
        except Exception as e:
            return {'score': 0, 'confidence': 'LOW', 'error': str(e)}
    
    def analyze_oi_flow_direction(self, current_data: pd.DataFrame, baseline_data: pd.DataFrame) -> Dict:
        """Analyze OI flow direction"""
        try:
            current_oi = current_data['OI'].sum()
            baseline_oi = baseline_data['OI'].sum() if not baseline_data.empty else current_oi
            
            if baseline_oi > 0:
                oi_change_pct = ((current_oi - baseline_oi) / baseline_oi) * 100
            else:
                oi_change_pct = 0
            
            # OI flow analysis (simplified)
            if abs(oi_change_pct) > 15:
                score = np.sign(oi_change_pct) * 0.5
                confidence = 'HIGH'
            elif abs(oi_change_pct) > 8:
                score = np.sign(oi_change_pct) * 0.3
                confidence = 'MEDIUM'
            else:
                score = 0
                confidence = 'LOW'
            
            return {
                'score': score,
                'confidence': confidence,
                'oi_change_pct': oi_change_pct,
                'current_oi': current_oi,
                'baseline_oi': baseline_oi
            }
            
        except Exception as e:
            return {'score': 0, 'confidence': 'LOW', 'error': str(e)}
    
    def analyze_level_proximity_bias(self, key_levels: Dict, current_spot: float) -> Dict:
        """Analyze proximity to key levels"""
        try:
            # Find nearest resistance and support
            classified_levels = key_levels.get('key_levels', {})
            all_levels = (classified_levels.get('critical_levels', []) + 
                         classified_levels.get('major_levels', []))
            
            nearest_resistance = None
            nearest_support = None
            
            for level in all_levels:
                if level['strike'] > current_spot:
                    if not nearest_resistance or level['strike'] < nearest_resistance['strike']:
                        nearest_resistance = level
                elif level['strike'] < current_spot:
                    if not nearest_support or level['strike'] > nearest_support['strike']:
                        nearest_support = level
            
            score = 0
            confidence = 'LOW'
            
            # Bias based on proximity to levels
            if nearest_resistance and nearest_support:
                res_distance = (nearest_resistance['strike'] - current_spot) / current_spot * 100
                sup_distance = (current_spot - nearest_support['strike']) / current_spot * 100
                
                if res_distance < sup_distance:
                    score = -0.3  # Closer to resistance = bearish bias
                    confidence = 'MEDIUM'
                elif sup_distance < res_distance:
                    score = 0.3   # Closer to support = bullish bias
                    confidence = 'MEDIUM'
            
            return {
                'score': score,
                'confidence': confidence,
                'nearest_resistance': nearest_resistance['strike'] if nearest_resistance else None,
                'nearest_support': nearest_support['strike'] if nearest_support else None
            }
            
        except Exception as e:
            return {'score': 0, 'confidence': 'LOW', 'error': str(e)}
    
    def get_fallback_prediction(self, current_spot: float) -> Dict:
        """Fallback prediction when main prediction fails"""
        return {
            'predictions': {
                'direction': {'direction': 'SIDEWAYS', 'confidence': 'LOW'},
                'ranges': {'range_predictions': []},
                'breakouts': {'breakout_predictions': []},
                'levels': {'level_predictions': []},
                'volatility': {'volatility_prediction': None}
            },
            'prediction_timestamp': datetime.now(),
            'current_spot': current_spot,
            'model_confidence': 'LOW',
            'note': 'Fallback prediction due to calculation error'
        }
    
    # Additional helper methods would be implemented here...
    def calculate_current_volatility_factor(self, current_data: pd.DataFrame) -> float:
        """Calculate current volatility factor"""
        return 1.0  # Simplified for now
    
    def find_nearby_levels(self, key_levels: Dict, current_spot: float, range_width: float) -> Dict:
        """Find levels near current spot"""
        return {'support': None, 'resistance': None}  # Simplified for now
    
    def calculate_range_confidence(self, low: float, high: float, levels: Dict, vol_factor: float) -> str:
        """Calculate confidence for range prediction"""
        return 'MEDIUM'  # Simplified for now
    
    def calculate_breakout_probability(self, current_data: pd.DataFrame, level: Dict, 
                                     current_spot: float, direction: str) -> Dict:
        """Calculate breakout probability"""
        return {
            'probability': 'MEDIUM',
            'confidence': 'MEDIUM', 
            'target': level.get('strike', current_spot) * 1.02,
            'factors': {}
        }
    
    def calculate_level_hold_probability(self, level: Dict, current_data: pd.DataFrame, 
                                       current_spot: float) -> Dict:
        """Calculate probability that level will hold"""
        return {
            'probability': 0.6,
            'confidence': 'MEDIUM',
            'factors': {}
        }
    
    def synthesize_overall_sentiment(self, predictions: Dict, current_spot: float) -> Dict:
        """Synthesize overall market sentiment from all predictions"""
        return {
            'overall_sentiment': 'NEUTRAL',
            'confidence': 'MEDIUM',
            'key_drivers': []
        }
    
    def calculate_model_confidence(self, predictions: Dict) -> str:
        """Calculate overall model confidence"""
        return 'MEDIUM'
    
    def summarize_breakout_potential(self, breakout_predictions: List) -> Dict:
        """Summarize breakout potential"""
        return {'summary': 'Neutral breakout potential'}
    
    def update_accuracy_history(self, prediction_type: str, accuracy: float):
        """Update accuracy history"""
        if prediction_type not in self.accuracy_history:
            self.accuracy_history[prediction_type] = []
        self.accuracy_history[prediction_type].append(accuracy)
    
    def analyze_factor_performance(self, results: List) -> Dict:
        """Analyze factor performance"""
        return {'analysis': 'Factor performance analysis'}
    
    def generate_improvement_recommendations(self, type_averages: Dict) -> List[str]:
        """Generate recommendations for improvement"""
        return ['Continue monitoring performance', 'Adjust parameters as needed']
    
    def save_model_state(self):
        """Save current model state"""
        try:
            with open('model_state.json', 'w') as f:
                json.dump({
                    'weights': self.weights,
                    'learning_rate': self.learning_rate,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving model state: {e}")