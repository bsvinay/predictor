import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import json

# Import your existing classes
from advanced_opt_analyzer import AdvancedOptionsAnalyzer
from enhanced_vega_analysis_tab import EnhancedVegaAnalysisTab

# Import new components (assuming they're in separate files)
# from enhanced_range_predictor import EnhancedRangePredictor
# from key_level_detector import KeyLevelDetector  
# from predictive_feedback_system import PredictiveModel

class ComprehensiveOptionsAnalyzer(AdvancedOptionsAnalyzer):
    """
    Comprehensive analyzer that integrates all advanced features:
    - Enhanced range prediction
    - Key level detection
    - Predictive modeling with feedback
    - Real-time learning system
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize new components
        self.range_predictor = EnhancedRangePredictor()
        self.level_detector = KeyLevelDetector()
        self.predictive_model = PredictiveModel()
        
        # Enhanced analysis history
        self.comprehensive_analysis_history = []
        self.prediction_accuracy_tracker = {}
        
        # Real-time feedback system
        self.feedback_enabled = True
        self.auto_learning_enabled = True
        self.prediction_review_interval = 30  # minutes
        
        # Performance metrics
        self.performance_metrics = {
            'range_accuracy': [],
            'direction_accuracy': [],
            'breakout_accuracy': [],
            'level_accuracy': []
        }
    
    def analyze_timestamp_comprehensive(self, timestamp_index: int) -> Dict:
        """
        Comprehensive analysis that includes all advanced features
        """
        try:
            # Get base analysis
            base_analysis = super().analyze_timestamp(timestamp_index)
            
            if not base_analysis:
                return base_analysis
            
            # Get current data
            timestamp = self.unique_timestamps[timestamp_index]
            current_data = self.get_data_for_timestamp(timestamp)
            baseline_data = self.baseline_data if hasattr(self, 'baseline_data') else pd.DataFrame()
            
            # Estimate current spot price
            current_spot = self.estimate_current_spot_price(current_data)
            
            # 1. Enhanced Range Prediction
            range_prediction = self.range_predictor.predict_daily_range(
                current_data, baseline_data, current_spot
            )
            
            # 2. Key Level Detection
            key_levels = self.level_detector.detect_all_key_levels(
                current_data, current_spot
            )
            
            # 3. Comprehensive Predictions
            predictions = self.predictive_model.make_comprehensive_prediction(
                current_data, baseline_data, key_levels, range_prediction, current_spot
            )
            
            # 4. Market Regime Analysis
            market_regime = self.analyze_market_regime(current_data, baseline_data, key_levels)
            
            # 5. Risk Assessment
            risk_assessment = self.assess_market_risks(
                current_data, range_prediction, key_levels, predictions
            )
            
            # 6. Trading Opportunities
            opportunities = self.identify_trading_opportunities(
                predictions, key_levels, range_prediction, current_spot
            )
            
            # Combine all analyses
            comprehensive_analysis = {
                **base_analysis,  # Include original analysis
                'enhanced_range': range_prediction,
                'key_levels': key_levels,
                'predictions': predictions,
                'market_regime': market_regime,
                'risk_assessment': risk_assessment,
                'trading_opportunities': opportunities,
                'current_spot': current_spot,
                'analysis_timestamp': datetime.now(),
                'comprehensive_confidence': self.calculate_comprehensive_confidence(
                    range_prediction, key_levels, predictions
                )
            }
            
            # Store for history and learning
            self.comprehensive_analysis_history.append(comprehensive_analysis)
            
            # Schedule prediction review if enabled
            if self.feedback_enabled:
                self.schedule_prediction_review(predictions, timestamp_index)
            
            return comprehensive_analysis
            
        except Exception as e:
            print(f"Error in comprehensive analysis: {e}")
            return base_analysis or {}
    
    def analyze_market_regime(self, current_data: pd.DataFrame, baseline_data: pd.DataFrame, 
                            key_levels: Dict) -> Dict:
        """
        Analyze current market regime (Trending/Range-bound/Volatile/Calm)
        """
        try:
            # Volatility analysis
            current_vega = abs(current_data['Vega'].sum())
            baseline_vega = abs(baseline_data['Vega'].sum()) if not baseline_data.empty else current_vega
            
            vega_ratio = current_vega / baseline_vega if baseline_vega > 0 else 1.0
            
            # Trend analysis using Greeks momentum
            vega_analysis = self.calculate_vega_based_sentiment(current_data)
            trend_strength = abs(vega_analysis.get('strength', 0))
            
            # Range analysis using key levels
            breakout_analysis = key_levels.get('breakout_analysis', {})
            range_bound_prob = breakout_analysis.get('range_bound_probability', {}).get('probability', 'LOW')
            
            # Determine regime
            if vega_ratio > 1.5:  # High volatility
                if trend_strength > 40:
                    regime = 'VOLATILE_TRENDING'
                else:
                    regime = 'VOLATILE_RANGING'
            else:  # Normal/Low volatility
                if trend_strength > 30:
                    regime = 'CALM_TRENDING'
                elif range_bound_prob == 'HIGH':
                    regime = 'CALM_RANGING'
                else:
                    regime = 'TRANSITIONAL'
            
            # Regime-specific insights
            regime_insights = self.get_regime_specific_insights(regime, vega_ratio, trend_strength)
            
            return {
                'regime': regime,
                'volatility_ratio': vega_ratio,
                'trend_strength': trend_strength,
                'confidence': self.calculate_regime_confidence(vega_ratio, trend_strength),
                'insights': regime_insights,
                'regime_duration_estimate': self.estimate_regime_duration(regime),
                'regime_change_probability': self.estimate_regime_change_probability(regime, vega_ratio)
            }
            
        except Exception as e:
            print(f"Error analyzing market regime: {e}")
            return {'regime': 'UNKNOWN', 'confidence': 'LOW'}
    
    def assess_market_risks(self, current_data: pd.DataFrame, range_prediction: Dict,
                           key_levels: Dict, predictions: Dict) -> Dict:
        """
        Assess various market risks
        """
        try:
            risks = {}
            
            # 1. Volatility Risk
            volatility_pred = predictions.get('predictions', {}).get('volatility', {})
            vol_direction = volatility_pred.get('volatility_prediction', {})
            
            if isinstance(vol_direction, dict):
                vol_change = vol_direction.get('predicted_value', {}).get('direction', 'STABLE')
                if vol_change == 'INCREASING':
                    risks['volatility_risk'] = 'HIGH'
                elif vol_change == 'DECREASING':
                    risks['volatility_risk'] = 'LOW'
                else:
                    risks['volatility_risk'] = 'MEDIUM'
            else:
                risks['volatility_risk'] = 'MEDIUM'
            
            # 2. Breakout Risk
            breakout_preds = predictions.get('predictions', {}).get('breakouts', {}).get('breakout_predictions', [])
            high_prob_breakouts = [b for b in breakout_preds 
                                 if b.predicted_value.get('probability', 'LOW') in ['HIGH', 'VERY_HIGH']]
            
            if len(high_prob_breakouts) > 0:
                risks['breakout_risk'] = 'HIGH'
            elif len(breakout_preds) > 0:
                risks['breakout_risk'] = 'MEDIUM'
            else:
                risks['breakout_risk'] = 'LOW'
            
            # 3. Level Failure Risk
            level_preds = predictions.get('predictions', {}).get('levels', {}).get('level_predictions', [])
            weak_levels = [l for l in level_preds 
                          if l.predicted_value.get('behavior') == 'BREAK']
            
            if len(weak_levels) > 2:
                risks['level_failure_risk'] = 'HIGH'
            elif len(weak_levels) > 0:
                risks['level_failure_risk'] = 'MEDIUM'
            else:
                risks['level_failure_risk'] = 'LOW'
            
            # 4. Range Expansion Risk
            range_width = range_prediction.get('range_width', 0)
            current_spot = range_prediction.get('current_spot', 0)
            
            if current_spot > 0:
                range_pct = (range_width / current_spot) * 100
                if range_pct > 6:
                    risks['range_expansion_risk'] = 'HIGH'
                elif range_pct > 4:
                    risks['range_expansion_risk'] = 'MEDIUM'
                else:
                    risks['range_expansion_risk'] = 'LOW'
            else:
                risks['range_expansion_risk'] = 'MEDIUM'
            
            # 5. Overall Risk Score
            risk_scores = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
            total_risk_score = sum(risk_scores.get(risk, 2) for risk in risks.values())
            max_possible_score = len(risks) * 3
            
            overall_risk_pct = (total_risk_score / max_possible_score) * 100
            
            if overall_risk_pct > 75:
                overall_risk = 'HIGH'
            elif overall_risk_pct > 50:
                overall_risk = 'MEDIUM'
            else:
                overall_risk = 'LOW'
            
            return {
                'individual_risks': risks,
                'overall_risk': overall_risk,
                'risk_score': overall_risk_pct,
                'risk_factors': self.identify_primary_risk_factors(risks),
                'risk_mitigation': self.suggest_risk_mitigation(risks)
            }
            
        except Exception as e:
            print(f"Error assessing risks: {e}")
            return {'overall_risk': 'MEDIUM', 'individual_risks': {}}
    
    def identify_trading_opportunities(self, predictions: Dict, key_levels: Dict,
                                     range_prediction: Dict, current_spot: float) -> Dict:
        """
        Identify potential trading opportunities
        """
        try:
            opportunities = []
            
            # Direction-based opportunities
            direction_pred = predictions.get('predictions', {}).get('direction', {})
            if direction_pred.get('confidence') in ['HIGH', 'VERY_HIGH']:
                direction = direction_pred.get('direction')
                if direction in ['BULLISH', 'BEARISH']:
                    opportunities.append({
                        'type': 'DIRECTIONAL',
                        'direction': direction,
                        'confidence': direction_pred.get('confidence'),
                        'entry_rationale': f"High confidence {direction.lower()} prediction",
                        'risk_level': 'MEDIUM'
                    })
            
            # Range-based opportunities
            range_bound_prob = key_levels.get('breakout_analysis', {}).get('range_bound_probability', {})
            if range_bound_prob.get('probability') == 'HIGH':
                opportunities.append({
                    'type': 'RANGE_TRADING',
                    'range_low': range_bound_prob.get('support_level'),
                    'range_high': range_bound_prob.get('resistance_level'),
                    'confidence': 'HIGH',
                    'entry_rationale': 'Strong range-bound probability',
                    'risk_level': 'LOW'
                })
            
            # Breakout opportunities
            breakout_preds = predictions.get('predictions', {}).get('breakouts', {}).get('breakout_predictions', [])
            for breakout in breakout_preds:
                if breakout.predicted_value.get('probability') in ['HIGH', 'VERY_HIGH']:
                    opportunities.append({
                        'type': 'BREAKOUT',
                        'direction': breakout.predicted_value.get('direction'),
                        'level': breakout.predicted_value.get('level'),
                        'target': breakout.predicted_value.get('target_after_break'),
                        'confidence': breakout.confidence,
                        'entry_rationale': f"High probability {breakout.predicted_value.get('direction').lower()} breakout",
                        'risk_level': 'HIGH'
                    })
            
            # Level-based opportunities
            level_preds = predictions.get('predictions', {}).get('levels', {}).get('level_predictions', [])
            strong_levels = [l for l in level_preds if l.confidence in ['HIGH', 'VERY_HIGH']]
            
            for level in strong_levels:
                if level.predicted_value.get('behavior') == 'HOLD':
                    level_type = level.predicted_value.get('level_type')
                    opportunities.append({
                        'type': 'LEVEL_PLAY',
                        'level': level.predicted_value.get('level_strike'),
                        'level_type': level_type,
                        'confidence': level.confidence,
                        'entry_rationale': f"Strong {level_type.lower()} level expected to hold",
                        'risk_level': 'MEDIUM'
                    })
            
            # Volatility opportunities
            vol_pred = predictions.get('predictions', {}).get('volatility', {}).get('volatility_prediction')
            if vol_pred and vol_pred.confidence in ['HIGH', 'VERY_HIGH']:
                vol_direction = vol_pred.predicted_value.get('direction')
                if vol_direction in ['INCREASING', 'DECREASING']:
                    opportunities.append({
                        'type': 'VOLATILITY',
                        'direction': vol_direction,
                        'confidence': vol_pred.confidence,
                        'entry_rationale': f"High confidence volatility {vol_direction.lower()}",
                        'risk_level': 'MEDIUM'
                    })
            
            # Rank opportunities by confidence and risk-adjusted return potential
            ranked_opportunities = self.rank_opportunities(opportunities)
            
            return {
                'opportunities': ranked_opportunities,
                'total_opportunities': len(opportunities),
                'high_confidence_opportunities': len([o for o in opportunities if o['confidence'] in ['HIGH', 'VERY_HIGH']]),
                'recommendation': self.generate_trading_recommendation(ranked_opportunities),
                'market_timing': self.assess_market_timing(predictions, key_levels)
            }
            
        except Exception as e:
            print(f"Error identifying opportunities: {e}")
            return {'opportunities': [], 'total_opportunities': 0}
    
    def schedule_prediction_review(self, predictions: Dict, timestamp_index: int):
        """
        Schedule automatic review of predictions for feedback learning
        """
        try:
            if not self.feedback_enabled:
                return
            
            # Schedule review for each prediction
            all_predictions = []
            for pred_category, pred_data in predictions.get('predictions', {}).items():
                if isinstance(pred_data, dict) and 'predictions' in pred_data:
                    all_predictions.extend(pred_data['predictions'])
                elif hasattr(pred_data, 'prediction_id'):
                    all_predictions.append(pred_data)
            
            # Schedule review based on prediction timeframe
            for pred in all_predictions:
                if hasattr(pred, 'timeframe') and hasattr(pred, 'prediction_id'):
                    timeframe_minutes = self.convert_timeframe_to_minutes(pred.timeframe)
                    review_time = datetime.now() + timedelta(minutes=timeframe_minutes)
                    
                    # Store for later review (in a real system, you'd use a task scheduler)
                    self.prediction_accuracy_tracker[pred.prediction_id] = {
                        'prediction': pred,
                        'review_time': review_time,
                        'timestamp_index': timestamp_index,
                        'reviewed': False
                    }
                    
        except Exception as e:
            print(f"Error scheduling prediction review: {e}")
    
    def review_and_update_predictions(self):
        """
        Review predictions that are due and update model based on accuracy
        """
        try:
            current_time = datetime.now()
            
            for pred_id, pred_info in self.prediction_accuracy_tracker.items():
                if not pred_info['reviewed'] and current_time >= pred_info['review_time']:
                    # Get actual outcome
                    actual_outcome = self.get_actual_outcome(pred_info['prediction'], pred_info['timestamp_index'])
                    
                    if actual_outcome is not None:
                        # Record the outcome
                        success = self.predictive_model.record_actual_outcome(
                            pred_id, actual_outcome, current_time
                        )
                        
                        if success:
                            pred_info['reviewed'] = True
                            print(f"Reviewed prediction {pred_id} with outcome: {actual_outcome}")
                    
        except Exception as e:
            print(f"Error reviewing predictions: {e}")
    
    def get_actual_outcome(self, prediction, timestamp_index: int):
        """
        Determine actual outcome for a prediction
        """
        try:
            # This would need to be implemented based on your data structure
            # For now, return None to indicate no outcome available yet
            return None
            
        except Exception as e:
            print(f"Error getting actual outcome: {e}")
            return None
    
    def calculate_comprehensive_confidence(self, range_prediction: Dict, key_levels: Dict, predictions: Dict) -> str:
        """
        Calculate overall confidence in the comprehensive analysis
        """
        try:
            confidence_scores = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'VERY_HIGH': 4}
            
            # Get individual confidences
            range_conf = confidence_scores.get(range_prediction.get('confidence', 'LOW'), 1)
            
            # Average prediction confidences
            pred_confidences = []
            for pred_category, pred_data in predictions.get('predictions', {}).items():
                if isinstance(pred_data, dict):
                    if 'confidence' in pred_data:
                        pred_confidences.append(confidence_scores.get(pred_data['confidence'], 1))
                    elif 'predictions' in pred_data:
                        for pred in pred_data['predictions']:
                            if hasattr(pred, 'confidence'):
                                pred_confidences.append(confidence_scores.get(pred.confidence, 1))
            
            avg_pred_conf = sum(pred_confidences) / len(pred_confidences) if pred_confidences else 1
            
            # Combined confidence
            overall_conf = (range_conf + avg_pred_conf) / 2
            
            if overall_conf >= 3.5:
                return 'VERY_HIGH'
            elif overall_conf >= 2.5:
                return 'HIGH'
            elif overall_conf >= 1.5:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception as e:
            print(f"Error calculating comprehensive confidence: {e}")
            return 'LOW'
    
    def estimate_current_spot_price(self, current_data: pd.DataFrame) -> float:
        """
        Estimate current spot price from options data
        """
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
    
    # Helper methods
    def get_regime_specific_insights(self, regime: str, volatility_ratio: float, trend_strength: float) -> List[str]:
        """Get insights specific to current market regime"""
        insights = []
        
        if regime == 'VOLATILE_TRENDING':
            insights.append("High volatility trending market - expect large directional moves")
            insights.append("Breakout trades favored over range trades")
            insights.append("Risk management crucial due to volatility")
            
        elif regime == 'VOLATILE_RANGING':
            insights.append("High volatility range-bound - expect whipsaws")
            insights.append("Fade extremes, avoid breakout trades")
            insights.append("Volatility selling strategies may work")
            
        elif regime == 'CALM_TRENDING':
            insights.append("Low volatility trending - sustained moves likely")
            insights.append("Trend following strategies favored")
            insights.append("Lower risk environment for directional trades")
            
        elif regime == 'CALM_RANGING':
            insights.append("Low volatility range-bound - tight trading range")
            insights.append("Range trading strategies optimal")
            insights.append("Volatility buying at extremes")
            
        else:
            insights.append("Market in transition - wait for clearer signals")
            insights.append("Reduced position sizing recommended")
        
        return insights
    
    def calculate_regime_confidence(self, volatility_ratio: float, trend_strength: float) -> str:
        """Calculate confidence in regime identification"""
        if abs(volatility_ratio - 1.0) > 0.5 and trend_strength > 25:
            return 'HIGH'
        elif abs(volatility_ratio - 1.0) > 0.3 or trend_strength > 15:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def estimate_regime_duration(self, regime: str) -> str:
        """Estimate how long the regime might last"""
        duration_estimates = {
            'VOLATILE_TRENDING': '2-4 hours',
            'VOLATILE_RANGING': '1-3 hours', 
            'CALM_TRENDING': '4-8 hours',
            'CALM_RANGING': '3-6 hours',
            'TRANSITIONAL': '30-90 minutes'
        }
        return duration_estimates.get(regime, '1-3 hours')
    
    def estimate_regime_change_probability(self, regime: str, volatility_ratio: float) -> str:
        """Estimate probability of regime change"""
        if volatility_ratio > 2.0 or volatility_ratio < 0.5:
            return 'HIGH'
        elif volatility_ratio > 1.5 or volatility_ratio < 0.7:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def identify_primary_risk_factors(self, risks: Dict) -> List[str]:
        """Identify primary risk factors"""
        high_risks = [risk_type for risk_type, level in risks.items() if level == 'HIGH']
        return high_risks if high_risks else ['General market uncertainty']
    
    def suggest_risk_mitigation(self, risks: Dict) -> List[str]:
        """Suggest risk mitigation strategies"""
        suggestions = []
        
        if risks.get('volatility_risk') == 'HIGH':
            suggestions.append("Reduce position sizes due to high volatility")
            suggestions.append("Consider volatility-adjusted stops")
            
        if risks.get('breakout_risk') == 'HIGH':
            suggestions.append("Prepare for potential breakout moves")
            suggestions.append("Have breakout and failure contingency plans")
            
        if risks.get('level_failure_risk') == 'HIGH':
            suggestions.append("Don't rely heavily on support/resistance")
            suggestions.append("Use wider stops around key levels")
            
        if not suggestions:
            suggestions.append("Standard risk management protocols")
            
        return suggestions
    
    def rank_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Rank opportunities by attractiveness"""
        # Simple ranking by confidence and inverse risk
        confidence_scores = {'VERY_HIGH': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        risk_scores = {'LOW': 3, 'MEDIUM': 2, 'HIGH': 1}
        
        for opp in opportunities:
            conf_score = confidence_scores.get(opp.get('confidence', 'LOW'), 1)
            risk_score = risk_scores.get(opp.get('risk_level', 'HIGH'), 1)
            opp['attractiveness_score'] = conf_score * risk_score
        
        return sorted(opportunities, key=lambda x: x.get('attractiveness_score', 0), reverse=True)
    
    def generate_trading_recommendation(self, opportunities: List[Dict]) -> str:
        """Generate overall trading recommendation"""
        if not opportunities:
            return "No clear opportunities identified - remain cautious"
        
        top_opp = opportunities[0]
        
        if top_opp.get('attractiveness_score', 0) >= 6:
            return f"Strong {top_opp['type']} opportunity identified"
        elif top_opp.get('attractiveness_score', 0) >= 4:
            return f"Moderate {top_opp['type']} opportunity - proceed with caution"
        else:
            return "Weak opportunities - consider staying sidelines"
    
    def assess_market_timing(self, predictions: Dict, key_levels: Dict) -> str:
        """Assess overall market timing"""
        # Simplified timing assessment
        direction_conf = predictions.get('predictions', {}).get('direction', {}).get('confidence', 'LOW')
        
        if direction_conf in ['HIGH', 'VERY_HIGH']:
            return 'GOOD'
        elif direction_conf == 'MEDIUM':
            return 'NEUTRAL'
        else:
            return 'POOR'
    
    def convert_timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            '30min': 30,
            '1hr': 60,
            '3hr': 180,
            'EOD': 480  # End of day
        }
        return timeframe_map.get(timeframe, 60)


class ComprehensiveOptionsGUI:
    """
    Enhanced GUI that integrates all advanced features with your existing interface
    """
    
    def __init__(self):
        self.analyzer = ComprehensiveOptionsAnalyzer()
        self.current_timestamp_index = 0
        self.auto_refresh_enabled = False
        self.refresh_interval = 30  # seconds
        self.setup_gui()
        
        # Background tasks
        self.background_tasks_running = False
        self.start_background_tasks()
    
    def setup_gui(self):
        """Setup the comprehensive GUI"""
        self.root = tk.Tk()
        self.root.title("Comprehensive Options Analysis Platform")
        self.root.geometry("1800x1200")
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Setup control panel
        self.setup_control_panel(main_frame)
        
        # Create comprehensive notebook
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Setup all tabs
        self.setup_comprehensive_tabs()
        
        # Status bar
        self.setup_status_bar(main_frame)
    
    def setup_control_panel(self, parent):
        """Setup enhanced control panel"""
        control_frame = ttk.LabelFrame(parent, text="Control Panel")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File controls
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="Load CSV File", command=self.load_file).pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        # Time controls
        time_frame = ttk.Frame(control_frame)
        time_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(time_frame, text="Time:").pack(side=tk.LEFT)
        self.time_var = tk.DoubleVar()
        self.time_slider = ttk.Scale(time_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                   variable=self.time_var, command=self.update_analysis)
        self.time_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.time_label = ttk.Label(time_frame, text="Select time point")
        self.time_label.pack(side=tk.RIGHT, padx=5)
        
        # Analysis controls
        analysis_frame = ttk.Frame(control_frame)
        analysis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(analysis_frame, text="Run Full Analysis", 
                  command=self.run_comprehensive_analysis).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(analysis_frame, text="Generate Report", 
                  command=self.generate_performance_report).pack(side=tk.LEFT, padx=5)
        
        # Auto-refresh toggle
        self.auto_refresh_var = tk.BooleanVar()
        ttk.Checkbutton(analysis_frame, text="Auto Refresh", 
                       variable=self.auto_refresh_var, 
                       command=self.toggle_auto_refresh).pack(side=tk.LEFT, padx=5)
        
        # Learning controls
        learning_frame = ttk.Frame(control_frame)
        learning_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(learning_frame, text="Review Predictions", 
                  command=self.review_predictions).pack(side=tk.LEFT, padx=5)
        
        self.learning_status = ttk.Label(learning_frame, text="Learning: Enabled")
        self.learning_status.pack(side=tk.LEFT, padx=10)
    
    def setup_comprehensive_tabs(self):
        """Setup all analysis tabs"""
        
        # 1. Comprehensive Dashboard
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="📊 Dashboard")
        self.setup_dashboard_tab(dashboard_frame)
        
        # 2. Enhanced Range Prediction
        range_frame = ttk.Frame(self.notebook)
        self.notebook.add(range_frame, text="📈 Range Prediction")
        self.setup_range_prediction_tab(range_frame)
        
        # 3. Key Levels Analysis
        levels_frame = ttk.Frame(self.notebook)
        self.notebook.add(levels_frame, text="🎯 Key Levels")
        self.setup_key_levels_tab(levels_frame)
        
        # 4. Predictions & Forecasts
        predictions_frame = ttk.Frame(self.notebook)
        self.notebook.add(predictions_frame, text="🔮 Predictions")
        self.setup_predictions_tab(predictions_frame)
        
        # 5. Risk Assessment
        risk_frame = ttk.Frame(self.notebook)
        self.notebook.add(risk_frame, text="⚠️ Risk Assessment")
        self.setup_risk_assessment_tab(risk_frame)
        
        # 6. Trading Opportunities
        opportunities_frame = ttk.Frame(self.notebook)
        self.notebook.add(opportunities_frame, text="💰 Opportunities")
        self.setup_opportunities_tab(opportunities_frame)
        
        # 7. Performance & Learning
        performance_frame = ttk.Frame(self.notebook)
        self.notebook.add(performance_frame, text="📈 Performance")
        self.setup_performance_tab(performance_frame)
        
        # 8. Original Analysis (for compatibility)
        original_frame = ttk.Frame(self.notebook)
        self.notebook.add(original_frame, text="📋 Original Analysis")
        self.setup_original_analysis_tab(original_frame)
    
    def setup_dashboard_tab(self, parent):
        """Setup comprehensive dashboard"""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Dashboard content
        self.setup_dashboard_content(scrollable_frame)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def setup_dashboard_content(self, parent):
        """Setup dashboard content widgets"""
        
        # Market Summary Section
        summary_frame = ttk.LabelFrame(parent, text="🎯 Market Summary")
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create summary widgets
        self.current_spot_var = tk.StringVar(value="Spot: --")
        self.market_regime_var = tk.StringVar(value="Regime: --")
        self.overall_sentiment_var = tk.StringVar(value="Sentiment: --")
        self.confidence_var = tk.StringVar(value="Confidence: --")
        
        summary_grid = ttk.Frame(summary_frame)
        summary_grid.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(summary_grid, textvariable=self.current_spot_var, font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", padx=10)
        ttk.Label(summary_grid, textvariable=self.market_regime_var, font=("Arial", 12, "bold")).grid(row=0, column=1, sticky="w", padx=10)
        ttk.Label(summary_grid, textvariable=self.overall_sentiment_var, font=("Arial", 12, "bold")).grid(row=1, column=0, sticky="w", padx=10)
        ttk.Label(summary_grid, textvariable=self.confidence_var, font=("Arial", 12, "bold")).grid(row=1, column=1, sticky="w", padx=10)
        
        # Predictions Summary
        predictions_frame = ttk.LabelFrame(parent, text="🔮 Current Predictions")
        predictions_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.predictions_text = tk.Text(predictions_frame, height=8, wrap=tk.WORD)
        pred_scrollbar = ttk.Scrollbar(predictions_frame, orient=tk.VERTICAL, command=self.predictions_text.yview)
        self.predictions_text.configure(yscrollcommand=pred_scrollbar.set)
        self.predictions_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        pred_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Key Levels Summary
        levels_frame = ttk.LabelFrame(parent, text="🎯 Key Levels")
        levels_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.levels_text = tk.Text(levels_frame, height=6, wrap=tk.WORD)
        levels_scrollbar = ttk.Scrollbar(levels_frame, orient=tk.VERTICAL, command=self.levels_text.yview)
        self.levels_text.configure(yscrollcommand=levels_scrollbar.set)
        self.levels_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        levels_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Opportunities Summary
        opportunities_frame = ttk.LabelFrame(parent, text="💰 Top Opportunities")
        opportunities_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.opportunities_text = tk.Text(opportunities_frame, height=6, wrap=tk.WORD)
        opp_scrollbar = ttk.Scrollbar(opportunities_frame, orient=tk.VERTICAL, command=self.opportunities_text.yview)
        self.opportunities_text.configure(yscrollcommand=opp_scrollbar.set)
        self.opportunities_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        opp_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def setup_range_prediction_tab(self, parent):
        """Setup range prediction tab"""
        # Range prediction details will be shown here
        self.range_prediction_text = tk.Text(parent, wrap=tk.WORD)
        range_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.range_prediction_text.yview)
        self.range_prediction_text.configure(yscrollcommand=range_scrollbar.set)
        self.range_prediction_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        range_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def setup_key_levels_tab(self, parent):
        """Setup key levels analysis tab"""
        self.key_levels_text = tk.Text(parent, wrap=tk.WORD)
        levels_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.key_levels_text.yview)
        self.key_levels_text.configure(yscrollcommand=levels_scrollbar.set)
        self.key_levels_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        levels_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def setup_predictions_tab(self, parent):
        """Setup predictions tab"""
        self.predictions_detail_text = tk.Text(parent, wrap=tk.WORD)
        pred_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.predictions_detail_text.yview)
        self.predictions_detail_text.configure(yscrollcommand=pred_scrollbar.set)
        self.predictions_detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        pred_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def setup_risk_assessment_tab(self, parent):
        """Setup risk assessment tab"""
        self.risk_assessment_text = tk.Text(parent, wrap=tk.WORD)
        risk_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.risk_assessment_text.yview)
        self.risk_assessment_text.configure(yscrollcommand=risk_scrollbar.set)
        self.risk_assessment_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        risk_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def setup_opportunities_tab(self, parent):
        """Setup opportunities tab"""
        self.opportunities_detail_text = tk.Text(parent, wrap=tk.WORD)
        opp_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.opportunities_detail_text.yview)
        self.opportunities_detail_text.configure(yscrollcommand=opp_scrollbar.set)
        self.opportunities_detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        opp_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def setup_performance_tab(self, parent):
        """Setup performance tracking tab"""
        self.performance_text = tk.Text(parent, wrap=tk.WORD)
        perf_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.performance_text.yview)
        self.performance_text.configure(yscrollcommand=perf_scrollbar.set)
        self.performance_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        perf_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def setup_original_analysis_tab(self, parent):
        """Setup original analysis tab for backward compatibility"""
        self.original_analysis_text = tk.Text(parent, wrap=tk.WORD)
        orig_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.original_analysis_text.yview)
        self.original_analysis_text.configure(yscrollcommand=orig_scrollbar.set)
        self.original_analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        orig_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def setup_status_bar(self, parent):
        """Setup status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, length=200)
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
    
    def load_file(self):
        """Enhanced file loading with comprehensive analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Options Chain CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.status_var.set("Loading file...")
                self.progress_var.set(10)
                
                # Load data
                self.analyzer.load_data(file_path)
                self.progress_var.set(30)
                
                # Set baseline for comprehensive analysis
                self.analyzer.set_baseline()
                self.progress_var.set(50)
                
                # Process full data
                self.analyzer.process_full_data()
                self.progress_var.set(70)
                
                # Setup time slider
                if len(self.analyzer.unique_timestamps) > 0:
                    self.time_slider.config(to=len(self.analyzer.unique_timestamps)-1)
                    self.time_slider.set(len(self.analyzer.unique_timestamps)-1)
                    self.current_timestamp_index = len(self.analyzer.unique_timestamps)-1
                
                self.progress_var.set(90)
                
                # Run initial comprehensive analysis
                self.run_comprehensive_analysis()
                
                self.file_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                self.progress_var.set(100)
                self.status_var.set("File loaded successfully")
                
                messagebox.showinfo("Success", "Data loaded with comprehensive analysis!")
                
            except Exception as e:
                self.status_var.set("Error loading file")
                self.progress_var.set(0)
                messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis for current timestamp"""
        try:
            if len(self.analyzer.unique_timestamps) == 0:
                return
            
            self.status_var.set("Running comprehensive analysis...")
            self.progress_var.set(0)
            
            # Get comprehensive analysis
            analysis = self.analyzer.analyze_timestamp_comprehensive(self.current_timestamp_index)
            
            if analysis:
                self.progress_var.set(20)
                self.update_dashboard(analysis)
                
                self.progress_var.set(40)
                self.update_range_prediction_display(analysis.get('enhanced_range', {}))
                
                self.progress_var.set(60)
                self.update_key_levels_display(analysis.get('key_levels', {}))
                
                self.progress_var.set(80)
                self.update_predictions_display(analysis.get('predictions', {}))
                
                self.progress_var.set(90)
                self.update_risk_assessment_display(analysis.get('risk_assessment', {}))
                
                self.update_opportunities_display(analysis.get('trading_opportunities', {}))
                
                self.progress_var.set(100)
                self.status_var.set("Comprehensive analysis complete")
            
        except Exception as e:
            self.status_var.set("Error in analysis")
            self.progress_var.set(0)
            print(f"Error in comprehensive analysis: {e}")
    
    def update_analysis(self, *args):
        """Update analysis based on slider position"""
        if len(self.analyzer.unique_timestamps) == 0:
            return
        
        pos = int(self.time_var.get())
        pos = min(pos, len(self.analyzer.unique_timestamps) - 1)
        self.current_timestamp_index = pos
        
        # Update time label
        timestamp = self.analyzer.unique_timestamps[pos]
        self.time_label.config(text=f"Time: {timestamp.strftime('%H:%M:%S')}")
        
        # Run comprehensive analysis for new timestamp
        self.run_comprehensive_analysis()
    
    def update_dashboard(self, analysis: Dict):
        """Update dashboard with analysis results"""
        try:
            # Update summary variables
            current_spot = analysis.get('current_spot', 0)
            self.current_spot_var.set(f"Spot: {current_spot:.2f}")
            
            market_regime = analysis.get('market_regime', {}).get('regime', 'Unknown')
            self.market_regime_var.set(f"Regime: {market_regime}")
            
            # Get overall sentiment from predictions
            predictions = analysis.get('predictions', {})
            direction_pred = predictions.get('predictions', {}).get('direction', {})
            sentiment = direction_pred.get('direction', 'Unknown')
            self.overall_sentiment_var.set(f"Sentiment: {sentiment}")
            
            confidence = analysis.get('comprehensive_confidence', 'Unknown')
            self.confidence_var.set(f"Confidence: {confidence}")
            
            # Update predictions summary
            self.update_predictions_summary(predictions)
            
            # Update levels summary
            self.update_levels_summary(analysis.get('key_levels', {}))
            
            # Update opportunities summary
            self.update_opportunities_summary(analysis.get('trading_opportunities', {}))
            
        except Exception as e:
            print(f"Error updating dashboard: {e}")
    
    def update_predictions_summary(self, predictions: Dict):
        """Update predictions summary in dashboard"""
        try:
            self.predictions_text.delete(1.0, tk.END)
            
            pred_summary = "CURRENT PREDICTIONS SUMMARY\n" + "="*40 + "\n\n"
            
            # Direction prediction
            direction_pred = predictions.get('predictions', {}).get('direction', {})
            if direction_pred:
                pred_summary += f"📈 DIRECTION: {direction_pred.get('direction', 'Unknown')} "
                pred_summary += f"(Confidence: {direction_pred.get('confidence', 'Unknown')})\n\n"
            
            # Range predictions
            range_preds = predictions.get('predictions', {}).get('ranges', {}).get('range_predictions', [])
            if range_preds:
                pred_summary += "📊 RANGES:\n"
                for pred in range_preds[:3]:  # Show top 3 timeframes
                    if hasattr(pred, 'timeframe') and hasattr(pred, 'predicted_value'):
                        pred_val = pred.predicted_value
                        pred_summary += f"  {pred.timeframe}: {pred_val.get('low', 0):.0f} - {pred_val.get('high', 0):.0f}\n"
                pred_summary += "\n"
            
            # Breakout predictions
            breakout_preds = predictions.get('predictions', {}).get('breakouts', {}).get('breakout_predictions', [])
            if breakout_preds:
                pred_summary += "⚡ BREAKOUTS:\n"
                for pred in breakout_preds[:2]:  # Show top 2
                    if hasattr(pred, 'predicted_value'):
                        pred_val = pred.predicted_value
                        pred_summary += f"  {pred_val.get('direction', 'Unknown')} at {pred_val.get('level', 0):.0f} "
                        pred_summary += f"({pred_val.get('probability', 'Unknown')} prob)\n"
                pred_summary += "\n"
            
            self.predictions_text.insert(tk.END, pred_summary)
            
        except Exception as e:
            print(f"Error updating predictions summary: {e}")
    
    def update_levels_summary(self, key_levels: Dict):
        """Update levels summary in dashboard"""
        try:
            self.levels_text.delete(1.0, tk.END)
            
            levels_summary = "KEY LEVELS SUMMARY\n" + "="*30 + "\n\n"
            
            classified_levels = key_levels.get('key_levels', {})
            
            # Critical levels
            critical_levels = classified_levels.get('critical_levels', [])
            if critical_levels:
                levels_summary += "🔴 CRITICAL LEVELS:\n"
                for level in critical_levels[:3]:
                    levels_summary += f"  {level.get('level_type', 'Unknown')}: {level.get('strike', 0):.0f} "
                    levels_summary += f"({level.get('strength', 'Unknown')})\n"
                levels_summary += "\n"
            
            # Major levels
            major_levels = classified_levels.get('major_levels', [])
            if major_levels:
                levels_summary += "🟡 MAJOR LEVELS:\n"
                for level in major_levels[:3]:
                    levels_summary += f"  {level.get('level_type', 'Unknown')}: {level.get('strike', 0):.0f} "
                    levels_summary += f"({level.get('strength', 'Unknown')})\n"
                levels_summary += "\n"
            
            # Max pain
            max_pain = key_levels.get('max_pain_level', {})
            if max_pain.get('max_pain_strike', 0) > 0:
                levels_summary += f"🎯 MAX PAIN: {max_pain.get('max_pain_strike', 0):.0f} "
                levels_summary += f"({max_pain.get('confidence', 'Unknown')} confidence)\n"
            
            self.levels_text.insert(tk.END, levels_summary)
            
        except Exception as e:
            print(f"Error updating levels summary: {e}")
    
    def update_opportunities_summary(self, opportunities: Dict):
        """Update opportunities summary in dashboard"""
        try:
            self.opportunities_text.delete(1.0, tk.END)
            
            opp_summary = "TOP OPPORTUNITIES\n" + "="*25 + "\n\n"
            
            opportunities_list = opportunities.get('opportunities', [])
            
            if opportunities_list:
                opp_summary += f"Total Opportunities: {len(opportunities_list)}\n"
                opp_summary += f"High Confidence: {opportunities.get('high_confidence_opportunities', 0)}\n\n"
                
                opp_summary += "TOP 3 OPPORTUNITIES:\n"
                for i, opp in enumerate(opportunities_list[:3]):
                    opp_summary += f"{i+1}. {opp.get('type', 'Unknown')} - {opp.get('confidence', 'Unknown')} confidence\n"
                    opp_summary += f"   {opp.get('entry_rationale', 'No rationale')}\n\n"
                
                recommendation = opportunities.get('recommendation', 'No recommendation')
                opp_summary += f"💡 RECOMMENDATION: {recommendation}\n"
            else:
                opp_summary += "No clear opportunities identified at this time.\n"
            
            self.opportunities_text.insert(tk.END, opp_summary)
            
        except Exception as e:
            print(f"Error updating opportunities summary: {e}")
    
    # Additional display update methods would be implemented here...
    def update_range_prediction_display(self, enhanced_range: Dict):
        """Update range prediction tab"""
        try:
            self.range_prediction_text.delete(1.0, tk.END)
            
            if not enhanced_range:
                self.range_prediction_text.insert(tk.END, "No range prediction data available")
                return
            
            range_text = "ENHANCED RANGE PREDICTION\n" + "="*50 + "\n\n"
            
            range_text += f"Predicted Range: {enhanced_range.get('predicted_low', 0):.2f} - {enhanced_range.get('predicted_high', 0):.2f}\n"
            range_text += f"Range Width: {enhanced_range.get('range_width', 0):.2f}\n"
            range_text += f"Current Spot: {enhanced_range.get('current_spot', 0):.2f}\n"
            range_text += f"Confidence: {enhanced_range.get('confidence', 'Unknown')}\n\n"
            
            # Factors analysis
            factors = enhanced_range.get('factors', {})
            if factors:
                range_text += "CONTRIBUTING FACTORS:\n" + "-"*30 + "\n"
                for factor, data in factors.items():
                    if isinstance(data, dict):
                        range_text += f"{factor.replace('_', ' ').title()}:\n"
                        for key, value in data.items():
                            if isinstance(value, (int, float)):
                                range_text += f"  {key}: {value:.2f}\n"
                            else:
                                range_text += f"  {key}: {value}\n"
                        range_text += "\n"
            
            # Breakout levels
            breakout_levels = enhanced_range.get('breakout_levels', [])
            if breakout_levels:
                range_text += "BREAKOUT LEVELS:\n" + "-"*20 + "\n"
                for level in breakout_levels:
                    range_text += f"{level.get('type', 'Unknown')}: {level.get('level', 0):.2f} "
                    range_text += f"({level.get('strength', 'Unknown')})\n"
                range_text += "\n"
            
            # Key levels within range
            key_levels = enhanced_range.get('key_levels', [])
            if key_levels:
                range_text += "KEY LEVELS IN RANGE:\n" + "-"*25 + "\n"
                for level in key_levels:
                    range_text += f"{level.get('type', 'Unknown')}: {level.get('level', 0):.2f} "
                    range_text += f"({level.get('source', 'Unknown')})\n"
            
            self.range_prediction_text.insert(tk.END, range_text)
            
        except Exception as e:
            print(f"Error updating range prediction display: {e}")
    
    def update_key_levels_display(self, key_levels: Dict):
        """Update key levels tab display"""
        # Implementation similar to other display methods
        pass
    
    def update_predictions_display(self, predictions: Dict):
        """Update predictions tab display"""
        # Implementation similar to other display methods
        pass
    
    def update_risk_assessment_display(self, risk_assessment: Dict):
        """Update risk assessment tab display"""
        # Implementation similar to other display methods
        pass
    
    def update_opportunities_display(self, opportunities: Dict):
        """Update opportunities tab display"""
        # Implementation similar to other display methods
        pass
    
    def toggle_auto_refresh(self):
        """Toggle auto-refresh functionality"""
        self.auto_refresh_enabled = self.auto_refresh_var.get()
        if self.auto_refresh_enabled:
            self.status_var.set("Auto-refresh enabled")
            self.start_auto_refresh()
        else:
            self.status_var.set("Auto-refresh disabled")
    
    def start_auto_refresh(self):
        """Start auto-refresh loop"""
        if self.auto_refresh_enabled:
            self.run_comprehensive_analysis()
            self.root.after(self.refresh_interval * 1000, self.start_auto_refresh)
    
    def review_predictions(self):
        """Manually trigger prediction review"""
        try:
            self.analyzer.review_and_update_predictions()
            self.status_var.set("Predictions reviewed")
            messagebox.showinfo("Success", "Prediction review completed")
        except Exception as e:
            messagebox.showerror("Error", f"Error reviewing predictions: {e}")
    
    def generate_performance_report(self):
        """Generate and display performance report"""
        try:
            report = self.analyzer.predictive_model.generate_performance_report()
            
            # Display report in performance tab
            self.performance_text.delete(1.0, tk.END)
            
            if 'error' in report:
                self.performance_text.insert(tk.END, f"Error generating report: {report['error']}")
                return
            
            report_text = "PERFORMANCE REPORT\n" + "="*40 + "\n\n"
            
            # Overall metrics
            overall = report.get('overall_metrics', {})
            report_text += f"Total Predictions: {overall.get('total_predictions', 0)}\n"
            report_text += f"Successful Predictions: {overall.get('successful_predictions', 0)}\n"
            report_text += f"Success Rate: {overall.get('success_rate', 0):.1%}\n"
            report_text += f"Average Accuracy: {overall.get('average_accuracy', 0):.3f}\n\n"
            
            # Accuracy by type
            accuracy_by_type = report.get('accuracy_by_type', {})
            if accuracy_by_type:
                report_text += "ACCURACY BY PREDICTION TYPE:\n" + "-"*35 + "\n"
                for pred_type, metrics in accuracy_by_type.items():
                    report_text += f"{pred_type}:\n"
                    report_text += f"  Average Accuracy: {metrics.get('average_accuracy', 0):.3f}\n"
                    report_text += f"  Total Predictions: {metrics.get('total_predictions', 0)}\n"
                    report_text += f"  Success Rate: {metrics.get('successful_predictions', 0)}/{metrics.get('total_predictions', 0)}\n\n"
            
            # Recommendations
            recommendations = report.get('recommendations', [])
            if recommendations:
                report_text += "RECOMMENDATIONS:\n" + "-"*20 + "\n"
                for rec in recommendations:
                    report_text += f"• {rec}\n"
            
            self.performance_text.insert(tk.END, report_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating performance report: {e}")
    
    def start_background_tasks(self):
        """Start background tasks for prediction review and learning"""
        if not self.background_tasks_running:
            self.background_tasks_running = True
            self.background_task_loop()
    
    def background_task_loop(self):
        """Background task loop for prediction review"""
        try:
            if self.background_tasks_running:
                # Review predictions periodically
                self.analyzer.review_and_update_predictions()
                
                # Schedule next review
                self.root.after(60000, self.background_task_loop)  # Every minute
                
        except Exception as e:
            print(f"Error in background tasks: {e}")
    
    def run(self):
        """Run the comprehensive GUI"""
        self.root.mainloop()
        self.background_tasks_running = False


# Usage example
if __name__ == "__main__":
    print("Starting Comprehensive Options Analysis Platform...")
    print("Features:")
    print("- Enhanced Range Prediction with multi-factor analysis")
    print("- Advanced Key Level Detection (Support/Resistance/Breakout)")
    print("- Predictive Modeling with Machine Learning")
    print("- Real-time Feedback and Learning System")
    print("- Risk Assessment and Trading Opportunities")
    print("- Performance Tracking and Optimization")
    print("\nLaunching GUI...")
    
    gui = ComprehensiveOptionsGUI()
    gui.run()