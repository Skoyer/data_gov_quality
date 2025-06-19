import pandas as pd
import numpy as np
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ====
# DIRECTORY CONFIGURATION
# ====

# Configurable root directories
DATA_DIR = Path("data")
PROCESSED_DIR = Path("processed")

def build_output_dirs(file_path: Path):
    """
    Returns the three output folders for a given raw file and
    makes sure they exist.
    """
    stem = file_path.stem
    root = PROCESSED_DIR / stem
    metadata_dir = root / "metadata"
    lineage_dir = root / "data_lineage"
    quality_dir = root / "quality_reports"

    for d in (metadata_dir, lineage_dir, quality_dir):
        d.mkdir(parents=True, exist_ok=True)

    return metadata_dir, lineage_dir, quality_dir

def next_seq(dir_path: Path, prefix: str, ts: str) -> int:
    """
    Find the next 3-digit sequence number for files that share
    a prefix and timestamp within dir_path.
    """
    pat = re.compile(rf"{re.escape(prefix)}_{ts}_(\d{{3}})\.(?:json|txt)$")
    nums = [int(m.group(1))
            for p in dir_path.iterdir()
            if (m := pat.match(p.name))]
    return (max(nums) + 1) if nums else 1

# ====
# COMPONENT 1: CRITICAL DATA FIELD IDENTIFIER
# ====

@dataclass
class FieldProfile:
    """Profile information for a data field"""
    name: str
    data_type: str
    is_critical: bool
    is_unique: bool
    expected_values: List[str]
    business_rules: List[str]
    completeness_threshold: float
    validity_patterns: List[str]

class CriticalDataElementIdentifier:
    """Identifies critical data fields and their characteristics"""
    
    def __init__(self):
        self.field_profiles = {}
        self.business_context_keywords = {
            'financial': ['price', 'cost', 'amount', 'total', 'spent', 'revenue'],
            'identity': ['id', 'identifier', 'key', 'number'],
            'temporal': ['date', 'time', 'timestamp', 'created', 'updated'],
            'quantity': ['quantity', 'count', 'amount', 'number'],
            'categorical': ['type', 'category', 'status', 'method', 'location']
        }
    
    def analyze_field_characteristics(self, df: pd.DataFrame) -> Dict[str, FieldProfile]:
        """Analyze each field to determine its characteristics and criticality"""
        profiles = {}
        
        for column in df.columns:
            # Basic analysis
            data_type = self._infer_data_type(df[column])
            is_critical = self._determine_criticality(column, df[column])
            is_unique = self._should_be_unique(column, df[column])
            expected_values = self._extract_expected_values(column, df[column])
            business_rules = self._generate_business_rules(column, df[column])
            completeness_threshold = self._set_completeness_threshold(column, is_critical)
            validity_patterns = self._extract_validity_patterns(column, df[column])
            
            profiles[column] = FieldProfile(
                name=column,
                data_type=data_type,
                is_critical=is_critical,
                is_unique=is_unique,
                expected_values=expected_values,
                business_rules=business_rules,
                completeness_threshold=completeness_threshold,
                validity_patterns=validity_patterns
            )
        
        self.field_profiles = profiles
        return profiles
    
    def _infer_data_type(self, series: pd.Series) -> str:
        """Infer the intended data type of a series"""
        # Clean the series first
        clean_series = series.dropna()
        clean_series = clean_series[~clean_series.astype(str).str.upper().isin(['ERROR', 'UNKNOWN', 'NAN'])]
        
        if len(clean_series) == 0:
            return 'unknown'
        
        # Check for dates
        if 'date' in series.name.lower() or 'time' in series.name.lower():
            return 'datetime'
        
        # Check for numeric
        try:
            pd.to_numeric(clean_series)
            if clean_series.astype(str).str.contains(r'\.').any():
                return 'float'
            else:
                return 'integer'
        except:
            pass
        
        # Check for categorical with limited unique values
        unique_ratio = len(clean_series.unique()) / len(clean_series)
        if unique_ratio < 0.1:  # Less than 10% unique values
            return 'categorical'
        
        return 'text'
    
    def _determine_criticality(self, column_name: str, series: pd.Series) -> bool:
        """Determine if a field is critical for business operations"""
        column_lower = column_name.lower()
        
        # Always critical
        critical_patterns = ['id', 'key', 'identifier', 'transaction']
        if any(pattern in column_lower for pattern in critical_patterns):
            return True
        
        # Business critical
        business_critical = ['item', 'product', 'service', 'amount', 'total', 'price']
        if any(pattern in column_lower for pattern in business_critical):
            return True
        
        # High completeness suggests criticality
        completeness = 1 - (series.isnull().sum() / len(series))
        if completeness > 0.95:
            return True
        
        return False
    
    def _should_be_unique(self, column_name: str, series: pd.Series) -> bool:
        """Determine if a field should contain unique values"""
        column_lower = column_name.lower()
        
        # Obvious unique fields
        unique_patterns = ['id', 'identifier', 'key', 'number']
        if any(pattern in column_lower for pattern in unique_patterns):
            return True
        
        # Check actual uniqueness
        clean_series = series.dropna()
        if len(clean_series) > 0:
            uniqueness_ratio = len(clean_series.unique()) / len(clean_series)
            return uniqueness_ratio > 0.95
        
        return False
    
    def _extract_expected_values(self, column_name: str, series: pd.Series) -> List[str]:
        """Extract expected values for categorical fields"""
        if self._infer_data_type(series) != 'categorical':
            return []
        
        # Clean the series
        clean_series = series.dropna()
        clean_series = clean_series[~clean_series.astype(str).str.upper().isin(['ERROR', 'UNKNOWN', 'NAN'])]
        
        # Get most common values
        value_counts = clean_series.value_counts()
        # Return values that appear more than once or represent >5% of data
        threshold = max(2, len(clean_series) * 0.05)
        expected = value_counts[value_counts >= threshold].index.tolist()
        
        return [str(val) for val in expected]
    
    def _generate_business_rules(self, column_name: str, series: pd.Series) -> List[str]:
        """Generate business rules for a field"""
        rules = []
        column_lower = column_name.lower()
        
        # Numeric rules
        if 'quantity' in column_lower:
            rules.append("Must be positive integer")
            rules.append("Reasonable range: 1-100")
        
        if any(word in column_lower for word in ['price', 'amount', 'total', 'cost']):
            rules.append("Must be positive number")
            rules.append("Reasonable range: 0.01-1000")
        
        # Date rules
        if 'date' in column_lower:
            rules.append("Must be valid date format")
            rules.append("Should not be in future")
            rules.append("Should be within last 5 years")
        
        # ID rules
        if any(word in column_lower for word in ['id', 'identifier']):
            rules.append("Must be unique")
            rules.append("Cannot be null")
        
        return rules
    
    def _set_completeness_threshold(self, column_name: str, is_critical: bool) -> float:
        """Set completeness threshold based on field importance"""
        if is_critical:
            return 0.95  # 95% completeness required
        else:
            return 0.80  # 80% completeness acceptable
    
    def _extract_validity_patterns(self, column_name: str, series: pd.Series) -> List[str]:
        """Extract validity patterns for the field"""
        patterns = []
        column_lower = column_name.lower()
        
        if 'date' in column_lower:
            patterns.append(r'\d{4}-\d{2}-\d{2}')
        
        if 'id' in column_lower:
            # Look for common ID patterns in the data
            clean_series = series.dropna().astype(str)
            if len(clean_series) > 0:
                sample = clean_series.iloc[0]
                if sample.startswith('TXN_'):
                    patterns.append(r'TXN_\d+')
        
        return patterns
    
    def get_critical_data_elements(self) -> List[str]:
        """Return list of critical field names"""
        return [name for name, profile in self.field_profiles.items() if profile.is_critical]
    
    def get_unique_fields(self) -> List[str]:
        """Return list of fields that should be unique"""
        return [name for name, profile in self.field_profiles.items() if profile.is_unique]

# ====
# COMPONENT 2: DATA QUALITY SCORER WITH AI ENHANCEMENT
# ====

class DataQualityScorer:
    """Measures data quality across multiple dimensions with AI enhancement capabilities"""
    
    def __init__(self):
        self.quality_dimensions = [
            'completeness', 'uniqueness', 'validity', 'consistency', 
            'accuracy', 'timeliness', 'business_rules'
        ]
        self.weights = {
            'completeness': 0.25,
            'uniqueness': 0.15,
            'validity': 0.20,
            'consistency': 0.15,
            'accuracy': 0.10,
            'timeliness': 0.10,
            'business_rules': 0.05
        }
    
    def calculate_completeness_score(self, df: pd.DataFrame, field_profiles: Dict[str, FieldProfile]) -> Dict[str, float]:
        """Calculate completeness score for each field"""
        scores = {}
        
        for column in df.columns:
            # Count various types of missing values
            null_count = df[column].isnull().sum()
            missing_values = df[column].astype(str).str.upper().isin(['NAN', 'UNKNOWN', 'ERROR']).sum()
            empty_strings = (df[column].astype(str).str.strip() == '').sum()
            
            total_missing = null_count + missing_values + empty_strings
            completeness = max(0, (len(df) - total_missing) / len(df))
            
            # Apply penalty for critical fields that don't meet threshold
            if column in field_profiles:
                threshold = field_profiles[column].completeness_threshold
                if completeness < threshold:
                    penalty = (threshold - completeness) * 0.5
                    completeness = max(0, completeness - penalty)
            
            scores[column] = completeness
        
        return scores
    
    def calculate_uniqueness_score(self, df: pd.DataFrame, unique_fields: List[str]) -> Dict[str, float]:
        """Calculate uniqueness score for fields that should be unique"""
        scores = {}
        
        for column in df.columns:
            if column in unique_fields:
                # For unique fields, calculate duplicate percentage
                clean_series = df[column].dropna()
                clean_series = clean_series[~clean_series.astype(str).str.upper().isin(['ERROR', 'UNKNOWN'])]
                
                if len(clean_series) > 0:
                    duplicates = clean_series.duplicated().sum()
                    uniqueness = max(0, (len(clean_series) - duplicates) / len(clean_series))
                else:
                    uniqueness = 0.0
            else:
                # For non-unique fields, this dimension doesn't apply
                uniqueness = 1.0
            
            scores[column] = uniqueness
        
        return scores
    
    def calculate_validity_score(self, df: pd.DataFrame, field_profiles: Dict[str, FieldProfile]) -> Dict[str, float]:
        """Calculate validity score based on data type consistency and format"""
        scores = {}
        
        for column in df.columns:
            valid_count = 0
            total_count = len(df)
            
            # Get expected data type
            expected_type = field_profiles.get(column, FieldProfile('', 'text', False, False, [], [], 0.8, [])).data_type
            
            for value in df[column].dropna():
                str_value = str(value).upper()
                if str_value in ['NAN', 'UNKNOWN', 'ERROR']:
                    continue
                
                is_valid = self._validate_value_type(value, expected_type)
                if is_valid:
                    valid_count += 1
            
            validity = valid_count / total_count if total_count > 0 else 0
            scores[column] = validity
        
        return scores
    
    def _validate_value_type(self, value, expected_type: str) -> bool:
        """Validate if a value matches the expected type"""
        try:
            if expected_type == 'integer':
                int(float(value))
                return True
            elif expected_type == 'float':
                float(value)
                return True
            elif expected_type == 'datetime':
                pd.to_datetime(value)
                return True
            elif expected_type in ['text', 'categorical']:
                return isinstance(value, str) or len(str(value)) > 0
            else:
                return True  # Default to valid for unknown types
        except:
            return False
    
    def calculate_consistency_score(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate consistency score based on standardization"""
        scores = {}
        
        for column in df.columns:
            if df[column].dtype == 'object':
                # Check for case inconsistencies and variations
                clean_values = df[column].dropna().astype(str)
                clean_values = clean_values[~clean_values.str.upper().isin(['ERROR', 'UNKNOWN', 'NAN'])]
                
                if len(clean_values) == 0:
                    consistency = 1.0
                else:
                    unique_values = clean_values.unique()
                    unique_upper = clean_values.str.upper().unique()
                    
                    # Calculate consistency based on case variations
                    consistency = len(unique_upper) / len(unique_values) if len(unique_values) > 0 else 1.0
                    consistency = min(1.0, consistency)
            else:
                consistency = 1.0
            
            scores[column] = consistency
        
        return scores
    
    def calculate_business_rules_score(self, df: pd.DataFrame, field_profiles: Dict[str, FieldProfile]) -> Dict[str, float]:
        """Calculate business rules compliance score"""
        scores = {}
        
        for column in df.columns:
            violations = 0
            total_records = len(df)
            
            # Get field profile
            profile = field_profiles.get(column)
            if not profile:
                scores[column] = 1.0
                continue
            
            # Apply business rules based on field type
            if 'quantity' in column.lower():
                # Quantity should be positive integer
                numeric_values = pd.to_numeric(df[column], errors='coerce')
                violations += (numeric_values <= 0).sum()
                violations += (numeric_values > 100).sum()  # Reasonable upper limit
            
            elif any(keyword in column.lower() for keyword in ['price', 'amount', 'total', 'spent']):
                # Financial values should be positive and reasonable
                numeric_values = pd.to_numeric(df[column], errors='coerce')
                violations += (numeric_values <= 0).sum()
                violations += (numeric_values > 1000).sum()  # Reasonable upper limit
            
            elif 'date' in column.lower():
                # Dates should be reasonable (not in future, not too old)
                try:
                    dates = pd.to_datetime(df[column], errors='coerce')
                    future_dates = dates > pd.Timestamp.now()
                    old_dates = dates < pd.Timestamp.now() - pd.Timedelta(days=5*365)  # 5 years ago
                    violations += future_dates.sum()
                    violations += old_dates.sum()
                except:
                    pass
            
            business_compliance = max(0, (total_records - violations) / total_records)
            scores[column] = business_compliance
        
        return scores
    
    def detect_contextual_anomalies(self, df: pd.DataFrame, column: str, field_profiles: Dict[str, FieldProfile]) -> List[str]:
        """Detect values that don't fit the expected context"""
        anomalies = []
        
        profile = field_profiles.get(column)
        if not profile or not profile.expected_values:
            return anomalies
        
        # Check for values not in expected list
        for value in df[column].dropna().unique():
            value_str = str(value).upper()
            if (value_str not in ['NAN', 'UNKNOWN', 'ERROR'] and
                str(value) not in profile.expected_values and
                not any(expected.lower() in str(value).lower() for expected in profile.expected_values)):
                anomalies.append(str(value))
        
        return anomalies
    
    def suggest_ai_corrections(self, anomalies: List[str], context: str) -> Dict[str, str]:
        """Suggest corrections for anomalous values (placeholder for AI integration)"""
        suggestions = {}
        
        # This is a simplified rule-based approach
        # In a real implementation, this would call OpenAI API
        for anomaly in anomalies:
            if context.lower() == 'item':
                # Cafe item suggestions
                cafe_items = ['Coffee', 'Tea', 'Cake', 'Cookie', 'Sandwich', 'Salad', 'Juice', 'Smoothie']
                # Simple fuzzy matching
                anomaly_lower = anomaly.lower()
                for item in cafe_items:
                    if anomaly_lower in item.lower() or item.lower() in anomaly_lower:
                        suggestions[anomaly] = item
                        break
                else:
                    suggestions[anomaly] = "REVIEW_REQUIRED"
            else:
                suggestions[anomaly] = "REVIEW_REQUIRED"
        
        return suggestions
    
    def calculate_overall_quality_score(self, df: pd.DataFrame, field_profiles: Dict[str, FieldProfile]) -> Dict[str, Any]:
        """Calculate overall data quality score"""
        
        critical_fields = [name for name, profile in field_profiles.items() if profile.is_critical]
        unique_fields = [name for name, profile in field_profiles.items() if profile.is_unique]
        
        # Calculate individual dimension scores
        completeness_scores = self.calculate_completeness_score(df, field_profiles)
        uniqueness_scores = self.calculate_uniqueness_score(df, unique_fields)
        validity_scores = self.calculate_validity_score(df, field_profiles)
        consistency_scores = self.calculate_consistency_score(df)
        business_rules_scores = self.calculate_business_rules_score(df, field_profiles)
        
        # Calculate weighted overall scores
        overall_scores = {}
        
        for column in df.columns:
            # Individual dimension scores
            dims = {
                'completeness': completeness_scores[column],
                'uniqueness': uniqueness_scores[column],
                'validity': validity_scores[column],
                'consistency': consistency_scores[column],
                'business_rules': business_rules_scores[column],
                'timeliness': 1.0,  # Simplified for this example
                'accuracy': 0.9  # Simplified baseline
            }
            
            # Calculate weighted score
            weighted_score = sum(dims[dim] * self.weights[dim] for dim in dims.keys())
            
            # Detect contextual anomalies
            anomalies = self.detect_contextual_anomalies(df, column, field_profiles)
            ai_suggestions = self.suggest_ai_corrections(anomalies, column) if anomalies else {}
            
            overall_scores[column] = {
                'overall_score': weighted_score * 100,  # Convert to percentage
                'dimension_scores': dims,
                'grade': self._get_quality_grade(weighted_score * 100),
                'contextual_anomalies': anomalies,
                'ai_suggestions': ai_suggestions
            }
        
        # Calculate dataset-level score
        dataset_score = np.mean([score['overall_score'] for score in overall_scores.values()])
        
        return {
            'dataset_score': dataset_score,
            'dataset_grade': self._get_quality_grade(dataset_score),
            'field_scores': overall_scores,
            'critical_fields_avg': np.mean([overall_scores[field]['overall_score'] 
                                          for field in critical_fields if field in overall_scores])
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'

# ====
# COMPONENT 3: METADATA GENERATOR
# ====

class MetadataGenerator:
    """Generates and manages metadata files for data quality assessment"""
    
    def __init__(self):
        self.metadata_template = {
            'dataset_info': {},
            'field_profiles': {},
            'quality_rules': {},
            'business_context': {},
            'ai_enhancement_config': {},
            'version': '1.0',
            'created_date': None,
            'last_updated': None
        }
    
    def create_metadata_file(self, df: pd.DataFrame, field_profiles: Dict[str, FieldProfile], 
                           quality_results: Dict[str, Any], output_dir: Path, 
                           dataset_name: str, timestamp: str) -> str:
        """Create comprehensive metadata file"""
        
        filename = output_dir / f"metadata_{timestamp}.json"
        
        metadata = self.metadata_template.copy()
        
        # Dataset information
        metadata['dataset_info'] = {
            'name': dataset_name,
            'description': f'{dataset_name} transaction data',
            'record_count': len(df),
            'field_count': len(df.columns),
            'data_types': {col: str(df[col].dtype) for col in df.columns},
            'overall_quality_score': quality_results['dataset_score'],
            'overall_quality_grade': quality_results['dataset_grade']
        }
        
        # Field profiles
        metadata['field_profiles'] = {
            name: asdict(profile) for name, profile in field_profiles.items()
        }
        
        # Quality rules
        metadata['quality_rules'] = {
            'completeness_thresholds': {
                name: profile.completeness_threshold 
                for name, profile in field_profiles.items()
            },
            'uniqueness_requirements': [
                name for name, profile in field_profiles.items() if profile.is_unique
            ],
            'critical_data_elements': [
                name for name, profile in field_profiles.items() if profile.is_critical
            ],
            'business_rules': {
                name: profile.business_rules 
                for name, profile in field_profiles.items() if profile.business_rules
            }
        }
        
        # Business context
        metadata['business_context'] = {
            'domain': 'retail_food_service',
            'primary_entities': ['transaction', 'item', 'customer_interaction'],
            'key_metrics': ['total_spent', 'quantity', 'transaction_frequency'],
            'data_lineage': {
                'source_system': 'pos_system',
                'extraction_method': 'batch_export',
                'transformation_applied': ['data_cleaning', 'quality_assessment']
            }
        }
        
        # AI enhancement configuration
        metadata['ai_enhancement_config'] = {
            'anomaly_detection_enabled': True,
            'auto_correction_threshold': 0.8,
            'manual_review_threshold': 0.6,
            'ai_model_suggestions': {
                'item_classification': 'enabled',
                'value_imputation': 'enabled',
                'pattern_recognition': 'enabled'
            }
        }
        
        # Timestamps
        metadata['created_date'] = datetime.now().isoformat()
        metadata['last_updated'] = datetime.now().isoformat()
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return str(filename)
    
    def load_metadata_file(self, filename: str) -> Dict[str, Any]:
        """Load metadata from file"""
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def update_metadata_file(self, filename: str, updates: Dict[str, Any]) -> None:
        """Update existing metadata file"""
        metadata = self.load_metadata_file(filename)
        metadata.update(updates)
        metadata['last_updated'] = datetime.now().isoformat()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)

# ====
# COMPONENT 4: DATA LINEAGE TRACKER
# ====

class DataLineageTracker:
    """Tracks data quality over time and maintains lineage information"""
    
    def __init__(self):
        self.lineage_log = []
    
    def log_data_quality_assessment(self, dataset_name: str, quality_results: Dict[str, Any], 
                                  metadata_file: str = None) -> Dict[str, Any]:
        """Log a data quality assessment"""
        
        lineage_entry = {
            'timestamp': datetime.now().isoformat(),
            'dataset_name': dataset_name,
            'overall_score': quality_results['dataset_score'],
            'overall_grade': quality_results['dataset_grade'],
            'critical_fields_score': quality_results.get('critical_fields_avg', 0),
            'field_scores': {
                field: scores['overall_score'] 
                for field, scores in quality_results['field_scores'].items()
            },
            'anomalies_detected': {
                field: len(scores['contextual_anomalies']) 
                for field, scores in quality_results['field_scores'].items()
                if scores['contextual_anomalies']
            },
            'metadata_file': metadata_file,
            'assessment_id': f"ASSESS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        self.lineage_log.append(lineage_entry)
        return lineage_entry
    
    def get_quality_trend(self, dataset_name: str, days: int = 30) -> Dict[str, Any]:
        """Get quality trend for a dataset over specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        relevant_entries = [
            entry for entry in self.lineage_log 
            if (entry['dataset_name'] == dataset_name and 
                datetime.fromisoformat(entry['timestamp']) > cutoff_date)
        ]
        
        if not relevant_entries:
            return {'message': 'No data available for the specified period'}
        
        scores = [entry['overall_score'] for entry in relevant_entries]
        
        return {
            'dataset_name': dataset_name,
            'period_days': days,
            'assessments_count': len(relevant_entries),
            'average_score': np.mean(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'trend': 'improving' if scores[-1] > scores[0] else 'declining' if scores[-1] < scores[0] else 'stable',
            'latest_assessment': relevant_entries[-1]
        }
    
    def save_lineage_log(self, output_dir: Path, timestamp: str, sequence: int) -> str:
        """Save lineage log to file"""
        filename = output_dir / f"data_lineage_{timestamp}_{sequence:03d}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.lineage_log, f, indent=2, default=str)
        
        return str(filename)

# ====
# MAIN DATA GOVERNANCE FRAMEWORK
# ====

class DataGovernanceFramework:
    """Main framework that orchestrates all components"""
    
    def __init__(self):
        self.identifier = CriticalDataElementIdentifier()
        self.scorer = DataQualityScorer()
        self.metadata_generator = MetadataGenerator()
        self.lineage_tracker = DataLineageTracker()
    
    def assess_data_quality(self, df: pd.DataFrame, dataset_name: str = "unknown_dataset") -> Dict[str, Any]:
        """Complete data quality assessment workflow"""
        
        print(f"🔍 Starting data quality assessment for: {dataset_name}")
        print(f"📊 Dataset shape: {df.shape}")
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Step 1: Identify critical data elements
        print("\n1️⃣ Identifying critical data elements...")
        field_profiles = self.identifier.analyze_field_characteristics(df)
        critical_fields = self.identifier.get_critical_data_elements()
        unique_fields = self.identifier.get_unique_fields()
        
        print(f"   ✅ Critical fields identified: {critical_fields}")
        print(f"   ✅ Unique fields identified: {unique_fields}")
        
        # Step 2: Calculate data quality scores
        print("\n2️⃣ Calculating data quality scores...")
        quality_results = self.scorer.calculate_overall_quality_score(df, field_profiles)
        
        print(f"   ✅ Overall dataset score: {quality_results['dataset_score']:.1f}% (Grade: {quality_results['dataset_grade']})")
        
        # Step 3: Create output directories
        print("\n3️⃣ Creating output directories...")
        # Use the original file path to determine the stem
        raw_file = Path(f"{dataset_name}.csv")  # Assuming CSV for now
        metadata_dir, lineage_dir, quality_dir = build_output_dirs(raw_file)
        print(f"   ✅ Output directories created under: {PROCESSED_DIR / raw_file.stem}")
        
        # Step 4: Generate metadata file
        print("\n4️⃣ Generating metadata file...")
        metadata_filename = self.metadata_generator.create_metadata_file(
            df, field_profiles, quality_results, metadata_dir, dataset_name, timestamp
        )
        print(f"   ✅ Metadata saved to: {metadata_filename}")
        
        # Step 5: Log to data lineage
        print("\n5️⃣ Logging to data lineage...")
        lineage_entry = self.lineage_tracker.log_data_quality_assessment(
            dataset_name, quality_results, metadata_filename
        )
        
        # Save lineage with sequence number
        lineage_seq = next_seq(lineage_dir, "data_lineage", timestamp)
        lineage_filename = self.lineage_tracker.save_lineage_log(lineage_dir, timestamp, lineage_seq)
        print(f"   ✅ Lineage entry created: {lineage_entry['assessment_id']}")
        print(f"   ✅ Lineage saved to: {lineage_filename}")
        
        return {
            'field_profiles': field_profiles,
            'quality_results': quality_results,
            'metadata_file': metadata_filename,
            'lineage_file': lineage_filename,
            'lineage_entry': lineage_entry,
            'output_dirs': {
                'metadata': metadata_dir,
                'lineage': lineage_dir,
                'quality': quality_dir
            }
        }
    
    def generate_quality_report(self, assessment_results: Dict[str, Any]) -> str:
        """Generate a comprehensive quality report"""
        
        quality_results = assessment_results['quality_results']
        field_profiles = assessment_results['field_profiles']
        quality_dir = assessment_results['output_dirs']['quality']
        
        report = []
        report.append("=" * 80)
        report.append("DATA QUALITY ASSESSMENT REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall Summary
        report.append("📊 OVERALL SUMMARY")
        report.append("-" * 40)
        report.append(f"Dataset Score: {quality_results['dataset_score']:.1f}% (Grade: {quality_results['dataset_grade']})")
        report.append(f"Critical Fields Average: {quality_results.get('critical_fields_avg', 0):.1f}%")
        report.append("")
        
        # Field-by-Field Analysis
        report.append("🔍 FIELD-BY-FIELD ANALYSIS")
        report.append("-" * 40)
        
        for field, scores in quality_results['field_scores'].items():
            profile = field_profiles.get(field)
            report.append(f"\n📋 {field}")
            report.append(f"   Overall Score: {scores['overall_score']:.1f}% (Grade: {scores['grade']})")
            report.append(f"   Critical Field: {'Yes' if profile and profile.is_critical else 'No'}")
            report.append(f"   Should be Unique: {'Yes' if profile and profile.is_unique else 'No'}")
            
            # Dimension scores
            dims = scores['dimension_scores']
            report.append(f"   Completeness: {dims['completeness']:.2f}")
            report.append(f"   Validity: {dims['validity']:.2f}")
            report.append(f"   Consistency: {dims['consistency']:.2f}")
            report.append(f"   Business Rules: {dims['business_rules']:.2f}")
            
            # Anomalies
            if scores['contextual_anomalies']:
                report.append(f"   ⚠️  Anomalies Found: {scores['contextual_anomalies']}")
                if scores['ai_suggestions']:
                    report.append(f"   🤖 AI Suggestions: {scores['ai_suggestions']}")
        
        # Recommendations
        report.append("\n💡 RECOMMENDATIONS")
        report.append("-" * 40)
        
        low_quality_fields = [
            field for field, scores in quality_results['field_scores'].items()
            if scores['overall_score'] < 70
        ]
        
        if low_quality_fields:
            report.append("🔴 Fields requiring immediate attention:")
            for field in low_quality_fields:
                score = quality_results['field_scores'][field]['overall_score']
                report.append(f"   - {field}: {score:.1f}%")
        
        critical_issues = []
        for field, scores in quality_results['field_scores'].items():
            profile = field_profiles.get(field)
            if profile and profile.is_critical and scores['overall_score'] < 80:
                critical_issues.append(field)
        
        if critical_issues:
            report.append("\n🚨 Critical fields with quality issues:")
            for field in critical_issues:
                report.append(f"   - {field}")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        # Save report to file with sequence number
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_seq = next_seq(quality_dir, "quality_report", timestamp)
        report_filename = quality_dir / f"quality_report_{timestamp}_{report_seq:03d}.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"📄 Quality report saved to: {report_filename}")
        return report_text

def process_data_file(file_path: Path) -> None:
    """Process a single data file through the complete workflow"""
    
    print(f"\n{'='*80}")
    print(f"PROCESSING FILE: {file_path.name}")
    print(f"{'='*80}")
    
    # Load the data
    try:
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            print(f"❌ Unsupported file format: {file_path.suffix}")
            return
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Initialize the framework
    framework = DataGovernanceFramework()
    
    # Run complete assessment
    dataset_name = file_path.stem
    assessment_results = framework.assess_data_quality(df, dataset_name)
    
    # Generate and display report
    print(f"\n{'='*80}")
    print("GENERATING COMPREHENSIVE QUALITY REPORT")
    print(f"{'='*80}")
    
    report = framework.generate_quality_report(assessment_results)
    print(report)
    
    print(f"\n✅ Data Governance Framework Assessment Complete for {file_path.name}!")
    print("\nFiles created:")
    print(f"  - Metadata: {assessment_results['metadata_file']}")
    print(f"  - Data Lineage: {assessment_results['lineage_file']}")
    print(f"  - Quality Report: Located in {assessment_results['output_dirs']['quality']}")

# ====
# MAIN EXECUTION
# ====

if __name__ == "__main__":
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)
    
    print("🚀 Data Governance Framework Starting...")
    print(f"📁 Data directory: {DATA_DIR.resolve()}")
    print(f"📁 Processed directory: {PROCESSED_DIR.resolve()}")
    
    # Process all files in the data directory
    data_files = list(DATA_DIR.glob("*.*"))
    
    if not data_files:
        print("❌ No data files found in the data directory!")
        print("Please place your data files (.csv, .xlsx, .xls) in the 'data' folder.")
    else:
        print(f"\n📊 Found {len(data_files)} file(s) to process:")
        for file_path in data_files:
            print(f"  - {file_path.name}")
        
        # Process each file
        for file_path in data_files:
            try:
                process_data_file(file_path)
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
                continue
    
    print("\n🎉 All processing complete!")
