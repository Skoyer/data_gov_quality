# 🚀 Data Governance Framework

A comprehensive, production-ready data quality assessment framework with AI-enhancement capabilities. This framework automatically identifies critical data elements, scores data quality across multiple dimensions, and provides actionable insights for data governance.

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Testing](#-testing)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Examples](#-examples)
- [Contributing](#-contributing)

## ✨ Features

### 🔍 Automated Data Quality Assessment
- **Multi-dimensional scoring**: Completeness, uniqueness, validity, consistency, business rules compliance
- **Critical Data Element (CDE) identification**: Automatically identifies business-critical fields
- **Contextual anomaly detection**: Finds values that don't fit expected patterns
- **Weighted scoring system**: Configurable weights for different quality dimensions

### 🤖 AI-Ready Architecture
- **Placeholder for OpenAI integration**: Ready for AI-powered suggestions and corrections
- **Pattern recognition**: Identifies data patterns and anomalies
- **Smart field profiling**: Infers data types and business rules automatically

### 📊 Comprehensive Reporting
- **Detailed quality reports**: Field-by-field analysis with actionable recommendations
- **Metadata generation**: Complete field profiles and business context
- **Data lineage tracking**: Historical quality trends and assessment tracking
- **Multiple output formats**: JSON metadata, text reports, structured logs

### 🏗️ Production-Ready Features
- **Error handling**: Graceful handling of data quality issues and edge cases
- **Configurable thresholds**: Customizable quality standards per field type
- **Modular architecture**: Easy to extend and customize
- **Command-line interface**: Batch processing and automation support

## 🚀 Quick Start

### 1. Setup
```bash
# Clone or download the framework files
# Run the setup script
python setup.py
```

### 2. Basic Test
```bash
# Test basic functionality
python run_tests.py --basic
```

### 3. Process Your Data
```bash
# Process all files in data/ directory
python main.py

# Process a specific file
python main.py --file your_data.csv
```

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.20.0

### Automated Setup
```bash
python setup.py
```

### Manual Setup
```bash
# Install dependencies
pip install pandas numpy

# Create required directories
mkdir -p data processed archive test_data logs

# Run tests to validate installation
python run_tests.py --all
```

## 📖 Usage

### Command Line Interface

```bash
# Basic usage - process all files in data/ directory
python main.py

# Process specific file
python main.py --file path/to/your/data.csv

# Override data directory
python main.py --data-dir /path/to/your/data

# Run basic functionality test
python main.py --test
```

### Programmatic Usage

```python
from main import DataGovernanceFramework
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize framework
framework = DataGovernanceFramework()

# Run complete assessment
results = framework.assess_data_quality(df, "your_dataset_name")

# Generate comprehensive report
report = framework.generate_quality_report(results)
print(report)

# Access individual components
field_profiles = results['field_profiles']
quality_scores = results['quality_results']
metadata_file = results['metadata_file']
```

## 🏗️ Architecture

### Core Components

#### 1. Critical Data Element Identifier
Automatically identifies business-critical fields based on:
- Field naming patterns (ID, key, identifier, etc.)
- Data completeness levels
- Business context keywords
- Uniqueness requirements

#### 2. Data Quality Scorer
Multi-dimensional quality assessment:
- **Completeness** (25%): Missing value detection and penalties
- **Validity** (20%): Data type consistency and format validation
- **Uniqueness** (15%): Duplicate detection for key fields
- **Consistency** (15%): Standardization and case consistency
- **Business Rules** (5%): Domain-specific validation rules
- **Accuracy** (10%): Baseline accuracy assessment
- **Timeliness** (10%): Date range and recency validation

#### 3. Metadata Generator
Creates comprehensive metadata including:
- Dataset information and statistics
- Field profiles with business rules
- Quality thresholds and requirements
- AI enhancement configuration
- Data lineage information

#### 4. Data Lineage Tracker
Maintains historical records:
- Quality assessment history
- Trend analysis over time
- Assessment comparison
- Change tracking

### Data Flow

```
Raw Data → Field Analysis → Quality Scoring → Report Generation
    ↓           ↓              ↓               ↓
Metadata ← Lineage Log ← Quality Results ← Recommendations
```

## 🧪 Testing

### Test Suite Overview
The framework includes comprehensive tests covering:

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: End-to-end workflow validation
3. **Error Handling Tests**: Edge case and error condition testing
4. **Data Quality Tests**: Various data quality scenarios

### Running Tests

```bash
# Run basic functionality test
python run_tests.py --basic

# Run comprehensive test suite
python run_tests.py --comprehensive

# Run all tests
python run_tests.py --all

# Run specific test file
python test_framework.py
```

### Test Coverage
- ✅ Critical Data Element Identification
- ✅ Data Quality Scoring (all dimensions)
- ✅ Metadata Generation and Management
- ✅ Data Lineage Tracking
- ✅ Error Handling and Edge Cases
- ✅ File I/O Operations
- ✅ Directory Management
- ✅ Full Framework Integration

## ⚙️ Configuration

### Quality Dimension Weights
Customize scoring weights in `DataQualityScorer`:

```python
self.weights = {
    'completeness': 0.25,    # 25% weight
    'uniqueness': 0.15,      # 15% weight
    'validity': 0.20,        # 20% weight
    'consistency': 0.15,     # 15% weight
    'accuracy': 0.10,        # 10% weight
    'timeliness': 0.10,      # 10% weight
    'business_rules': 0.05   # 5% weight
}
```

### Completeness Thresholds
- **Critical fields**: 95% completeness required
- **Non-critical fields**: 80% completeness acceptable

### Business Rules
Automatically applied based on field names:
- **Quantity fields**: Must be positive integers (1-100 range)
- **Financial fields**: Must be positive numbers (0.01-1000 range)
- **Date fields**: Must be valid dates, not in future, within 5 years
- **ID fields**: Must be unique and non-null

## 📚 API Reference

### DataGovernanceFramework

Main orchestration class that coordinates all components.

#### Methods

**`assess_data_quality(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]`**
- Runs complete data quality assessment
- Returns comprehensive results including profiles, scores, and file paths

**`generate_quality_report(assessment_results: Dict[str, Any]) -> str`**
- Generates human-readable quality report
- Saves report to file and returns content

### CriticalDataElementIdentifier

Identifies critical data fields and their characteristics.

#### Methods

**`analyze_field_characteristics(df: pd.DataFrame) -> Dict[str, FieldProfile]`**
- Analyzes each field to determine characteristics
- Returns field profiles with business rules and thresholds

**`get_critical_data_elements() -> List[str]`**
- Returns list of critical field names

**`get_unique_fields() -> List[str]`**
- Returns list of fields that should be unique

### DataQualityScorer

Measures data quality across multiple dimensions.

#### Methods

**`calculate_overall_quality_score(df: pd.DataFrame, field_profiles: Dict[str, FieldProfile]) -> Dict[str, Any]`**
- Calculates comprehensive quality scores
- Returns dataset-level and field-level scores with grades

**`detect_contextual_anomalies(df: pd.DataFrame, column: str, field_profiles: Dict[str, FieldProfile]) -> List[str]`**
- Detects values that don't fit expected context
- Returns list of anomalous values

## 📊 Examples

### Sample Output

```
🔍 Starting data quality assessment for: sales_data
📊 Dataset shape: (10000, 8)

1️⃣ Identifying critical data elements...
   ✅ Critical fields identified: ['Transaction_ID', 'Item', 'Quantity', 'Total_Spent']
   ✅ Unique fields identified: ['Transaction_ID']

2️⃣ Calculating data quality scores...
   ✅ Overall dataset score: 87.3% (Grade: B)

3️⃣ Creating output directories...
   ✅ Output directories created under: processed/sales_data

4️⃣ Generating metadata file...
   ✅ Metadata saved to: processed/sales_data/metadata/metadata_20240101_120000.json

5️⃣ Logging to data lineage...
   ✅ Lineage entry created: ASSESS_20240101_120000
```

### Quality Report Sample

```
================================================================================
DATA QUALITY ASSESSMENT REPORT
================================================================================
Generated: 2024-01-01 12:00:00

📊 OVERALL SUMMARY
----------------------------------------
Dataset Score: 87.3% (Grade: B)
Critical Fields Average: 92.1%

🔍 FIELD-BY-FIELD ANALYSIS
----------------------------------------

📋 Transaction_ID
   Overall Score: 99.0% (Grade: A)
   Critical Field: Yes
   Should be Unique: Yes
   Completeness: 1.00
   Validity: 0.99
   Consistency: 1.00
   Business Rules: 0.98

💡 RECOMMENDATIONS
----------------------------------------
🔴 Fields requiring immediate attention:
   - Location: 66.5%
   - Payment_Method: 73.6%
```

### Directory Structure

After processing, the framework creates:

```
processed/
└── your_dataset/
    ├── metadata/
    │   └── metadata_20240101_120000.json
    ├── data_lineage/
    │   └── data_lineage_20240101_120000_001.json
    └── quality_reports/
        └── quality_report_20240101_120000_001.txt
```

## 🔮 AI Integration (Future Enhancement)

The framework is designed for easy AI integration:

```python
def suggest_ai_corrections(self, anomalies: List[str], context: str) -> Dict[str, str]:
    """
    Ready for OpenAI API integration:

    import openai

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{
            "role": "user", 
            "content": f"Suggest corrections for these {context} values: {anomalies}"
        }]
    )

    return parse_ai_suggestions(response)
    """
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`python run_tests.py --all`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd data-governance-framework

# Run setup
python setup.py

# Run tests to ensure everything works
python run_tests.py --all
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**ImportError: No module named 'pandas'**
```bash
pip install pandas numpy
```

**Permission denied when creating directories**
```bash
# Ensure you have write permissions in the current directory
chmod 755 .
```

**Tests failing with "File not found"**
```bash
# Ensure all required files are present
python setup.py
```

**Memory issues with large datasets**
- Process data in chunks
- Use `--data-dir` to process files individually
- Monitor memory usage during processing

### Getting Help

1. Check the test output: `python run_tests.py --all`
2. Validate your data format (CSV/Excel with headers)
3. Ensure required directories exist
4. Check Python version compatibility (3.7+)

## 🎯 Roadmap

- [ ] OpenAI API integration for smart corrections
- [ ] Real-time data quality monitoring
- [ ] Web dashboard for quality metrics
- [ ] Integration with popular data platforms
- [ ] Advanced statistical anomaly detection
- [ ] Custom business rule engine
- [ ] Data quality SLA monitoring
- [ ] Automated data remediation workflows

---

**Built with ❤️ for better data governance**
