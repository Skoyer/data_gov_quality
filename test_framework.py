import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import json
import sys
import os

# Add the current directory to path to import our modules
sys.path.insert(0, '.')

# Import our framework components
from main import (
    DataGovernanceFramework,
    CriticalDataElementIdentifier,
    DataQualityScorer,
    MetadataGenerator,
    DataLineageTracker,
    FieldProfile,
    create_test_data,
    build_output_dirs,
    next_seq
)

class TestDataGovernanceFramework(unittest.TestCase):
    """Comprehensive test suite for the Data Governance Framework"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_dir = self.test_dir / "data"
        self.processed_dir = self.test_dir / "processed"

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Create test data
        self.test_df = create_test_data()

        # Initialize framework components
        self.framework = DataGovernanceFramework()
        self.identifier = CriticalDataElementIdentifier()
        self.scorer = DataQualityScorer()
        self.metadata_generator = MetadataGenerator()
        self.lineage_tracker = DataLineageTracker()

    def tearDown(self):
        """Clean up after each test method."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_create_test_data(self):
        """Test that test data is created correctly"""
        df = create_test_data()

        # Check basic properties
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1000)
        self.assertEqual(len(df.columns), 8)

        # Check expected columns
        expected_columns = [
            'Transaction_ID', 'Item', 'Quantity', 'Price_Per_Unit',
            'Total_Spent', 'Transaction_Date', 'Payment_Method', 'Location'
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)

        # Check data types
        self.assertTrue(df['Transaction_ID'].dtype == 'object')
        self.assertTrue(df['Quantity'].dtype in ['int64', 'object'])
        self.assertTrue(df['Price_Per_Unit'].dtype in ['float64', 'object'])

        print("[CHECK] Test data creation test passed")

    def test_critical_data_element_identifier(self):
        """Test the Critical Data Element Identifier component"""

        # Analyze field characteristics
        field_profiles = self.identifier.analyze_field_characteristics(self.test_df)

        # Check that profiles were created for all fields
        self.assertEqual(len(field_profiles), len(self.test_df.columns))

        # Check that Transaction_ID is identified as critical and unique
        txn_id_profile = field_profiles.get('Transaction_ID')
        self.assertIsNotNone(txn_id_profile)
        self.assertTrue(txn_id_profile.is_critical)
        self.assertTrue(txn_id_profile.is_unique)

        # Check that financial fields are identified as critical
        financial_fields = ['Price_Per_Unit', 'Total_Spent']
        for field in financial_fields:
            if field in field_profiles:
                self.assertTrue(field_profiles[field].is_critical)

        # Test helper methods
        critical_fields = self.identifier.get_critical_data_elements()
        unique_fields = self.identifier.get_unique_fields()

        self.assertIsInstance(critical_fields, list)
        self.assertIsInstance(unique_fields, list)
        self.assertIn('Transaction_ID', unique_fields)

        print("[CHECK] Critical Data Element Identifier test passed")

    def test_data_quality_scorer(self):
        """Test the Data Quality Scorer component"""

        # First get field profiles
        field_profiles = self.identifier.analyze_field_characteristics(self.test_df)

        # Test individual scoring methods
        completeness_scores = self.scorer.calculate_completeness_score(self.test_df, field_profiles)
        self.assertEqual(len(completeness_scores), len(self.test_df.columns))

        # All scores should be between 0 and 1
        for score in completeness_scores.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

        # Test uniqueness scoring
        unique_fields = self.identifier.get_unique_fields()
        uniqueness_scores = self.scorer.calculate_uniqueness_score(self.test_df, unique_fields)
        self.assertEqual(len(uniqueness_scores), len(self.test_df.columns))

        # Test validity scoring
        validity_scores = self.scorer.calculate_validity_score(self.test_df, field_profiles)
        self.assertEqual(len(validity_scores), len(self.test_df.columns))

        # Test consistency scoring
        consistency_scores = self.scorer.calculate_consistency_score(self.test_df)
        self.assertEqual(len(consistency_scores), len(self.test_df.columns))

        # Test business rules scoring
        business_scores = self.scorer.calculate_business_rules_score(self.test_df, field_profiles)
        self.assertEqual(len(business_scores), len(self.test_df.columns))

        # Test overall quality score calculation
        overall_results = self.scorer.calculate_overall_quality_score(self.test_df, field_profiles)

        self.assertIn('dataset_score', overall_results)
        self.assertIn('dataset_grade', overall_results)
        self.assertIn('field_scores', overall_results)

        # Dataset score should be between 0 and 100
        self.assertGreaterEqual(overall_results['dataset_score'], 0)
        self.assertLessEqual(overall_results['dataset_score'], 100)

        # Grade should be A, B, C, D, or F
        self.assertIn(overall_results['dataset_grade'], ['A', 'B', 'C', 'D', 'F'])

        print("[CHECK] Data Quality Scorer test passed")

    def test_metadata_generator(self):
        """Test the Metadata Generator component"""

        # Get field profiles and quality results
        field_profiles = self.identifier.analyze_field_characteristics(self.test_df)
        quality_results = self.scorer.calculate_overall_quality_score(self.test_df, field_profiles)

        # Create metadata file
        output_dir = self.test_dir / "metadata"
        output_dir.mkdir(exist_ok=True)

        timestamp = "20240101_120000"
        metadata_file = self.metadata_generator.create_metadata_file(
            self.test_df, field_profiles, quality_results, 
            output_dir, "test_dataset", timestamp
        )

        # Check that file was created
        self.assertTrue(Path(metadata_file).exists())

        # Load and validate metadata content
        metadata = self.metadata_generator.load_metadata_file(metadata_file)

        # Check required sections
        required_sections = [
            'dataset_info', 'field_profiles', 'quality_rules',
            'business_context', 'ai_enhancement_config', 'version'
        ]
        for section in required_sections:
            self.assertIn(section, metadata)

        # Check dataset info
        dataset_info = metadata['dataset_info']
        self.assertEqual(dataset_info['record_count'], len(self.test_df))
        self.assertEqual(dataset_info['field_count'], len(self.test_df.columns))

        print("[CHECK] Metadata Generator test passed")

    def test_data_lineage_tracker(self):
        """Test the Data Lineage Tracker component"""

        # Get quality results
        field_profiles = self.identifier.analyze_field_characteristics(self.test_df)
        quality_results = self.scorer.calculate_overall_quality_score(self.test_df, field_profiles)

        # Log assessment
        lineage_entry = self.lineage_tracker.log_data_quality_assessment(
            "test_dataset", quality_results, "test_metadata.json"
        )

        # Check lineage entry structure
        required_fields = [
            'timestamp', 'dataset_name', 'overall_score', 'overall_grade',
            'field_scores', 'assessment_id'
        ]
        for field in required_fields:
            self.assertIn(field, lineage_entry)

        # Check that log was updated
        self.assertEqual(len(self.lineage_tracker.lineage_log), 1)

        # Test saving lineage log
        output_dir = self.test_dir / "lineage"
        output_dir.mkdir(exist_ok=True)

        lineage_file = self.lineage_tracker.save_lineage_log(
            output_dir, "20240101_120000", 1
        )

        self.assertTrue(Path(lineage_file).exists())

        print("[CHECK] Data Lineage Tracker test passed")

    def test_build_output_dirs(self):
        """Test output directory creation"""

        # Test with a sample file path
        file_path = Path("test_data.csv")

        # Mock the PROCESSED_DIR for this test
        import main
        original_processed_dir = main.PROCESSED_DIR
        main.PROCESSED_DIR = self.processed_dir

        try:
            metadata_dir, lineage_dir, quality_dir = build_output_dirs(file_path)

            # Check that directories were created
            self.assertTrue(metadata_dir.exists())
            self.assertTrue(lineage_dir.exists())
            self.assertTrue(quality_dir.exists())

            # Check directory structure
            expected_base = self.processed_dir / "test_data"
            self.assertEqual(metadata_dir, expected_base / "metadata")
            self.assertEqual(lineage_dir, expected_base / "data_lineage")
            self.assertEqual(quality_dir, expected_base / "quality_reports")

        finally:
            # Restore original PROCESSED_DIR
            main.PROCESSED_DIR = original_processed_dir

        print("[CHECK] Output directory creation test passed")

    def test_next_seq(self):
        """Test sequence number generation"""

        # Create test directory with some files
        test_dir = self.test_dir / "seq_test"
        test_dir.mkdir(exist_ok=True)

        # Create some test files
        timestamp = "20240101_120000"
        prefix = "test_file"

        # Create files with sequence numbers
        (test_dir / f"{prefix}_{timestamp}_001.json").touch()
        (test_dir / f"{prefix}_{timestamp}_002.json").touch()
        (test_dir / f"{prefix}_{timestamp}_005.json").touch()

        # Test next sequence number
        next_num = next_seq(test_dir, prefix, timestamp)
        self.assertEqual(next_num, 6)  # Should be max(1,2,5) + 1 = 6

        # Test with no existing files
        empty_dir = self.test_dir / "empty_seq_test"
        empty_dir.mkdir(exist_ok=True)
        next_num_empty = next_seq(empty_dir, prefix, timestamp)
        self.assertEqual(next_num_empty, 1)  # Should be 1 for empty directory

        print("[CHECK] Sequence number generation test passed")

    def test_full_framework_integration(self):
        """Test the complete framework integration"""

        # Run complete assessment
        assessment_results = self.framework.assess_data_quality(self.test_df, "integration_test")

        # Check that all expected results are present
        required_keys = [
            'field_profiles', 'quality_results', 'metadata_file',
            'lineage_file', 'lineage_entry', 'output_dirs'
        ]
        for key in required_keys:
            self.assertIn(key, assessment_results)

        # Check that files were created
        self.assertTrue(Path(assessment_results['metadata_file']).exists())
        self.assertTrue(Path(assessment_results['lineage_file']).exists())

        # Generate quality report
        report = self.framework.generate_quality_report(assessment_results)
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 100)  # Report should have substantial content

        # Check that report contains expected sections
        self.assertIn("DATA QUALITY ASSESSMENT REPORT", report)
        self.assertIn("OVERALL SUMMARY", report)
        self.assertIn("FIELD-BY-FIELD ANALYSIS", report)

        print("[CHECK] Full framework integration test passed")

    def test_error_handling(self):
        """Test error handling with problematic data"""

        # Create DataFrame with various data quality issues
        problematic_data = {
            'id': [1, 2, 2, None, 'invalid'],  # Duplicates and invalid values
            'amount': [10.5, -5, 'not_a_number', None, 9999],  # Negative and invalid values
            'date': ['2023-01-01', '2025-12-31', 'invalid_date', None, '1900-01-01'],  # Future and invalid dates
            'category': ['A', 'B', '', None, 'UNKNOWN']  # Missing and unknown values
        }

        problematic_df = pd.DataFrame(problematic_data)

        # Framework should handle this without crashing
        try:
            assessment_results = self.framework.assess_data_quality(problematic_df, "problematic_test")

            # Should still produce results
            self.assertIn('quality_results', assessment_results)
            self.assertIn('field_profiles', assessment_results)

            # Quality scores should reflect the issues
            quality_results = assessment_results['quality_results']
            self.assertLess(quality_results['dataset_score'], 90)  # Should have lower score due to issues

            print("[CHECK] Error handling test passed")

        except Exception as e:
            self.fail(f"Framework should handle problematic data gracefully, but raised: {e}")

    def test_field_profile_dataclass(self):
        """Test the FieldProfile dataclass"""

        profile = FieldProfile(
            name="test_field",
            data_type="integer",
            is_critical=True,
            is_unique=False,
            expected_values=["1", "2", "3"],
            business_rules=["Must be positive"],
            completeness_threshold=0.95,
            validity_patterns=[r"\d+"]
        )

        # Test basic properties
        self.assertEqual(profile.name, "test_field")
        self.assertEqual(profile.data_type, "integer")
        self.assertTrue(profile.is_critical)
        self.assertFalse(profile.is_unique)

        # Test that it can be converted to dict (for JSON serialization)
        profile_dict = profile.__dict__
        self.assertIn('name', profile_dict)
        self.assertIn('data_type', profile_dict)

        print("[CHECK] FieldProfile dataclass test passed")

def run_all_tests():
    """Run all tests and return results"""

    print("[TEST] Starting Comprehensive Test Suite")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDataGovernanceFramework)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("[TARGET] TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print("\n[ERROR] FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\n[CRITICAL] ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0

    if success:
        print("\n[PARTY] ALL TESTS PASSED!")
    else:
        print("\n[ERROR] SOME TESTS FAILED!")

    return success

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
