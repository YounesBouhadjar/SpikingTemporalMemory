#!/usr/bin/env python3
"""
Simple test script for PyNEST SpikingTemporalMemory
"""

import unittest
import sys
import os

# Add the examples directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'sequence_learning_and_prediction'))

class TestSimple(unittest.TestCase):
    """Simple test cases"""
    
    def test_import_training(self):
        """Test that we can import the training module"""
        try:
            import training
            self.assertTrue(True, "Training module imported successfully")
        except ImportError as e:
            self.fail(f"Could not import training module: {e}")
    
    def test_import_parameters(self):
        """Test that we can import the parameters_space module"""
        try:
            import parameters_space
            self.assertTrue(True, "Parameters space module imported successfully")
        except ImportError as e:
            self.fail(f"Could not import parameters_space module: {e}")
    
    def test_parameters_structure(self):
        """Test basic parameters structure"""
        try:
            import parameters_space
            p = parameters_space.p
            
            # Check some basic parameters exist
            self.assertIn('M', p, "Parameter 'M' should exist")
            self.assertIn('n_E', p, "Parameter 'n_E' should exist")
            self.assertIn('task', p, "Parameter 'task' should exist")
            
            # Check task parameters
            self.assertIn('vocabulary_size', p['task'], "Task should have vocabulary_size")
            self.assertIn('S', p['task'], "Task should have S parameter")
            
        except ImportError:
            self.skipTest("Parameters space module not available")
    
    def test_training_parser(self):
        """Test that the training parser can be created"""
        try:
            import training
            parser = training.create_parser()
            self.assertIsNotNone(parser, "Parser should be created successfully")
        except ImportError:
            self.skipTest("Training module not available")

if __name__ == '__main__':
    unittest.main(verbosity=2)
