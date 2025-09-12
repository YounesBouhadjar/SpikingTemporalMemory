#!/usr/bin/env python3
"""
Simple test script focused on model functionality from training.py
"""

import unittest
import time
import numpy as np

import nest
from spikingtemporalmemory import default_params as dp
from spikingtemporalmemory import model


class TestPipeline(unittest.TestCase):
    def test_pipeline_with_real_nest(self):
        # Copy params so we don’t mutate defaults
        params = dp.p.copy()
        neuron_model = params["soma_model"]
        synapse_model = params["syn_dict_ee_synapse_model"]

        # -------------------------------------------------------
        # Install compiled nestml modules
        # -------------------------------------------------------
        try:
            nest.Install(f"module/nestml_{neuron_model}_module")
            nest.Install(f"module/nestml_{neuron_model}_{synapse_model}_module")
        except Exception as e:
            self.fail(f"NEST module installation failed: {e}")

        # Update params with NESTML model names
        params["soma_model"] = f"{neuron_model}_nestml__with_{synapse_model}_nestml"
        params["syn_dict_ee"]["synapse_model"] = f"{synapse_model}_nestml__with_{neuron_model}_nestml"

        params["label"] = "test_run"

        # -------------------------------------------------------
        # Minimal pipeline (skip sequence generation)
        # -------------------------------------------------------
        params["M"] = 5  # fake vocabulary size just to initialize
        vocab = ["a", "b", "c"]

        model_instance = model.Model(params, vocab)

        # Model creation
        model_instance.create()
        # Connect network
        model_instance.connect()
        # Set dummy input (normally from sg)
        fake_element_activations = np.array([[140., 150., 160.], [1, 2, 3]])
        fake_seq_set_instance = {0: {"times": [110., 120., 130.], "elements": [1, 2, 3]}}
        model_instance.set_input(fake_element_activations, fake_seq_set_instance)

        # -------------------------------------------------------
        # Assertions
        # -------------------------------------------------------
        self.assertIn("soma_model", params)
        self.assertIn("synapse_model", params["syn_dict_ee"])
        self.assertTrue(params["M"] > 0)


if __name__ == "__main__":
    unittest.main()
