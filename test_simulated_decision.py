#!/usr/bin/env python3
"""
Test script for simulated_decision.py
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print('in here')
    from SunSight.Models.Incentives.simulated_decision import SolarAdoptionModelZipCode
    
    # Test with a sample zipcode
    print("Testing SolarAdoptionModelZipCode...")
    
    # Use a zipcode that might exist in your data
    test_zipcode = 10001  # New York zipcode
    
    model = SolarAdoptionModelZipCode(test_zipcode)
    print(f"Created model for zipcode: {test_zipcode}")
    
    # Try to generate agents
    try:
        model.generate_agents()
        print(f"Generated {len(model.agents)} agents")
        
        # Test getting discount rates
        model.get_all_needed_discount_rates()
        print("Successfully calculated needed discount rates")
        
        # Test stepping the model
        model.step()
        print("Successfully stepped the model")
        
    except Exception as e:
        print(f"Error during agent generation: {e}")
        print("This might be due to missing data files or incorrect file paths")
    
    print("Test completed!")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
except Exception as e:
    print(f"Unexpected error: {e}") 