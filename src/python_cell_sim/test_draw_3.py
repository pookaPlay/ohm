#!/usr/bin/env python3
"""Test script to verify OHM1_ARRAY.Draw works with 3 stacked objects"""

import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from OHM1_ARRAY import OHM1_ARRAY

# Test with 3 OHM objects
print("Testing OHM1_ARRAY.Draw with 3 objects...")
K = 4
state0 = [5, 7, 6]  # 3 values for 3 OHM objects
N = len(state0)

print(f"State0({N}): {state0}")
ohm = OHM1_ARRAY(N, K, ptf="max", debugDone=0)

ohm.InitState(state0, initMode=1)

# Test the Draw method
print("Drawing...")
try:
    ohm.Draw("Test State")
    print("✓ Draw method executed successfully!")
    
    # Save the figure to verify it looks correct
    if ohm.fig is not None:
        ohm.fig.savefig('/tmp/test_ohm_draw.png', dpi=100, bbox_inches='tight')
        print("✓ Figure saved to /tmp/test_ohm_draw.png")
    
except Exception as e:
    print(f"✗ Error during Draw: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTest completed successfully!")
plt.close('all')
