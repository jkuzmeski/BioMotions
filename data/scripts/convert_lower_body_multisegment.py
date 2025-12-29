"""
Convert lower_body_multisegment MJCF to USD with the correct structure for ProtoMotions.

This script creates a USD file that matches the expected format used by other robots
in the codebase (flat body structure with single articulation root on Pelvis).

Run this from the project root:
    python data/scripts/convert_lower_body_multisegment.py
"""

import argparse
import os
import sys

# Must import Isaac Sim/Lab extensions before torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert lower_body_multisegment MJCF to USD")
parser.add_argument("--headless", action="store_true", default=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now safe to import the rest
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

def main():
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mjcf_path = os.path.join(
        project_root, 
        "protomotions/data/assets/mjcf/lower_body_multisegment.xml"
    )
    usd_dir = os.path.join(project_root, "protomotions/data/assets/usd")
    usd_file = "lower_body_multisegment_flat.usda"
    
    print(f"Converting: {mjcf_path}")
    print(f"Output: {os.path.join(usd_dir, usd_file)}")
    
    # Create converter config - key settings for floating-base robot
    mjcf_converter_cfg = MjcfConverterCfg(
        asset_path=mjcf_path,
        usd_dir=usd_dir,
        usd_file_name=usd_file,
        fix_base=False,  # CRITICAL: floating base robot
        import_sites=False,
        force_usd_conversion=True,
        make_instanceable=False,  # Simpler structure
    )
    
    # Convert
    mjcf_converter = MjcfConverter(mjcf_converter_cfg)
    print(f"Generated USD: {mjcf_converter.usd_path}")
    
    # Close simulation
    simulation_app.close()

if __name__ == "__main__":
    main()
