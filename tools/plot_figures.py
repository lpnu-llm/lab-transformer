"""Validate the bundled original paper figure; do not redraw it."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    figure = ROOT / "figures/switch_transformer_paper.svg"
    assert ET.parse(figure).getroot().tag == "{http://www.w3.org/2000/svg}svg"
    print("✓ original Switch Transformer paper figure")
