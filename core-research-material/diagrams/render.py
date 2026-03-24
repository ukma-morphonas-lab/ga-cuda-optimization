#!/usr/bin/env python3
"""Render PlantUML diagrams to PNG using the public PlantUML server."""

import plantuml
from pathlib import Path

server = plantuml.PlantUML(url="http://www.plantuml.com/plantuml/png/")

diagrams_dir = Path(__file__).parent
for puml_file in sorted(diagrams_dir.glob("*.puml")):
    out_png = puml_file.with_suffix(".png")
    print(f"Rendering {puml_file.name} -> {out_png.name} ...")
    ok = server.processes_file(str(puml_file), outfile=str(out_png))
    if ok:
        size = out_png.stat().st_size
        print(f"  OK ({size:,} bytes)")
    else:
        print(f"  FAILED")
