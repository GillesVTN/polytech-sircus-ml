from models.heatmap import HeatMapBuilder

builder = HeatMapBuilder()

heatmap = builder.generate_heatmap(group="Patients", file_name="P_0011_73_M_JT.xlsx", sheet_name="mosob01.jpg - G")

builder.print_heatmap(heatmap)