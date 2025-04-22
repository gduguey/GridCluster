import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from pathlib import Path
import hashlib
from dataclasses import asdict
import json
import folium
from .settings import Config

class Visualizer:
    def __init__(self, config: Config, data: dict[str, pd.DataFrame | dict[str, pd.DataFrame]]):
        self.config = config
        self.base_path = Path(config.path.figures)
        self.data = data

    def save_figure(self, fig, fig_name):
        """
        Save the figure.
        """
        fig_path = self.base_path / fig_name
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, bbox_inches='tight')
        print(f"Figure saved as {fig_name} at {fig_path}")

    def plot_spatial_clusters(self, spatial_agg_results: dict, save_fig: bool = False):
        """
        Plot the map of nodes with representative nodes highlighted.
        """
        # Generate a colormap for the clusters
        colormap = plt.get_cmap('viridis', self.config.model_hyper.n_representative_nodes)

        # Plot the nodes
        fig = plt.figure(figsize=(7, 7))

        for idx, (rep, members) in enumerate(spatial_agg_results["clusters"].items()):
            cluster_color = colormap(idx)
            for member in members:
                if member != rep:
                    lat, lon = self.data["nodes"][["Lat", "Lon"]].iloc[member].values
                    plt.plot(lon, lat, 'o', color=cluster_color)
            rep_lat, rep_lon = self.data["nodes"][["Lat", "Lon"]].iloc[rep].values
            plt.plot(rep_lon, rep_lat, 'kx', markersize=8)
            plt.scatter(
                rep_lon, rep_lat,
                s=100,
                facecolors='none',
                edgecolors=cluster_color,
                linewidths=2,
                marker='o',
                zorder=5
            )

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Node Aggregation Map for {self.config.model_hyper.n_representative_nodes} Representative Nodes")
        weights = self.config.model_hyper.weights
        lines = [f"{k}: {v:.2f}" for k, v in weights.items()]
        txt = "\n".join(lines)
        at = AnchoredText(
            txt,
            prop=dict(size=8),
            frameon=True,
            loc='upper left',
        )
        at.patch.set_boxstyle("round,pad=0.5")
        at.patch.set_alpha(0.7)
        plt.gca().add_artist(at)
        plt.tight_layout()

        if save_fig:
            config_dict = {
                "data_preproc" : asdict(self.config.data_preproc),
                "model_hyper": self.config.model_hyper.__dict__
            }
            version_hash = hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:8]
            fig_name = f"spatial_clusters_from_{self.data["nodes"].shape[0]}_to_{self.config.model_hyper.n_representative_nodes}_{version_hash}.png"
            fig_path = Path("spatial_agg_results") / fig_name
            self.save_figure(fig, fig_path)
            
        return fig
    
    def plot_data(self, dataprocessor = None, save_fig: bool = False):
        """
        Plot the initial data.
        """
        if self.config.data_preproc.granularity == "high":
            if dataprocessor is None:
                raise ValueError("dataprocessor must be provided for high granularity data")
            solar_data = dataprocessor._processor.raw_solar
            county_data = dataprocessor._processor.demand_points_df
            solar_lat = solar_data.lat.values
            solar_lon = solar_data.lon.values

        # Center the map on New England
        NE_map = folium.Map(location=[43.0, -71.5], zoom_start=7)

        # Plot buses
        for idx, row in enumerate(self.data["nodes"].itertuples(index=False)):
            lat, lon = row.Lat, row.Lon
            tooltip = f"Bus ID: {idx}"
            
            folium.CircleMarker(
                location=(lat, lon),
                radius=2,
                fill=True,
                fill_opacity=0.7,
                tooltip=tooltip
            ).add_to(NE_map)

        if self.config.data_preproc.granularity == "high":
            # Plot counties
            for _, row in county_data.iterrows():
                lat, lon = row["Lat"], row["Lon"]
                triangle_vertices = [
                    (lat, lon),
                    (lat + 0.05, lon + 0.05),
                    (lat - 0.05, lon + 0.05),
                ]

                folium.Polygon(
                    locations=triangle_vertices,
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.5
                ).add_to(NE_map)

            # Plot solar and wind data
            for i in range(len(solar_lat)):
                folium.CircleMarker(
                    location=(solar_lat[i], solar_lon[i]),
                    radius=2,
                    color="orange",
                    fill=True,
                    fill_color="orange",
                    fill_opacity=0.7
                ).add_to(NE_map)

        if save_fig:
            fig_name = f"data_map_{self.config.data_preproc.granularity}.html"
            NE_map.save(fig_name)

        return NE_map

    def plot_network(self, save_fig: bool = False):
        """
        Plot the network of buses and lines.
        """
        fig = plt.figure(figsize=(10, 10))

        bus_to_node = self._precompute_bus_id_mappings()

        # Plot buses
        for idx, row in enumerate(self.data["nodes"].itertuples(index=False)):
            lat, lon = row.Lat, row.Lon
            plt.plot(lon, lat, 'o', color='blue', markersize=5)

        # Plot lines
        for line in self.data["branches"].itertuples(index=False):
            lat1, lon1 = self.data["nodes"].iloc[bus_to_node[line.from_bus_id]].Lat, self.data["nodes"].iloc[bus_to_node[line.from_bus_id]].Lon
            lat2, lon2 = self.data["nodes"].iloc[bus_to_node[line.to_bus_id]].Lat, self.data["nodes"].iloc[bus_to_node[line.to_bus_id]].Lon
            plt.plot([lon1, lon2], [lat1, lat2], color='gray', linewidth=0.5)

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Network of Buses and Lines")
        plt.tight_layout()

        if save_fig:
            fig_name = f"network_map_{self.config.data_preproc.granularity}.png"
            self.save_figure(fig, fig_name)

        return fig
    
    def _precompute_bus_id_mappings(self):
        """Precompute cluster mappings for bus IDs to representatives"""
        bus_to_node = {}
        nodes_df = self.data['nodes']

        for idx, row in enumerate(nodes_df.itertuples(index=False)):
            if self.config.data_preproc.granularity == "high":
                bus_ids = row.bus_id
                for bus_id in bus_ids:
                    bus_to_node[bus_id] = idx
            elif self.config.data_preproc.granularity == "low":
                bus_to_node[row.bus_id] = idx            

        return bus_to_node
    
    def plot_aggregated_network(self, results: dict[str, pd.DataFrame | dict[str, pd.DataFrame]], save_fig: bool = False):
        """
        Plot the aggregated network of buses and lines.
        """
        NE_map = folium.Map(location=[43.0, -71.5], zoom_start=7)

        # Plot buses
        for row in results["nodes"].itertuples(index=False):
            lat, lon = row.Lat, row.Lon
            folium.CircleMarker(
                location=(lat, lon),
                radius=2,
                fill=True,
                fill_opacity=0.7
            ).add_to(NE_map)

        # Plot lines
        for line in results["branches"].itertuples(index=False):
            lat1, lon1 = self.data["nodes"].iloc[line.from_bus_id].Lat, self.data["nodes"].iloc[line.from_bus_id].Lon
            lat2, lon2 = self.data["nodes"].iloc[line.to_bus_id].Lat, self.data["nodes"].iloc[line.to_bus_id].Lon
            tooltip = f"Susceptance: {line.b}, max flow: {line.max_flow}"
            folium.PolyLine(
                locations=[[lat1, lon1], [lat2, lon2]],
                color="blue",               
                weight=2,                   # Line thickness
                opacity=0.8,                 # Line opacity
                tooltip=tooltip
            ).add_to(NE_map)

        if save_fig:
            config_dict = {
                "data_preproc" : asdict(self.config.data_preproc),
                "model_hyper": self.config.model_hyper.__dict__
            }
            version_hash = hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:8]
            fig_name = f"aggregated_network_map_{self.config.data_preproc.granularity}_{version_hash}.html"
            fig_path = Path("spatial_agg_network") / fig_name
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            NE_map.save(fig_path)

        return NE_map
    

