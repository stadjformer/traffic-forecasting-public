from pathlib import Path
from typing import Optional

import contextily as cx
import geopandas as gpd
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sensor_locations(sensor_locations: pd.DataFrame) -> None:
    required_cols = ["latitude", "longitude"]
    if not all(col in sensor_locations.columns for col in required_cols):
        raise ValueError(f"expecting {required_cols} in supplied sensor locations")

    gdf = gpd.GeoDataFrame(
        sensor_locations,
        geometry=gpd.points_from_xy(
            sensor_locations.longitude, sensor_locations.latitude
        ),
        crs="EPSG:4326",
    )

    ax = gdf.plot(figsize=(10, 10), color="red", markersize=50, alpha=0.6)
    cx.add_basemap(ax, crs=gdf.crs, source=cx.providers.CartoDB.Positron)
    plt.show()


def animate_traffic_heatmap(
    values: np.ndarray,
    locations: pd.DataFrame,
    output_path: str | Path,
    fps: int = 4,
    duration_seconds: float = 3.0,
    timestamps: Optional[list[str]] = None,
    figsize: tuple[int, int] = (12, 10),
    cmap: str = "RdYlBu",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    values_comparison: Optional[np.ndarray] = None,
    title_left: str = "Ground Truth",
    title_right: str = "Model",
    cvd_friendly: bool = False,
    basemap_source: str = "CartoDB.Voyager",
) -> None:
    """
    Create an animated GIF heatmap of traffic values over time.

    Args:
        values: Array of shape [num_timesteps, num_sensors] containing traffic speeds or predictions
        locations: DataFrame with 'latitude' and 'longitude' columns for each sensor
        output_path: Path where the GIF will be saved
        fps: Frames per second for the animation (default: 4)
        duration_seconds: Target duration of the animation in seconds (default: 3.0)
        timestamps: Optional list of timestamp labels for each frame
        figsize: Figure size in inches (default: (12, 10))
        cmap: Colormap name (default: 'RdYlBu' - red=low/congestion, blue=high/free-flow)
        vmin: Minimum value for color scale (default: min of values)
        vmax: Maximum value for color scale (default: max of values)
        values_comparison: Optional second array for side-by-side comparison
        title_left: Title for left panel (default: "Ground Truth")
        title_right: Title for right panel (default: "Model")
        cvd_friendly: If True, use CVD-safe (Color Vision Deficiency) colormap (default: False)
        basemap_source: Basemap provider (default: "CartoDB.Voyager"). Examples: "CartoDB.Positron", "OpenStreetMap.Mapnik"
    """
    required_cols = ["latitude", "longitude"]
    if not all(col in locations.columns for col in required_cols):
        raise ValueError(f"expecting {required_cols} in supplied locations")

    if values.shape[1] != len(locations):
        raise ValueError(
            f"values shape {values.shape} doesn't match locations length {len(locations)}"
        )

    num_timesteps = values.shape[0]
    if timestamps is not None and len(timestamps) != num_timesteps:
        raise ValueError(
            f"timestamps length {len(timestamps)} doesn't match num_timesteps {num_timesteps}"
        )

    # Determine if we're doing side-by-side comparison
    dual_panel = values_comparison is not None

    if dual_panel and values_comparison.shape != values.shape:
        raise ValueError(
            f"values_comparison shape {values_comparison.shape} must match values shape {values.shape}"
        )

    # Set color scale limits - use global min/max for consistent colormap across frames
    if vmin is None:
        vmin = np.nanmin(values)
        if dual_panel:
            vmin = min(vmin, np.nanmin(values_comparison))
    if vmax is None:
        vmax = np.nanmax(values)
        if dual_panel:
            vmax = max(vmax, np.nanmax(values_comparison))

    # Ensure consistent scale across all frames
    global_vmin = vmin
    global_vmax = vmax

    # Use CVD-friendly colormap if requested
    if cvd_friendly:
        cmap = "viridis"  # Viridis is CVD-safe

    # Parse basemap source
    basemap_provider = eval(f"cx.providers.{basemap_source}")

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        locations,
        geometry=gpd.points_from_xy(locations.longitude, locations.latitude),
        crs="EPSG:4326",
    )

    # Create colormap normalization once for all frames
    from PIL import Image

    # Generate frames
    frames = []
    for t in range(num_timesteps):
        if dual_panel:
            # Create side-by-side layout with minimal gap using subplots
            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8))
            plt.subplots_adjust(
                wspace=0.02, left=0.02, right=0.98, top=0.88, bottom=0.15
            )

            # Left panel: primary values
            gdf_left = gdf.copy()
            gdf_left["value"] = values[t, :]
            gdf_left.plot(
                ax=ax_left,
                column="value",
                cmap=cmap,
                vmin=global_vmin,
                vmax=global_vmax,
                markersize=80,
                alpha=0.8,
                legend=False,
                zorder=2,
            )
            cx.add_basemap(ax_left, crs=gdf.crs, source=basemap_provider)
            ax_left.set_title(title_left, fontsize=14, fontweight="bold")
            ax_left.set_axis_off()

            # Right panel: comparison values
            gdf_right = gdf.copy()
            gdf_right["value"] = values_comparison[t, :]
            gdf_right.plot(
                ax=ax_right,
                column="value",
                cmap=cmap,
                vmin=global_vmin,
                vmax=global_vmax,
                markersize=80,
                alpha=0.8,
                legend=False,
                zorder=2,
            )
            cx.add_basemap(ax_right, crs=gdf.crs, source=basemap_provider)
            ax_right.set_title(title_right, fontsize=14, fontweight="bold")
            ax_right.set_axis_off()

            # Add timestamp info at top
            if timestamps is not None:
                fig.suptitle(timestamps[t], fontsize=16, fontweight="bold", y=0.95)
            else:
                fig.suptitle(
                    f"Timestep {t + 1}/{num_timesteps}",
                    fontsize=16,
                    fontweight="bold",
                    y=0.95,
                )

            # Add shared colorbar at the bottom
            from matplotlib import cm
            from matplotlib.colors import Normalize

            norm = Normalize(vmin=global_vmin, vmax=global_vmax)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.025])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
            cbar.set_label("Speed (mph)", fontsize=12)
        else:
            # Single panel layout
            fig = plt.figure(figsize=(figsize[0] * 0.9, figsize[1] * 0.8))
            ax_map = fig.add_axes([0.05, 0.1, 0.9, 0.85])

            # Add values to GeoDataFrame for this timestep
            gdf_t = gdf.copy()
            gdf_t["value"] = values[t, :]

            # Plot sensors colored by value with consistent scale
            gdf_t.plot(
                ax=ax_map,
                column="value",
                cmap=cmap,
                vmin=global_vmin,
                vmax=global_vmax,
                markersize=80,
                alpha=0.8,
                legend=False,
                zorder=2,
            )

            # Add basemap
            cx.add_basemap(ax_map, crs=gdf.crs, source=basemap_provider)

            # Add title with timestamp if provided
            if timestamps is not None:
                ax_map.set_title(
                    f"Traffic Speed - {timestamps[t]}", fontsize=14, fontweight="bold"
                )
            else:
                ax_map.set_title(
                    f"Traffic Speed - Timestep {t + 1}/{num_timesteps}",
                    fontsize=14,
                    fontweight="bold",
                )

            ax_map.set_axis_off()

            # Add colorbar at the bottom
            from matplotlib import cm
            from matplotlib.colors import Normalize

            norm = Normalize(vmin=global_vmin, vmax=global_vmax)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar_ax = fig.add_axes([0.25, 0.03, 0.5, 0.02])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
            cbar.set_label("Speed (mph)", fontsize=11)

        # Get map as image
        fig.canvas.draw()
        map_buf = fig.canvas.buffer_rgba()
        map_img = np.asarray(map_buf)[:, :, :3].copy()
        plt.close(fig)

        # No colorbar - just use the map directly
        frames.append(map_img)

    # Calculate frame duration for target animation duration
    total_frames = len(frames)
    duration_per_frame = duration_seconds / total_frames

    # Save as GIF with compression
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # For Medium's 5MB limit, compress moderately
    # Strategy: Skip every other frame + resize to 45% + reduce colors
    compressed_frames = []
    for i, frame in enumerate(frames):
        # Skip every other frame to halve file size
        if i % 2 != 0:
            continue
        img = Image.fromarray(frame)
        # Resize to 45% using high-quality Lanczos filter
        new_size = (int(img.width * 0.45), int(img.height * 0.45))
        img_resized = img.resize(new_size, Image.LANCZOS)
        compressed_frames.append(img_resized)

    # Adjust duration since we're skipping frames (PIL uses milliseconds)
    adjusted_duration_ms = int(duration_per_frame * 2 * 1000)

    # Use PIL instead of imageio for better GIF timing control
    compressed_frames[0].save(
        output_path,
        save_all=True,
        append_images=compressed_frames[1:],
        duration=adjusted_duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"Animation saved to {output_path}")


def visualize_node_degree(
    adj_matrix: np.ndarray,
    locations: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (12, 10),
    cmap: str = "YlOrRd",
    title: str = "Node Degree (Total Connection Strength)",
    cvd_friendly: bool = False,
) -> None:
    """
    Visualize node degree based on graph structure.

    Node degree is computed as the sum of absolute edge weights for each sensor,
    showing which sensors have the strongest total connections in the graph.

    Args:
        adj_matrix: Adjacency matrix of shape [num_sensors, num_sensors]
        locations: DataFrame with 'latitude' and 'longitude' columns for each sensor
        output_path: Optional path to save the figure (if None, displays with plt.show())
        figsize: Figure size in inches (default: (12, 10))
        cmap: Colormap name (default: 'YlOrRd' - yellow=low degree, red=high degree)
        title: Plot title (default: "Node Degree (Total Connection Strength)")
        cvd_friendly: If True, use CVD-safe (Color Vision Deficiency) colormap (default: False)
    """
    required_cols = ["latitude", "longitude"]
    if not all(col in locations.columns for col in required_cols):
        raise ValueError(f"expecting {required_cols} in supplied locations")

    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError(f"adj_matrix must be square, got shape {adj_matrix.shape}")

    if adj_matrix.shape[0] != len(locations):
        raise ValueError(
            f"adj_matrix shape {adj_matrix.shape} doesn't match locations length {len(locations)}"
        )

    # Compute node degree as sum of absolute edge weights
    degree = np.abs(adj_matrix).sum(axis=1)

    # Normalize to [0, 1] for better visualization
    degree_normalized = degree / np.max(degree) if np.max(degree) > 0 else degree

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        locations,
        geometry=gpd.points_from_xy(locations.longitude, locations.latitude),
        crs="EPSG:4326",
    )
    gdf["degree"] = degree
    gdf["degree_normalized"] = degree_normalized

    # Use CVD-friendly colormap if requested
    if cvd_friendly:
        cmap = "viridis"

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot sensors sized and colored by degree
    gdf.plot(
        ax=ax,
        column="degree_normalized",
        cmap=cmap,
        markersize=gdf["degree_normalized"] * 200 + 20,  # Size varies with degree
        alpha=0.7,
        legend=True,
        legend_kwds={"label": "Normalized degree", "shrink": 0.8},
    )

    # Add basemap
    cx.add_basemap(ax, crs=gdf.crs, source=cx.providers.CartoDB.Positron)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()

    # Save or show
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {output_path}")
        plt.close(fig)
    else:
        plt.show()


def get_top_neighbors(
    adj_matrix: np.ndarray,
    node_id: int,
    top_k: int = 5,
    direction: str = "outgoing",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the top-K strongest neighbors for a given node.

    Args:
        adj_matrix: Adjacency matrix of shape [num_sensors, num_sensors]
        node_id: Node index to analyze
        top_k: Number of top neighbors to return
        direction: 'outgoing' (nodes this node affects, i.e., row),
                  'incoming' (nodes that affect this node, i.e., column),
                  'both' (union of both directions)

    Returns:
        neighbor_ids: Array of neighbor node indices (sorted by weight, descending)
        weights: Array of corresponding edge weights (for 'both', uses max of two directions)
    """
    if node_id < 0 or node_id >= adj_matrix.shape[0]:
        raise ValueError(f"node_id {node_id} out of range [0, {adj_matrix.shape[0]})")

    if direction not in ("outgoing", "incoming", "both"):
        raise ValueError(
            f"direction must be 'outgoing', 'incoming', or 'both', got {direction}"
        )

    # Get edge weights based on direction
    if direction == "outgoing":
        edge_weights = adj_matrix[node_id, :]  # Row: edges FROM this node
    elif direction == "incoming":
        edge_weights = adj_matrix[:, node_id]  # Column: edges TO this node
    else:  # both
        outgoing = adj_matrix[node_id, :]
        incoming = adj_matrix[:, node_id]
        edge_weights = np.maximum(outgoing, incoming)  # Union via max

    # Get top-K neighbors (excluding self-loops)
    sorted_indices = np.argsort(edge_weights)[::-1]
    # Filter out self and zero-weight edges
    valid_indices = [i for i in sorted_indices if i != node_id and edge_weights[i] > 0]
    top_k_indices = valid_indices[:top_k]

    neighbor_ids = np.array(top_k_indices)
    weights = edge_weights[top_k_indices]

    return neighbor_ids, weights


def get_neighborhood_stats(
    adj_matrix: np.ndarray,
    weighted: bool = False,
) -> dict[str, np.ndarray]:
    """
    Compute statistics about the neighborhood structure of the adjacency matrix.

    Args:
        adj_matrix: Adjacency matrix of shape [num_sensors, num_sensors]
        weighted: If True, compute weighted degrees (sum of edge weights).
                 If False, compute unweighted degrees (count of nonzero edges).

    Returns:
        Dictionary with:
            'out_degrees': Outgoing degree per node (row-wise)
            'in_degrees': Incoming degree per node (column-wise)
            'total_degrees': Total degree per node (union of both directions)

        For weighted=True:
            - out_degrees: Sum of outgoing edge weights (should be ~1.0 for row-stochastic)
            - in_degrees: Sum of incoming edge weights
            - total_degrees: Sum of all edge weights touching this node

        For weighted=False:
            - out_degrees: Count of outgoing neighbors
            - in_degrees: Count of incoming neighbors
            - total_degrees: Count of unique neighbors in either direction
    """
    eps = 1e-9

    if weighted:
        # Weighted degrees: sum of edge weights
        out_degrees = np.sum(adj_matrix, axis=1)  # Sum per row
        in_degrees = np.sum(adj_matrix, axis=0)  # Sum per column
        total_degrees = out_degrees + in_degrees  # Total weight
    else:
        # Unweighted degrees: count nonzero edges
        adj_binary = (adj_matrix > eps).astype(int)

        out_degrees = np.sum(adj_binary, axis=1)  # Nonzero per row
        in_degrees = np.sum(adj_binary, axis=0)  # Nonzero per column

        # Total degree: union of incoming and outgoing
        # For each node, count unique neighbors in either direction
        total_degrees = np.zeros(adj_matrix.shape[0], dtype=int)
        for i in range(adj_matrix.shape[0]):
            outgoing_neighbors = set(np.where(adj_binary[i, :] > 0)[0])
            incoming_neighbors = set(np.where(adj_binary[:, i] > 0)[0])
            total_degrees[i] = len(outgoing_neighbors | incoming_neighbors)

    return {
        "out_degrees": out_degrees,
        "in_degrees": in_degrees,
        "total_degrees": total_degrees,
    }


def visualize_node_and_neighbors(
    adj_matrix: np.ndarray,
    locations: pd.DataFrame,
    node_id: int,
    top_k: int = 5,
    figsize: tuple[int, int] = (12, 10),
    output_path: Optional[str | Path] = None,
    show_edges: bool = False,
    direction: str = "outgoing",
    cvd_friendly: bool = False,
) -> None:
    """
    Visualize a specific node and its top-K neighbors on the map.

    Args:
        adj_matrix: Adjacency matrix of shape [num_sensors, num_sensors]
        locations: DataFrame with 'latitude' and 'longitude' columns
        node_id: Node index to visualize
        top_k: Number of top neighbors to highlight
        figsize: Figure size in inches
        output_path: Optional path to save the figure
        show_edges: If True, draw edges from focal node to neighbors (default: False)
        direction: 'outgoing', 'incoming', or 'both' - which neighbors to show
        cvd_friendly: If True, use CVD-safe (Color Vision Deficiency) colormap (default: False)
    """
    # Get top neighbors
    neighbor_ids, weights = get_top_neighbors(
        adj_matrix, node_id, top_k, direction=direction
    )

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        locations,
        geometry=gpd.points_from_xy(locations.longitude, locations.latitude),
        crs="EPSG:4326",
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Plot all sensors in gray (background)
    gdf.plot(ax=ax, color="lightgray", markersize=20, alpha=0.4, zorder=1)

    # Optionally draw edges
    if show_edges and len(neighbor_ids) > 0:
        focal_lon = locations.iloc[node_id]["longitude"]
        focal_lat = locations.iloc[node_id]["latitude"]
        max_weight = weights.max() if len(weights) > 0 else 1

        for neighbor_id, weight in zip(neighbor_ids, weights):
            neighbor_lon = locations.iloc[neighbor_id]["longitude"]
            neighbor_lat = locations.iloc[neighbor_id]["latitude"]
            line_width = (weight / max_weight) * 3 + 0.5

            ax.plot(
                [focal_lon, neighbor_lon],
                [focal_lat, neighbor_lat],
                "b-",
                alpha=0.6,
                linewidth=line_width,
                zorder=2,
            )

    # Plot top neighbors with size/color based on relative importance
    if len(neighbor_ids) > 0:
        # Get global max weight for consistent normalization
        global_max = np.abs(adj_matrix).max()

        # Normalize weights using global scale
        weights_normalized = weights / global_max if global_max > 0 else weights

        # Add weight info to GeoDataFrame
        gdf_neighbors = gdf.iloc[neighbor_ids].copy()
        gdf_neighbors["weight"] = weights
        gdf_neighbors["weight_norm"] = weights_normalized

        # Use CVD-friendly colormap if requested
        neighbor_cmap = "viridis" if cvd_friendly else "YlOrRd"

        # Plot neighbors colored by weight with FIXED scale [0, 1]
        gdf_neighbors.plot(
            ax=ax,
            column="weight_norm",
            cmap=neighbor_cmap,
            vmin=0,
            vmax=1,
            markersize=weights_normalized * 100 + 30,
            alpha=0.8,
            zorder=3,
            legend=True,
            legend_kwds={"label": "Relative importance", "shrink": 0.6},
        )

    # Plot focal node in red (on top)
    gdf.iloc[[node_id]].plot(ax=ax, color="red", markersize=150, alpha=0.9, zorder=4)

    # Add basemap
    cx.add_basemap(ax, crs=gdf.crs, source=cx.providers.CartoDB.Positron)

    # Title with node degree info
    if direction == "outgoing":
        node_degree = np.sum(adj_matrix[node_id, :] > 1e-9)
        direction_label = "Outgoing"
    elif direction == "incoming":
        node_degree = np.sum(adj_matrix[:, node_id] > 1e-9)
        direction_label = "Incoming"
    else:
        out_deg = np.sum(adj_matrix[node_id, :] > 1e-9)
        in_deg = np.sum(adj_matrix[:, node_id] > 1e-9)
        node_degree = out_deg + in_deg  # Could count union instead
        direction_label = "Both"

    ax.set_title(
        f"Node {node_id} ({direction_label} Degree={node_degree}) and Top {top_k} Neighbors",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_axis_off()
    plt.tight_layout()

    # Save or show
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {output_path}")
        plt.close(fig)
    else:
        plt.show()


def animate_node_neighbors(
    adj_matrix: np.ndarray,
    locations: pd.DataFrame,
    node_ids: list[int] | np.ndarray,
    output_path: str | Path,
    top_k: int = 5,
    fps: int = 2,
    duration_seconds: float = None,
    show_edges: bool = False,
    figsize: tuple[int, int] = (12, 10),
    cvd_friendly: bool = False,
) -> None:
    """
    Create an animated GIF looping through multiple nodes and their neighbors.

    Args:
        adj_matrix: Adjacency matrix of shape [num_sensors, num_sensors]
        locations: DataFrame with 'latitude' and 'longitude' columns
        node_ids: List of node indices to visualize
        output_path: Path where the GIF will be saved
        top_k: Number of top neighbors to show per node
        fps: Frames per second (default: 2 for slower viewing)
        duration_seconds: Total duration in seconds (if None, uses fps)
        show_edges: If True, draw edges from focal node to neighbors
        figsize: Figure size in inches
        cvd_friendly: If True, use CVD-safe (Color Vision Deficiency) colormap (default: False)
    """
    gdf = gpd.GeoDataFrame(
        locations,
        geometry=gpd.points_from_xy(locations.longitude, locations.latitude),
        crs="EPSG:4326",
    )

    # Get global max weight for consistent normalization across all frames
    global_max = np.abs(adj_matrix).max()

    # Use CVD-friendly colormap if requested
    neighbor_cmap = "viridis" if cvd_friendly else "YlOrRd"

    frames = []
    for node_id in node_ids:
        # Get top neighbors
        neighbor_ids, weights = get_top_neighbors(adj_matrix, node_id, top_k)

        fig, ax = plt.subplots(figsize=figsize)

        # Plot all sensors in gray (background)
        gdf.plot(ax=ax, color="lightgray", markersize=20, alpha=0.4, zorder=1)

        # Optionally draw edges
        if show_edges and len(neighbor_ids) > 0:
            focal_lon = locations.iloc[node_id]["longitude"]
            focal_lat = locations.iloc[node_id]["latitude"]
            max_weight = weights.max() if len(weights) > 0 else 1

            for neighbor_id, weight in zip(neighbor_ids, weights):
                neighbor_lon = locations.iloc[neighbor_id]["longitude"]
                neighbor_lat = locations.iloc[neighbor_id]["latitude"]
                line_width = (weight / max_weight) * 3 + 0.5

                ax.plot(
                    [focal_lon, neighbor_lon],
                    [focal_lat, neighbor_lat],
                    "b-",
                    alpha=0.6,
                    linewidth=line_width,
                    zorder=2,
                )

        # Plot neighbors with heatmap using global scale
        if len(neighbor_ids) > 0:
            weights_normalized = weights / global_max if global_max > 0 else weights
            gdf_neighbors = gdf.iloc[neighbor_ids].copy()
            gdf_neighbors["weight_norm"] = weights_normalized

            gdf_neighbors.plot(
                ax=ax,
                column="weight_norm",
                cmap=neighbor_cmap,
                vmin=0,
                vmax=1,
                markersize=weights_normalized * 100 + 30,
                alpha=0.8,
                zorder=3,
                legend=True,
                legend_kwds={"label": "Relative importance", "shrink": 0.6},
            )

        # Plot focal node in red
        gdf.iloc[[node_id]].plot(
            ax=ax, color="red", markersize=150, alpha=0.9, zorder=4
        )

        # Add basemap
        cx.add_basemap(ax, crs=gdf.crs, source=cx.providers.CartoDB.Positron)

        # Title
        node_degree = np.abs(adj_matrix[node_id, :]).sum()
        ax.set_title(
            f"Node {node_id} (degree={node_degree:.1f}) and Top {top_k} Neighbors",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_axis_off()
        plt.tight_layout()

        # Save frame
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)
        frame_rgb = frame[:, :, :3].copy()
        frames.append(frame_rgb)
        plt.close(fig)

    # Calculate frame duration
    if duration_seconds is not None:
        duration_per_frame = duration_seconds / len(frames)
    else:
        duration_per_frame = 1.0 / fps

    # Save as GIF with higher quality
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        output_path,
        frames,
        duration=duration_per_frame,
        loop=0,
        quantizer="nq",
        palettesize=256,
    )
    print(f"Animation saved to {output_path}")


def create_node_degree_comparison(
    adj_mx_ground_truth: np.ndarray,
    adj_mx_model: np.ndarray,
    locations: pd.DataFrame,
    output_path: Path,
    dataset_name: str,
    degree_type: str = "weighted_in",
    cvd_friendly: bool = False,
    basemap_source: str = "CartoDB.Voyager",
) -> None:
    """Create side-by-side node degree comparison visualization.

    Args:
        adj_mx_ground_truth: Ground truth adjacency matrix
        adj_mx_model: Model's learned adjacency matrix
        locations: DataFrame with sensor locations
        output_path: Where to save the PNG
        dataset_name: Name of dataset (for title)
        degree_type: Type of degree to visualize:
            - 'weighted_in': Incoming weighted degree
            - 'weighted_out': Outgoing weighted degree
            - 'weighted_total': Total weighted degree (in + out)
            - 'unweighted_in': Count of incoming neighbors
            - 'unweighted_out': Count of outgoing neighbors
            - 'unweighted_total': Count of neighbors in either direction
        cvd_friendly: If True, use CVD-safe (Color Vision Deficiency) colormap (default: False)
        basemap_source: Basemap provider (default: "CartoDB.Voyager"). Examples: "CartoDB.Positron", "OpenStreetMap.Mapnik"
    """
    import contextily as cx
    from matplotlib import cm
    from matplotlib.colors import Normalize

    # Parse degree type
    if degree_type.startswith("weighted"):
        weighted = True
        direction = degree_type.replace("weighted_", "")
    elif degree_type.startswith("unweighted"):
        weighted = False
        direction = degree_type.replace("unweighted_", "")
    else:
        raise ValueError(f"Invalid degree_type: {degree_type}")

    if direction not in ("in", "out", "total"):
        raise ValueError(f"Invalid direction in degree_type: {direction}")

    # Compute node degrees
    stats_gt = get_neighborhood_stats(adj_mx_ground_truth, weighted=weighted)
    stats_model = get_neighborhood_stats(adj_mx_model, weighted=weighted)

    degree_key = f"{direction}_degrees"
    degree_gt = stats_gt[degree_key]
    degree_model = stats_model[degree_key]

    # Normalize
    max_degree = max(degree_gt.max(), degree_model.max())
    degree_gt_norm = degree_gt / max_degree if max_degree > 0 else degree_gt
    degree_model_norm = degree_model / max_degree if max_degree > 0 else degree_model

    gdf = gpd.GeoDataFrame(
        locations,
        geometry=gpd.points_from_xy(locations.longitude, locations.latitude),
        crs="EPSG:4326",
    )

    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    plt.subplots_adjust(wspace=0.02, left=0.02, right=0.98, top=0.88, bottom=0.12)

    # Create descriptive title
    degree_label = {
        "weighted_in": "Weighted Incoming Degree (Sum of Incoming Edge Weights)",
        "weighted_out": "Weighted Outgoing Degree (Sum of Outgoing Edge Weights)",
        "weighted_total": "Weighted Total Degree (Sum of All Edge Weights)",
        "unweighted_in": "Incoming Degree (Count of Incoming Neighbors)",
        "unweighted_out": "Outgoing Degree (Count of Outgoing Neighbors)",
        "unweighted_total": "Total Degree (Count of All Neighbors)",
    }[degree_type]

    fig.suptitle(
        f"Node Degree Comparison: {degree_label}",
        fontsize=16,
        fontweight="bold",
        y=0.94,
    )

    # Use CVD-friendly colormap if requested
    degree_cmap = "viridis" if cvd_friendly else "plasma"

    # Parse basemap source
    basemap_provider = eval(f"cx.providers.{basemap_source}")

    # Left panel: Ground truth
    gdf_gt = gdf.copy()
    gdf_gt["degree_normalized"] = degree_gt_norm
    gdf_gt.plot(
        ax=ax1,
        column="degree_normalized",
        cmap=degree_cmap,
        markersize=80,
        alpha=0.8,
        legend=False,
        vmin=0,
        vmax=1,
    )
    cx.add_basemap(ax1, crs=gdf.crs, source=basemap_provider)
    ax1.set_title("Ground Truth (Geographic Graph)", fontsize=14, fontweight="bold")
    ax1.set_axis_off()

    # Right panel: Model
    gdf_model = gdf.copy()
    gdf_model["degree_normalized"] = degree_model_norm
    gdf_model.plot(
        ax=ax2,
        column="degree_normalized",
        cmap=degree_cmap,
        markersize=80,
        alpha=0.8,
        legend=False,
        vmin=0,
        vmax=1,
    )
    cx.add_basemap(ax2, crs=gdf.crs, source=basemap_provider)
    ax2.set_title("Model (Learned Graph)", fontsize=14, fontweight="bold")
    ax2.set_axis_off()

    # Add shared colorbar
    norm = Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap=degree_cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Normalized degree", fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Node degree comparison saved: {output_path}")


def create_geographic_nodes_animation(
    adj_mx_ground_truth: np.ndarray,
    adj_mx_model: np.ndarray,
    locations: pd.DataFrame,
    output_path: Path,
    selected_nodes: np.ndarray,
    frame_seconds: float = 3.0,
    cvd_friendly: bool = False,
    basemap_source: str = "CartoDB.Voyager",
) -> None:
    """Create animated GIF showing graph adjacency for selected nodes.

    Args:
        adj_mx_ground_truth: Ground truth adjacency matrix
        adj_mx_model: Model's learned adjacency matrix
        locations: DataFrame with sensor locations
        output_path: Where to save the GIF
        selected_nodes: Array of node IDs to visualize
        frame_seconds: Duration per frame in seconds
        cvd_friendly: If True, use CVD-safe (Color Vision Deficiency) colormap (default: False)
        basemap_source: Basemap provider (default: "CartoDB.Voyager"). Examples: "CartoDB.Positron", "OpenStreetMap.Mapnik"
    """
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    from PIL import Image as PILImage

    # Compute global min/max for consistent coloring
    positive_weights_gt = adj_mx_ground_truth[adj_mx_ground_truth > 0]
    positive_weights_model = adj_mx_model[adj_mx_model > 0]

    all_weights = np.concatenate([positive_weights_gt, positive_weights_model])
    if all_weights.size == 0:
        global_min, global_max = 0.0, 1.0
    else:
        global_min = float(all_weights.min())
        global_max = float(all_weights.max())

    if global_min == global_max:
        global_min, global_max = 0.0, max(global_max, 1.0)

    gdf_all = gpd.GeoDataFrame(
        locations,
        geometry=gpd.points_from_xy(locations.longitude, locations.latitude),
        crs="EPSG:4326",
    )

    norm = Normalize(vmin=global_min, vmax=global_max)

    # Use CVD-friendly colormap if requested
    cmap_name = "viridis" if cvd_friendly else "plasma"
    cmap = plt.colormaps.get_cmap(cmap_name)

    # Parse basemap source
    basemap_provider = eval(f"cx.providers.{basemap_source}")

    frames = []
    for node_id in selected_nodes:
        sensor_id = int(locations.iloc[node_id]["sensor_id"])

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        plt.subplots_adjust(wspace=0.02, left=0.02, right=0.98, top=0.86, bottom=0.18)

        fig.suptitle(
            f"Graph Adjacency for Sensor {sensor_id}",
            fontsize=14,
            fontweight="bold",
            y=0.94,
        )

        for ax, adj_mx, title in [
            (axes[0], adj_mx_ground_truth, "Ground Truth (Geographic Graph)"),
            (axes[1], adj_mx_model, "Model (Learned Graph)"),
        ]:
            edge_weights = adj_mx[node_id, :]
            neighbor_mask = (edge_weights > 0) & (
                np.arange(len(edge_weights)) != node_id
            )
            neighbor_ids = np.where(neighbor_mask)[0]
            weights = edge_weights[neighbor_ids]

            # Plot all sensors in gray
            gdf_all.plot(ax=ax, color="lightgray", markersize=25, alpha=0.4, zorder=1)

            # Plot neighbors colored by connection strength
            if len(neighbor_ids) > 0:
                lon_neighbors = locations.iloc[neighbor_ids]["longitude"].to_numpy()
                lat_neighbors = locations.iloc[neighbor_ids]["latitude"].to_numpy()

                ax.scatter(
                    lon_neighbors,
                    lat_neighbors,
                    s=80,
                    c=weights,
                    cmap=cmap_name,
                    norm=norm,
                    alpha=0.9,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=3,
                )

            # Plot focal node in black (colorblind-friendly)
            focal_lon = float(locations.iloc[node_id]["longitude"])
            focal_lat = float(locations.iloc[node_id]["latitude"])
            ax.scatter(
                focal_lon,
                focal_lat,
                s=120,
                c="black",
                edgecolors="white",
                linewidths=2,
                zorder=5,
            )

            cx.add_basemap(ax, crs=gdf_all.crs, source=basemap_provider)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_axis_off()

        # Add categorical legend
        focal_patch = mpatches.Patch(color="black", label="Focal sensor")
        neighbor_patch = mpatches.Patch(color=cmap(0.7), label="Connected neighbors")
        other_patch = mpatches.Patch(color="lightgray", label="Other sensors")
        fig.legend(
            handles=[focal_patch, neighbor_patch, other_patch],
            loc="lower center",
            ncol=3,
            fontsize=10,
            frameon=False,
            bbox_to_anchor=(0.5, 0.09),
        )

        # Add colorbar to show connection strength
        from matplotlib import cm

        sm = cm.ScalarMappable(cmap=cmap_name, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.25, 0.03, 0.5, 0.025])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Connection strength (edge weight)", fontsize=11)

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3].copy()
        plt.close(fig)
        frames.append(frame)

    # Compress and save
    compressed_frames = []
    for frame in frames:
        img = PILImage.fromarray(frame)
        new_size = (int(img.width * 0.55), int(img.height * 0.55))
        img_resized = img.resize(new_size, PILImage.LANCZOS)
        compressed_frames.append(img_resized)

    duration_ms = int(frame_seconds * 1000)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compressed_frames[0].save(
        output_path,
        save_all=True,
        append_images=compressed_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Geographic nodes animation saved: {output_path}")
    print(
        f"  Size: {file_size_mb:.2f} MB, Frames: {len(compressed_frames)}, Duration: {len(compressed_frames) * frame_seconds:.1f}s"
    )


def select_geographically_dispersed_nodes(
    locations: pd.DataFrame,
    degree: np.ndarray,
    grid_rows: int = 4,
    grid_cols: int = 4,
) -> np.ndarray:
    """Select geographically dispersed nodes with high degree.

    Divides the region into a grid and selects the highest-degree node from each cell.

    Args:
        locations: DataFrame with latitude/longitude
        degree: Node degree values
        grid_rows: Number of grid rows
        grid_cols: Number of grid columns

    Returns:
        Array of selected node indices
    """
    lat_min, lat_max = locations["latitude"].min(), locations["latitude"].max()
    lon_min, lon_max = locations["longitude"].min(), locations["longitude"].max()

    lat_bins = np.linspace(lat_min, lat_max, grid_rows + 1)
    lon_bins = np.linspace(lon_min, lon_max, grid_cols + 1)

    selected_nodes = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            mask = (
                (locations["latitude"] >= lat_bins[i])
                & (locations["latitude"] < lat_bins[i + 1])
                & (locations["longitude"] >= lon_bins[j])
                & (locations["longitude"] < lon_bins[j + 1])
            )
            cell_indices = np.where(mask)[0]

            if len(cell_indices) > 0:
                cell_degrees = degree[cell_indices]
                best_in_cell = cell_indices[np.argmax(cell_degrees)]
                selected_nodes.append(best_in_cell)

    return np.array(selected_nodes)
