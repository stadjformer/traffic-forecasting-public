"""Preprocess LargeST dataset to METR-LA/PEMS-BAY format - memory efficient version."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

REGIONS = {
    "GLA": ["Los Angeles", "Orange", "San Bernardino", "Riverside", "Ventura"],
    "GBA": ["San Francisco", "Alameda", "Contra Costa", "San Mateo", "Santa Clara", "Marin", "Solano", "Sonoma", "Napa"],
    "SD": ["San Diego"],
}

def load_metadata(raw_dir="data/LargeST-raw"):
    meta = pd.read_csv(Path(raw_dir) / "ca_meta.csv")
    return meta

def filter_sensors_by_region(meta, region="GLA"):
    if region not in REGIONS:
        raise ValueError(f"Region must be one of {list(REGIONS.keys())}")
    counties = REGIONS[region]
    filtered = meta[meta["County"].isin(counties)].copy()
    filtered["new_id"] = range(len(filtered))
    print(f"\n{region} Region:")
    print(f"  Counties: {', '.join(counties)}")
    print(f"  Sensors: {len(filtered)}")
    return filtered

def load_traffic_data(year, raw_dir="data/LargeST-raw"):
    h5_file = Path(raw_dir) / f"ca_his_raw_{year}.h5"
    print(f"\nLoading {h5_file}")
    df = pd.read_hdf(h5_file)
    print(f"  Shape: {df.shape}")
    print(f"  Time range: {df.index[0]} to {df.index[-1]}")
    return df

def create_sliding_windows_by_month(traffic_df, sensor_meta, output_path, in_steps=12, out_steps=12, sensors_per_batch=50):
    """Create sliding windows by processing month x sensor_batch chunks to avoid memory leaks."""
    print(f"\nCreating sliding windows ({in_steps} input + {out_steps} output)")
    sensor_ids = sensor_meta["ID"].astype(str).tolist()
    traffic_df = traffic_df[sensor_ids]
    num_sensors = len(sensor_ids)

    print(f"  Total sensors: {num_sensors}")
    print(f"  Sensors per batch: {sensors_per_batch}")

    id_map = sensor_meta.set_index("ID")["new_id"].to_dict()
    window_size = in_steps + out_steps

    # Create tmp directory for checkpoints
    tmp_dir = output_path / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    # Write to parquet file
    temp_parquet = output_path / "temp_all_samples.parquet"
    writer = None
    total_samples = 0

    # Get unique months
    months = traffic_df.index.to_period('M').unique().sort_values()
    num_batches = (num_sensors + sensors_per_batch - 1) // sensors_per_batch

    total_chunks = len(months) * num_batches
    print(f"  Processing {len(months)} months x {num_batches} sensor batches = {total_chunks} chunks with checkpointing")

    pbar = tqdm(total=total_chunks, desc="Processing chunks")

    for month in months:
        # Get data for this month + overlap for windows
        month_start = month.to_timestamp()
        month_end = (month + 1).to_timestamp()

        # Need extra rows for window context
        context_start = month_start - pd.Timedelta(minutes=5 * (in_steps - 1))

        # Select time range with context
        month_data = traffic_df.loc[context_start:month_end]

        if len(month_data) < window_size:
            pbar.update(num_batches)  # Skip all batches for this month
            continue

        flow_data = month_data.values
        timestamps = month_data.index.tolist()
        num_timesteps = len(timestamps)
        num_windows = num_timesteps - window_size + 1

        # Process sensor batches for this month
        for batch_idx in range(num_batches):
            pbar.update(1)
            batch_start = batch_idx * sensors_per_batch
            batch_end = min(batch_start + sensors_per_batch, num_sensors)

            # Check if chunk checkpoint exists
            chunk_checkpoint = tmp_dir / f"month_{month}_batch_{batch_idx:03d}.parquet"

            if chunk_checkpoint.exists():
                # Load existing checkpoint
                chunk_table = pq.read_table(chunk_checkpoint)
                if writer is None:
                    writer = pq.ParquetWriter(temp_parquet, chunk_table.schema)
                writer.write_table(chunk_table)
                total_samples += len(chunk_table)
                continue

            # Process sensors in this batch
            batch_samples = []
            for sensor_idx in range(batch_start, batch_end):
                original_id = int(sensor_ids[sensor_idx])
                new_id = id_map[original_id]

                for t in range(num_windows):
                    t0_idx = t + in_steps - 1

                    # Only include windows where t0 is in the current month
                    if timestamps[t0_idx] < month_start or timestamps[t0_idx] >= month_end:
                        continue

                    sample = {"node_id": new_id, "t0_timestamp": timestamps[t0_idx].isoformat()}

                    for i in range(in_steps):
                        offset = i - in_steps + 1
                        sample[f"x_t{offset:+d}_d0"] = flow_data[t + i, sensor_idx]
                        # Add normalized time-of-day [0, 1) to match METR-LA/PEMS-BAY format
                        ts = timestamps[t + i]
                        tod = (ts.hour * 3600 + ts.minute * 60 + ts.second) / 86400.0
                        sample[f"x_t{offset:+d}_d1"] = tod

                    for i in range(out_steps):
                        offset = i + 1
                        sample[f"y_t{offset:+d}_d0"] = flow_data[t + in_steps + i, sensor_idx]
                        # Note: y's d1 (TOD) not needed - model only uses y[..., 0:1] for loss

                    batch_samples.append(sample)

            # Write chunk's data to parquet and save checkpoint
            if batch_samples:
                batch_df = pd.DataFrame(batch_samples)
                table = pa.Table.from_pandas(batch_df, preserve_index=False)

                # Save checkpoint
                pq.write_table(table, chunk_checkpoint)

                if writer is None:
                    writer = pq.ParquetWriter(temp_parquet, table.schema)

                writer.write_table(table)
                total_samples += len(batch_df)

                # Explicitly delete to free memory
                del batch_samples, batch_df, table

        # Explicitly delete month data to free memory
        del month_data, flow_data, timestamps

    pbar.close()

    if writer:
        writer.close()

    print(f"  Created {total_samples:,} total samples in parquet")
    return temp_parquet

def split_with_duckdb(temp_parquet, output_path, train_ratio=0.7, val_ratio=0.1):
    """Use DuckDB to split data without loading into memory."""
    print(f"\nSplitting data with DuckDB (zero memory overhead)")

    con = duckdb.connect(':memory:')

    # Get unique timestamps and calculate split points
    print("  Calculating split boundaries")
    result = con.execute(f"""
        SELECT DISTINCT t0_timestamp
        FROM read_parquet('{temp_parquet}')
        ORDER BY t0_timestamp
    """).fetchall()

    unique_times = [row[0] for row in result]
    n = len(unique_times)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_end_time = unique_times[train_end - 1]
    val_end_time = unique_times[val_end - 1]

    # Write train split
    print("  Writing train split")
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{temp_parquet}')
            WHERE t0_timestamp <= '{train_end_time}'
        ) TO '{output_path / "train.parquet"}' (FORMAT PARQUET)
    """)

    # Write val split
    print("  Writing validation split")
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{temp_parquet}')
            WHERE t0_timestamp > '{train_end_time}' AND t0_timestamp <= '{val_end_time}'
        ) TO '{output_path / "val.parquet"}' (FORMAT PARQUET)
    """)

    # Write test split
    print("  Writing test split")
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{temp_parquet}')
            WHERE t0_timestamp > '{val_end_time}'
        ) TO '{output_path / "test.parquet"}' (FORMAT PARQUET)
    """)

    # Get counts
    train_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path / 'train.parquet'}')").fetchone()[0]
    val_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path / 'val.parquet'}')").fetchone()[0]
    test_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path / 'test.parquet'}')").fetchone()[0]

    print(f"  Train: {train_count:,} samples")
    print(f"  Val:   {val_count:,} samples")
    print(f"  Test:  {test_count:,} samples")

    con.close()

def create_adjacency_matrix(sensor_meta):
    print("\nCreating adjacency matrix")
    from sklearn.metrics.pairwise import haversine_distances
    coords = sensor_meta[["Lat", "Lng"]].values
    coords_rad = np.radians(coords)
    dist_matrix = haversine_distances(coords_rad) * 6371
    threshold_km = 10
    adj_mx = (dist_matrix < threshold_km).astype(float)
    print(f"  Shape: {adj_mx.shape}")
    return adj_mx, dist_matrix

def main(region="GLA", year="2019", output_dir=None, raw_dir="data/LargeST-raw"):
    print("=" * 80)
    print(f"LargeST Preprocessing: {region} ({year})")
    print("=" * 80)
    if output_dir is None:
        output_dir = f"data/LargeST-{region}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(raw_dir)
    sensor_meta = filter_sensors_by_region(meta, region)
    traffic_df = load_traffic_data(year, raw_dir)

    # Create sliding windows and write to parquet (month by month)
    temp_parquet = create_sliding_windows_by_month(traffic_df, sensor_meta, output_path)

    # Free memory
    del traffic_df

    # Split using DuckDB (no memory overhead)
    split_with_duckdb(temp_parquet, output_path)

    # Clean up temp file
    if temp_parquet.exists():
        temp_parquet.unlink()
        print("  Cleaned up temporary parquet file")

    # Create graph metadata
    graph_dir = output_path / "sensor_graph"
    graph_dir.mkdir(exist_ok=True)
    adj_mx, dist_matrix = create_adjacency_matrix(sensor_meta)
    np.save(graph_dir / "adj_mx.npy", adj_mx)

    sensor_ids = sensor_meta["new_id"].values
    distances_list = []
    for i in range(len(sensor_ids)):
        for j in range(len(sensor_ids)):
            distances_list.append({"from": sensor_ids[i], "to": sensor_ids[j], "distance": dist_matrix[i, j]})
    pd.DataFrame(distances_list).to_csv(graph_dir / "distances.csv", index=False)

    sensor_meta[["new_id", "Lat", "Lng", "County", "Fwy"]].rename(
        columns={"new_id": "sensor_id", "Lat": "latitude", "Lng": "longitude"}
    ).to_csv(graph_dir / "sensor_locations.csv", index=False)
    print("  Saved sensor_graph/")

    readme = f"""# LargeST-{region} Dataset
Preprocessed from LargeST benchmark (California traffic data).
Region: {region}
Year: {year}
Sensors: {len(sensor_meta)}
Compatible with METR-LA and PEMS-BAY format.
"""
    (output_path / "README.md").write_text(readme)

    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print(f"  Output: {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="GLA", choices=["GLA", "GBA", "SD"])
    parser.add_argument("--year", default="2019")
    parser.add_argument("--output", default=None)
    parser.add_argument("--raw-dir", default="data/LargeST-raw")
    args = parser.parse_args()
    main(args.region, args.year, args.output, args.raw_dir)
