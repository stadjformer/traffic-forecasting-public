"""Generate and push W&B report from CSV results."""

import pandas as pd
import wandb_workspaces.reports.v2 as wr

# Configuration
ENTITY = "cs224w-traffic-forecasting"
PROJECT = "traffic-forecasting"
HF_USERNAME = "witgaw"
METR_LA_CSV = "results/metr-la_results.csv"
PEMS_BAY_CSV = "results/pems-bay_results.csv"

# Model identifiers
BASELINE_MODEL = "STGFORMER_BS200_SHORT"
FINAL_MODEL = "STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16"
FULL_MODEL = "STGFORMER_CHEB_TCN_XAVIER_DOW_EXCLUDE_MISSING_K16"

# Load results from CSVs
metr_df = pd.read_csv(METR_LA_CSV)
pems_df = pd.read_csv(PEMS_BAY_CSV)


def get_metrics(df, model_name, horizon):
    """Extract metrics for a specific model and horizon."""
    row = df[df["model"] == model_name]
    if row.empty:
        return None
    row = row.iloc[0]
    # Column names have spaces, not underscores
    return {
        "mae": row[f"mae_{horizon}"],
        "rmse": row[f"rmse_{horizon}"],
        "mape": row[f"mape_{horizon}"],
    }


def generate_performance_section():
    """Generate Performance vs Internal Baseline section."""
    blocks = []

    # Add heading and baseline info
    blocks.append(wr.H2("Performance vs Internal Baseline"))
    blocks.append(
        wr.MarkdownBlock(
            f"Baseline: `{BASELINE_MODEL}`: Baseline with batch_size=200, 20 epochs, early_stop=5"
        )
    )

    # METR-LA table
    blocks.append(wr.H3("METR-LA"))
    metr_table_lines = [
        "| Horizon | Metric | Baseline | Final | Improvement |",
        "|---------|--------|----------|-------|-------------|",
    ]

    for horizon in ["15 min", "30 min", "1 hour"]:
        baseline_m = get_metrics(metr_df, BASELINE_MODEL, horizon)
        final_m = get_metrics(metr_df, FINAL_MODEL, horizon)

        if baseline_m and final_m:
            mae_imp = ((baseline_m["mae"] - final_m["mae"]) / baseline_m["mae"]) * 100
            rmse_imp = (
                (baseline_m["rmse"] - final_m["rmse"]) / baseline_m["rmse"]
            ) * 100

            metr_table_lines.append(
                f"| {horizon}  | MAE    | {baseline_m['mae']:.3f}    | {final_m['mae']:.3f} | {mae_imp:.1f}% |"
            )
            metr_table_lines.append(
                f"|         | RMSE   | {baseline_m['rmse']:.3f}    | {final_m['rmse']:.3f} | {rmse_imp:.1f}% |"
            )

    blocks.append(wr.MarkdownBlock("\n".join(metr_table_lines)))

    # PEMS-BAY table
    blocks.append(wr.H3("PEMS-BAY"))
    pems_table_lines = [
        "| Horizon | Metric | Baseline | Final | Improvement |",
        "|---------|--------|----------|-------|-------------|",
    ]

    for horizon in ["15 min", "30 min", "1 hour"]:
        baseline_m = get_metrics(pems_df, BASELINE_MODEL, horizon)
        final_m = get_metrics(pems_df, FINAL_MODEL, horizon)

        if baseline_m and final_m:
            mae_imp = ((baseline_m["mae"] - final_m["mae"]) / baseline_m["mae"]) * 100
            rmse_imp = (
                (baseline_m["rmse"] - final_m["rmse"]) / baseline_m["rmse"]
            ) * 100

            pems_table_lines.append(
                f"| {horizon}  | MAE    | {baseline_m['mae']:.3f}    | {final_m['mae']:.3f} | {mae_imp:.1f}% |"
            )
            pems_table_lines.append(
                f"|         | RMSE   | {baseline_m['rmse']:.3f}    | {final_m['rmse']:.3f} | {rmse_imp:.1f}% |"
            )

    blocks.append(wr.MarkdownBlock("\n".join(pems_table_lines)))

    return blocks


def generate_ablation_section():
    """Generate Ablation Study section."""
    blocks = []

    blocks.append(wr.H2("Ablation Study: Component Contributions"))
    blocks.append(
        wr.MarkdownBlock(
            "Impact measured as Δ from full model (positive = degradation when removed)\n\n"
            "**Note on ablations:**\n\n"
            "- **w/o graph propagation (GraphMode.NONE)**: Uses identity matrix (each node only sees itself, no message passing)\n"
            "- **w/o temporal processing (TemporalMode.NONE)**: Skips attention layers, uses sum of projected graph-propagated features"
        )
    )

    # Ablation configurations
    # Note: Sparsity ablation (K=∞) was not run as a separate experiment, so it's excluded
    ablations = [
        ("ABL_NO_DOW", "w/o DOW"),
        ("ABL_NO_EXCLUDE_MISSING", "w/o ExcludeMissing"),
        ("ABL_NO_TCN", "w/o TCN"),
        ("ABL_NO_CHEB", "w/o Cheb"),
        ("ABL_NO_GRAPH", "w/o graph propagation (GraphMode.NONE)"),
        ("ABL_NO_TEMPORAL", "w/o temporal processing (TemporalMode.NONE)"),
    ]

    # METR-LA ablation table
    blocks.append(wr.H3("METR-LA (1-hour horizon)"))
    full_m = get_metrics(metr_df, FULL_MODEL, "1 hour")

    if full_m:
        metr_abl_lines = [
            "| Component | ΔMAE | ΔRMSE |",
            "|-----------|------|-------|",
        ]

        for model_name, label in ablations:
            abl_m = get_metrics(metr_df, model_name, "1 hour")
            if abl_m:
                delta_mae = abl_m["mae"] - full_m["mae"]
                delta_rmse = abl_m["rmse"] - full_m["rmse"]
                metr_abl_lines.append(
                    f"| {label} | {delta_mae:+.3f} | {delta_rmse:+.3f} |"
                )

        blocks.append(wr.MarkdownBlock("\n".join(metr_abl_lines)))

    # PEMS-BAY ablation table
    blocks.append(wr.H3("PEMS-BAY (1-hour horizon)"))
    full_p = get_metrics(pems_df, FULL_MODEL, "1 hour")

    if full_p:
        pems_abl_lines = [
            "| Component | ΔMAE | ΔRMSE |",
            "|-----------|------|-------|",
        ]

        for model_name, label in ablations:
            abl_p = get_metrics(pems_df, model_name, "1 hour")
            if abl_p:
                delta_mae = abl_p["mae"] - full_p["mae"]
                delta_rmse = abl_p["rmse"] - full_p["rmse"]
                pems_abl_lines.append(
                    f"| {label} | {delta_mae:+.3f} | {delta_rmse:+.3f} |"
                )

        blocks.append(wr.MarkdownBlock("\n".join(pems_abl_lines)))

    return blocks


def generate_research_highlights():
    """Generate Research Direction Highlights section."""
    blocks = []

    blocks.append(wr.H2("Research Direction Highlights"))

    # Calculate baseline metrics once for all subsections
    baseline_metr = get_metrics(metr_df, BASELINE_MODEL, "1 hour")
    baseline_pems = get_metrics(pems_df, BASELINE_MODEL, "1 hour")

    # Initial experiments
    blocks.append(wr.H3("Initial (1-hour MAE)"))

    initial_experiments = [
        ("STGFORMER_BS200_SHORT", "Baseline with batch_size=200, 20 epochs"),
        ("STGFORMER_BS200_XAVIER", "Baseline + Xavier initialization"),
        ("STGFORMER_BS200_DOW", "Baseline + DOW embeddings"),
        ("STGFORMER_BS200_EXCLUDE_MISSING", "Baseline + ExcludeMissing normalization"),
    ]

    initial_lines = ["**METR-LA:**"]
    for model, desc in initial_experiments:
        m = get_metrics(metr_df, model, "1 hour")
        if m and baseline_metr:
            improvement = (
                (baseline_metr["mae"] - m["mae"]) / baseline_metr["mae"]
            ) * 100
            initial_lines.append(
                f"- `{model}`: {desc} — {m['mae']:.3f} ({improvement:.2f}% improvement over baseline)"
            )

    initial_lines.append("")
    initial_lines.append("**PEMS-BAY:**")
    for model, desc in initial_experiments:
        m = get_metrics(pems_df, model, "1 hour")
        if m and baseline_pems:
            improvement = (
                (baseline_pems["mae"] - m["mae"]) / baseline_pems["mae"]
            ) * 100
            initial_lines.append(
                f"- `{model}`: {desc} — {m['mae']:.3f} ({improvement:.2f}% improvement)"
            )

    blocks.append(wr.MarkdownBlock("\n".join(initial_lines)))

    # Add PanelGrids for Initial experiments
    runset_initial_metr = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="Initial Experiments - METR-LA",
        filters='Config("dataset") = "METR-LA" and Config("epochs") = 20',
    )

    runset_initial_pems = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="Initial Experiments - PEMS-BAY",
        filters='Config("dataset") = "PEMS-BAY" and Config("epochs") = 20',
    )

    blocks.append(
        wr.PanelGrid(
            runsets=[runset_initial_metr],
            panels=[
                wr.LinePlot(
                    title="Training Loss - METR-LA",
                    x="Step",
                    y=["train/loss"],
                    smoothing_factor=0.6,
                    layout={"w": 12, "h": 10},
                ),
                wr.LinePlot(
                    title="Validation MAE - METR-LA",
                    x="Step",
                    y=["val/mae"],
                    smoothing_factor=0.6,
                    layout={"w": 12, "h": 10},
                ),
            ],
        )
    )

    blocks.append(
        wr.PanelGrid(
            runsets=[runset_initial_pems],
            panels=[
                wr.LinePlot(
                    title="Training Loss - PEMS-BAY",
                    x="Step",
                    y=["train/loss"],
                    smoothing_factor=0.6,
                    layout={"w": 12, "h": 10},
                ),
                wr.LinePlot(
                    title="Validation MAE - PEMS-BAY",
                    x="Step",
                    y=["val/mae"],
                    smoothing_factor=0.6,
                    layout={"w": 12, "h": 10},
                ),
            ],
        )
    )

    # Pretraining experiments
    blocks.append(wr.H3("Pretraining (1-hour MAE)"))

    pretraining_experiments = [
        ("STGFORMER_BS200_SHORT", "Baseline (no pretraining)"),
        (
            "STGFORMER_PRETRAIN_IMPUTE_NORM",
            "Two-stage pretraining with imputation + normalization",
        ),
        (
            "STGFORMER_PRETRAIN_STAGE1ONLY_NORM",
            "Stage 1 pretraining only + normalization",
        ),
        (
            "STGFORMER_PRETRAIN_STAGE2ONLY_NORM",
            "Stage 2 pretraining only + normalization",
        ),
    ]

    pretraining_lines = ["**METR-LA:**"]
    for model, desc in pretraining_experiments:
        m = get_metrics(metr_df, model, "1 hour")
        if m and baseline_metr:
            improvement = (
                (baseline_metr["mae"] - m["mae"]) / baseline_metr["mae"]
            ) * 100
            pretraining_lines.append(
                f"- `{model}`: {desc} — {m['mae']:.3f} ({improvement:.2f}% improvement over baseline)"
            )

    pretraining_lines.append("")
    pretraining_lines.append("**PEMS-BAY:**")
    for model, desc in pretraining_experiments:
        m = get_metrics(pems_df, model, "1 hour")
        if m and baseline_pems:
            improvement = (
                (baseline_pems["mae"] - m["mae"]) / baseline_pems["mae"]
            ) * 100
            pretraining_lines.append(
                f"- `{model}`: {desc} — {m['mae']:.3f} ({improvement:.2f}% improvement)"
            )

    blocks.append(wr.MarkdownBlock("\n".join(pretraining_lines)))

    # Add PanelGrids for Pretraining experiments
    runset_pretraining_metr = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="Pretraining Experiments - METR-LA",
        filters='Config("dataset") = "METR-LA" and Config("epochs") = 20',
    )

    runset_pretraining_pems = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="Pretraining Experiments - PEMS-BAY",
        filters='Config("dataset") = "PEMS-BAY" and Config("epochs") = 20',
    )

    blocks.append(
        wr.PanelGrid(
            runsets=[runset_pretraining_metr],
            panels=[
                wr.LinePlot(
                    title="Training Loss - METR-LA",
                    x="Step",
                    y=["train/loss"],
                    smoothing_factor=0.6,
                    layout={"w": 12, "h": 10},
                ),
                wr.LinePlot(
                    title="Validation MAE - METR-LA",
                    x="Step",
                    y=["val/mae"],
                    smoothing_factor=0.6,
                    layout={"w": 12, "h": 10},
                ),
            ],
        )
    )

    blocks.append(
        wr.PanelGrid(
            runsets=[runset_pretraining_pems],
            panels=[
                wr.LinePlot(
                    title="Training Loss - PEMS-BAY",
                    x="Step",
                    y=["train/loss"],
                    smoothing_factor=0.6,
                    layout={"w": 12, "h": 10},
                ),
                wr.LinePlot(
                    title="Validation MAE - PEMS-BAY",
                    x="Step",
                    y=["val/mae"],
                    smoothing_factor=0.6,
                    layout={"w": 12, "h": 10},
                ),
            ],
        )
    )

    # Spatial experiments
    blocks.append(wr.H3("Spatial (1-hour MAE)"))

    spatial_experiments_metr = [
        ("STGFORMER_HYBRID_CHEB", "Hybrid graph + Chebyshev polynomial propagation"),
        ("STGFORMER_GEO_CHEB", "Geographic graph + Chebyshev polynomial propagation"),
        ("STGFORMER_CHEBYSHEV", "Chebyshev polynomial propagation"),
    ]

    spatial_experiments_pems = [
        (
            "STGFORMER_SPECTRAL_INIT",
            "Learned graph initialized from Laplacian eigenvectors",
        ),
        ("STGFORMER_HYBRID_CHEB", "Hybrid graph + Chebyshev polynomial propagation"),
        ("STGFORMER_CHEBYSHEV", "Chebyshev polynomial propagation"),
    ]

    spatial_lines = ["**METR-LA:**"]
    for model, desc in spatial_experiments_metr:
        m = get_metrics(metr_df, model, "1 hour")
        if m and baseline_metr:
            improvement = (
                (baseline_metr["mae"] - m["mae"]) / baseline_metr["mae"]
            ) * 100
            spatial_lines.append(
                f"- `{model}`: {desc} — {m['mae']:.3f} ({improvement:.2f}% improvement over baseline)"
            )

    spatial_lines.append("")
    spatial_lines.append("**PEMS-BAY:**")
    for model, desc in spatial_experiments_pems:
        m = get_metrics(pems_df, model, "1 hour")
        if m and baseline_pems:
            improvement = (
                (baseline_pems["mae"] - m["mae"]) / baseline_pems["mae"]
            ) * 100
            spatial_lines.append(
                f"- `{model}`: {desc} — {m['mae']:.3f} ({improvement:.2f}% improvement)"
            )

    blocks.append(wr.MarkdownBlock("\n".join(spatial_lines)))

    # Add PanelGrids filtering by dataset and epochs=20
    runset_spatial_metr = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="Spatial Experiments - METR-LA",
        filters='Config("dataset") = "METR-LA" and Config("epochs") = 20',
    )

    runset_spatial_pems = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="Spatial Experiments - PEMS-BAY",
        filters='Config("dataset") = "PEMS-BAY" and Config("epochs") = 20',
    )

    blocks.append(
        wr.PanelGrid(
            runsets=[runset_spatial_metr],
            panels=[
                wr.LinePlot(
                    title="Training Loss - METR-LA",
                    x="Step",
                    y=["train/loss"],
                    smoothing_factor=0.6,
                    layout={"w": 12, "h": 10},
                ),
                wr.LinePlot(
                    title="Validation MAE - METR-LA",
                    x="Step",
                    y=["val/mae"],
                    smoothing_factor=0.6,
                    layout={"w": 12, "h": 10},
                ),
            ],
        )
    )

    blocks.append(
        wr.PanelGrid(
            runsets=[runset_spatial_pems],
            panels=[
                wr.LinePlot(
                    title="Training Loss - PEMS-BAY",
                    x="Step",
                    y=["train/loss"],
                    smoothing_factor=0.6,
                    layout={"w": 12, "h": 10},
                ),
                wr.LinePlot(
                    title="Validation MAE - PEMS-BAY",
                    x="Step",
                    y=["val/mae"],
                    smoothing_factor=0.6,
                    layout={"w": 12, "h": 10},
                ),
            ],
        )
    )

    # Temporal experiments
    blocks.append(wr.H3("Temporal (1-hour MAE)"))

    temporal_experiments_metr = [
        ("STGFORMER_TCN", "TCN temporal mode (causal dilated convolutions)"),
        ("STGFORMER_MAMBA", "Mamba SSM temporal mode (d_state=16, requires CUDA)"),
        ("STGFORMER_BS200_SHORT", "Baseline (Transformer)"),
        ("STGFORMER_MLP", "MLP temporal mode"),
    ]

    temporal_experiments_pems = [
        ("STGFORMER_MAMBA", "Mamba SSM temporal mode (d_state=16, requires CUDA)"),
        ("STGFORMER_BS200_SHORT", "Baseline (Transformer)"),
        (
            "STGFORMER_DEPTHWISE",
            "Depthwise separable conv temporal mode",
        ),
        ("STGFORMER_TCN", "TCN temporal mode (causal dilated convolutions)"),
    ]

    temporal_lines = ["**METR-LA:**"]
    for model, desc in temporal_experiments_metr:
        m = get_metrics(metr_df, model, "1 hour")
        if m and baseline_metr:
            improvement = (
                (baseline_metr["mae"] - m["mae"]) / baseline_metr["mae"]
            ) * 100
            temporal_lines.append(
                f"- `{model}`: {desc} — {m['mae']:.3f} ({improvement:.2f}% improvement)"
            )

    temporal_lines.append("")
    temporal_lines.append("**PEMS-BAY:**")
    for model, desc in temporal_experiments_pems:
        m = get_metrics(pems_df, model, "1 hour")
        if m and baseline_pems:
            improvement = (
                (baseline_pems["mae"] - m["mae"]) / baseline_pems["mae"]
            ) * 100
            temporal_lines.append(
                f"- `{model}`: {desc} — {m['mae']:.3f} ({improvement:.2f}% improvement)"
            )

    blocks.append(wr.MarkdownBlock("\n".join(temporal_lines)))

    # Add PanelGrids for Temporal - filtering by dataset and epochs=20
    runset_temporal_metr = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="Temporal Experiments - METR-LA",
        filters='Config("dataset") = "METR-LA" and Config("epochs") = 20',
    )

    runset_temporal_pems = wr.Runset(
        entity=ENTITY,
        project=PROJECT,
        name="Temporal Experiments - PEMS-BAY",
        filters='Config("dataset") = "PEMS-BAY" and Config("epochs") = 20',
    )

    blocks.append(
        wr.PanelGrid(
            runsets=[runset_temporal_metr],
            panels=[
                wr.LinePlot(
                    title="Training Loss - METR-LA",
                    x="Step",
                    y=["train/loss"],
                    smoothing_factor=0.6,
                    layout={"w": 12, "h": 10},
                ),
                wr.LinePlot(
                    title="Validation MAE - METR-LA",
                    x="Step",
                    y=["val/mae"],
                    smoothing_factor=0.6,
                    layout={"w": 12, "h": 10},
                ),
            ],
        )
    )

    blocks.append(
        wr.PanelGrid(
            runsets=[runset_temporal_pems],
            panels=[
                wr.LinePlot(
                    title="Training Loss - PEMS-BAY",
                    x="Step",
                    y=["train/loss"],
                    smoothing_factor=0.6,
                    layout={"w": 12, "h": 10},
                ),
                wr.LinePlot(
                    title="Validation MAE - PEMS-BAY",
                    x="Step",
                    y=["val/mae"],
                    smoothing_factor=0.6,
                    layout={"w": 12, "h": 10},
                ),
            ],
        )
    )

    return blocks


def generate_key_takeaways():
    """Generate Key Takeaways section."""
    blocks = []

    blocks.append(wr.H2("Key Takeaways"))
    blocks.append(
        wr.MarkdownBlock(
            "1. **Chebyshev propagation** is the most critical component for both datasets\n"
            "2. **TCN temporal mode** consistently outperforms Transformer attention\n"
            "3. **ExcludeMissing normalization** essential for METR-LA (sparse data), minimal impact on PEMS-BAY\n"
            "4. **DOW embeddings** provide consistent small improvements across both datasets\n"
            "5. **Graph sparsity (K=16)** has minimal impact vs unlimited neighbors\n"
            "6. **Pretraining** showed limited benefits for traffic forecasting"
        )
    )

    return blocks


def generate_appendix():
    """Generate HuggingFace Repository Appendix."""
    blocks = []

    blocks.append(wr.MarkdownBlock("---"))
    blocks.append(wr.H2("Appendix: HuggingFace Repository Name Reference"))

    # Dataset URLs
    metr_la_url = f"https://huggingface.co/datasets/{HF_USERNAME}/METR-LA"
    pems_bay_url = f"https://huggingface.co/datasets/{HF_USERNAME}/PEMS-BAY"
    final_metr_url = f"https://huggingface.co/{HF_USERNAME}/STGFORMER_FINAL_METR-LA"
    final_pems_url = f"https://huggingface.co/{HF_USERNAME}/STGFORMER_FINAL_PEMS-BAY"

    # Ablation model names (these match what's in the CSV)
    ablation_links = [
        ("ABL_NO_DOW", "Remove DOW embeddings from final model"),
        (
            "ABL_NO_EXCLUDE_MISSING",
            "Remove ExcludeMissing normalization from final model",
        ),
        ("ABL_NO_TCN", "Replace TCN with standard Transformer in final model"),
        ("ABL_NO_CHEB", "Replace Chebyshev with standard graph convolution"),
        ("ABL_NO_GRAPH", "Remove graph propagation (GraphMode.NONE)"),
        ("ABL_NO_TEMPORAL", "Remove temporal processing (TemporalMode.NONE)"),
    ]

    appendix_content = [
        "All experiments are logged to W&B with their HF repository prefix as the run name. Below is a mapping of experiment names to their HF repo prefixes and descriptions:\n",
        "**Datasets:**",
        f"- [METR-LA]({metr_la_url}): Los Angeles traffic dataset",
        f"- [PEMS-BAY]({pems_bay_url}): Bay Area traffic dataset\n",
        "**Final Model (100 epochs):**",
        f"- [STGFORMER_FINAL]({final_metr_url}): Final architecture trained for 100 epochs ([METR-LA]({final_metr_url}), [PEMS-BAY]({final_pems_url}))\n",
        "**Initial Experiments:**",
    ]

    # Add initial experiment links
    initial_links = [
        ("STGFORMER_BS200_SHORT", "Baseline with batch_size=200, 20 epochs"),
        ("STGFORMER_BS200_XAVIER", "Baseline + Xavier initialization"),
        ("STGFORMER_BS200_DOW", "Baseline + DOW embeddings"),
        ("STGFORMER_BS200_EXCLUDE_MISSING", "Baseline + ExcludeMissing normalization"),
    ]

    for prefix, description in initial_links:
        metr_url = f"https://huggingface.co/{HF_USERNAME}/{prefix}_METR-LA"
        pems_url = f"https://huggingface.co/{HF_USERNAME}/{prefix}_PEMS-BAY"
        appendix_content.append(
            f"- [`{prefix}`]({metr_url}): {description} ([METR-LA]({metr_url}), [PEMS-BAY]({pems_url}))"
        )

    appendix_content.append("\n**Pretraining Experiments:**")

    # Add pretraining experiment links
    pretraining_links = [
        (
            "STGFORMER_PRETRAIN_IMPUTE_NORM",
            "Two-stage pretraining with imputation + normalization",
        ),
        (
            "STGFORMER_PRETRAIN_STAGE1ONLY_NORM",
            "Stage 1 pretraining only + normalization",
        ),
        (
            "STGFORMER_PRETRAIN_STAGE2ONLY_NORM",
            "Stage 2 pretraining only + normalization",
        ),
    ]

    for prefix, description in pretraining_links:
        metr_url = f"https://huggingface.co/{HF_USERNAME}/{prefix}_METR-LA"
        pems_url = f"https://huggingface.co/{HF_USERNAME}/{prefix}_PEMS-BAY"
        appendix_content.append(
            f"- [`{prefix}`]({metr_url}): {description} ([METR-LA]({metr_url}), [PEMS-BAY]({pems_url}))"
        )

    appendix_content.append("\n**Spatial Experiments:**")

    # Add spatial experiment links
    spatial_links = [
        ("STGFORMER_HYBRID_CHEB", "Hybrid graph + Chebyshev polynomial propagation"),
        ("STGFORMER_GEO_CHEB", "Geographic graph + Chebyshev polynomial propagation"),
        ("STGFORMER_CHEBYSHEV", "Chebyshev polynomial propagation"),
        (
            "STGFORMER_SPECTRAL_INIT",
            "Learned graph initialized from Laplacian eigenvectors",
        ),
    ]

    for prefix, description in spatial_links:
        metr_url = f"https://huggingface.co/{HF_USERNAME}/{prefix}_METR-LA"
        pems_url = f"https://huggingface.co/{HF_USERNAME}/{prefix}_PEMS-BAY"
        appendix_content.append(
            f"- [`{prefix}`]({metr_url}): {description} ([METR-LA]({metr_url}), [PEMS-BAY]({pems_url}))"
        )

    appendix_content.append("\n**Temporal Experiments:**")

    # Add temporal experiment links
    temporal_links = [
        ("STGFORMER_TCN", "TCN temporal mode (causal dilated convolutions)"),
        ("STGFORMER_MAMBA", "Mamba SSM temporal mode (d_state=16, requires CUDA)"),
        ("STGFORMER_MLP", "MLP temporal mode"),
        ("STGFORMER_DEPTHWISE", "Depthwise separable conv temporal mode"),
    ]

    for prefix, description in temporal_links:
        metr_url = f"https://huggingface.co/{HF_USERNAME}/{prefix}_METR-LA"
        pems_url = f"https://huggingface.co/{HF_USERNAME}/{prefix}_PEMS-BAY"
        appendix_content.append(
            f"- [`{prefix}`]({metr_url}): {description} ([METR-LA]({metr_url}), [PEMS-BAY]({pems_url}))"
        )

    appendix_content.append("\n**Ablation Studies:**")

    # Add ablation links with hardcoded prefixes (since config files don't exist for ablations)
    for prefix, description in ablation_links:
        metr_url = f"https://huggingface.co/{HF_USERNAME}/{prefix}_METR-LA"
        pems_url = f"https://huggingface.co/{HF_USERNAME}/{prefix}_PEMS-BAY"
        appendix_content.append(
            f"- [`{prefix}`]({metr_url}): {description} ([METR-LA]({metr_url}), [PEMS-BAY]({pems_url}))"
        )

    blocks.append(wr.MarkdownBlock("\n".join(appendix_content)))

    return blocks


# Build report
print("Generating W&B report from CSV results...")
print(f"Entity: {ENTITY}")
print(f"Project: {PROJECT}")
print(f"METR-LA CSV: {METR_LA_CSV}")
print(f"PEMS-BAY CSV: {PEMS_BAY_CSV}\n")

report = wr.Report(
    entity=ENTITY,
    project=PROJECT,
    title="STAdjFormer: Experimental Results Summary",
    description="Comprehensive analysis of STAdjFormer architecture improvements, ablation studies, and performance comparisons",
)

# Build all blocks
blocks = [wr.TableOfContents()]

# Title
blocks.append(wr.H1("STAdjFormer: Experimental Results Summary"))

# Final Architecture
blocks.append(wr.H2("Final Architecture"))
blocks.append(
    wr.MarkdownBlock(
        f"`{FINAL_MODEL}`: Chebyshev+TCN+Xavier+DOW excluding missing values from data normalization and sparsity K=16"
    )
)

# Add all sections
blocks.extend(generate_performance_section())
blocks.extend(generate_ablation_section())
blocks.extend(generate_research_highlights())
blocks.extend(generate_key_takeaways())
blocks.extend(generate_appendix())

report.blocks = blocks

# Save report
print("\nPushing report to W&B with PanelGrids (filtered by dataset only)...")
try:
    url = report.save()
    print("\n✓ Report created successfully!")
    print(f"Direct URL: {url}")
    print(f"Reports page: https://wandb.ai/{ENTITY}/{PROJECT}/reports")
except Exception as e:
    print(f"\n✗ Error creating report: {e}")
    raise
