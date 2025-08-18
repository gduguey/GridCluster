# Source Code Layout

This repository contains the implementation of the **aggregation pipeline** and the **generation and transmission expansion planning (GTEP) model**. Both modules are designed to work together but can also be used independently.

## Structure

```
src/
├── aggregation/   # Preprocessing and spatiotemporal aggregation pipeline
├── gtep/          # Core GTEP optimization model and deaggregation
```

### `aggregation/`

This folder implements the pipeline for **reducing spatial and temporal complexity** of input data before running the planning model.

* Supports both a **fine network** (385 nodes) and a **coarse network** (17 zones, derived from the fine network).
* Includes utilities for data loading, correlation-based clustering, and representative day selection.
* Configuration is handled through `settings.py`.

See [aggregation/README.md](aggregation/README.md) for details.

### `gtep/`

This folder implements the **Generation and Transmission Expansion Planning (GTEP)** model.

* Includes model definitions, typed dataclasses, utilities, and a pipeline for running single or multi-stage experiments.
* The model supports investment in generation, storage, and transmission, and is designed for use with aggregated inputs.

See [gtep/README.md](gtep/README.md) for details.

## Usage

Typical workflow:

1. Run the **aggregation pipeline** (`src/aggregation/`) to prepare inputs from either the fine or coarse network.
2. Pass the aggregated inputs to the **GTEP model** (`src/gtep/`) for optimization experiments.
