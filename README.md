# Virtual Memory Simulator

A Python-based tool designed to simulate and analyze operating system memory management through three primary page replacement algorithms. This project provides both a command-line interface and a web-based dashboard to visualize how memory frames handle page requests.

## Overview

This simulator implements FIFO (First In First Out), LRU (Least Recently Used), and Optimal page replacement algorithms. It tracks page faults, calculates hit rates, and provides a step-by-step breakdown of memory state transitions for educational and technical analysis.

## How It Works

The simulation follows a structured data flow to ensure accurate comparison across different logic models:

### 1. Input Processing

The user provides two main parameters:

* **Reference String**: A sequence of integers representing page requests made by the CPU.
* **Frame Count**: The number of available slots in physical memory (RAM).

### 2. Algorithm Execution Logic

Each algorithm processes the reference string differently when a "page fault" occurs (when a requested page is not already in a frame):

* **FIFO (First-In-First-Out)**: This algorithm uses a queue-based approach. It tracks the order in which pages enter memory and replaces the oldest page regardless of how often it is used.
* **LRU (Least Recently Used)**: This is implemented using an `OrderedDict` to track the recency of access. When a page is accessed, it is moved to the "most recent" end. When a replacement is needed, the page at the "least recent" end is removed.
* **Optimal (Belady's Algorithm)**: This logic requires looking ahead into the future of the reference string. It identifies which page currently in memory will not be used for the longest period of time and selects it for replacement, providing the lowest possible page fault rate.

### 3. Analysis and Visualization

* **Step-by-Step Tracking**: For every request in the string, the system records whether a fault occurred and the resulting state of the memory frames.
* **Efficiency Metrics**: The system calculates the total page faults, hit rates, and fault rates for each algorithm.
* **Comparative Data**: The results are aggregated into tables and interactive bar charts to show performance differences side-by-side.

## System Architecture

### Core Components

* **Simulation Engine**: Managed by the `PageReplacementSimulator` class, which orchestrates the independent execution of all three algorithms.
* **Web Interface**: A Streamlit-based application (`streamlit_app.py`) that uses Plotly and Pandas for interactive data exploration.
* **CLI Interface**: A terminal-based script (`memory_simulator.py`) that uses `PrettyTable` for formatted console output.

### Dependencies

* **Standard Libraries**: `collections` (for `OrderedDict`), `sys`.
* **Third-Party Libraries**: `streamlit`, `pandas`, `plotly`, `matplotlib`, `prettytable`.

## Installation and Usage

### Prerequisites

It is recommended to install the following required dependencies:

```bash
pip install streamlit pandas matplotlib plotly prettytable

```

### Running the Web Dashboard

```bash
streamlit run streamlit_app.py

```

### Running the CLI Tool

```bash
python memory_simulator.py

```

