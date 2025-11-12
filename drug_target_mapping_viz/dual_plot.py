#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive Dual Drug-Target Visualization Generator

Creates a self-contained HTML file with:
- Two side-by-side UMAP visualizations (networks and expression)
- Shared dropdown selector to choose any drug or target by name
- Interactive display that shows connections on both plots when an item is selected
"""

import os
import pandas as pd
import json
import plotly.graph_objects as go
import argparse

print("=" * 80)
print("INTERACTIVE DUAL DRUG-TARGET VISUALIZATION GENERATOR")
print("=" * 80)

# ============================================================================
# Parse command-line arguments
# ============================================================================
parser = argparse.ArgumentParser(
    description='Generate a dual interactive HTML visualization from two drug-target mapping CSV files.'
)
parser.add_argument(
    'networks_csv',
    type=str,
    help='Path to the networks CSV file containing plot data'
)
parser.add_argument(
    'expression_csv',
    type=str,
    help='Path to the expression CSV file containing plot data'
)
args = parser.parse_args()

# ============================================================================
# Part 1: Load the plot data for both datasets
# ============================================================================
print("\n" + "=" * 80)
print("LOADING PLOT DATA")
print("=" * 80)

print(f"\nLoading networks data from: {args.networks_csv}")
networks_df = pd.read_csv(args.networks_csv)
print(f"  Total samples: {len(networks_df):,}")
print(f"  Compounds: {(networks_df['pert_type'] == 'compound').sum():,}")
print(f"  Targets: {(networks_df['pert_type'] == 'shRNA').sum():,}")

print(f"\nLoading expression data from: {args.expression_csv}")
expression_df = pd.read_csv(args.expression_csv)
print(f"  Total samples: {len(expression_df):,}")
print(f"  Compounds: {(expression_df['pert_type'] == 'compound').sum():,}")
print(f"  Targets: {(expression_df['pert_type'] == 'shRNA').sum():,}")

# Parse the maps_to JSON strings
print("\nParsing connection mappings...")
networks_df['maps_to_list'] = networks_df['maps_to'].apply(json.loads)
expression_df['maps_to_list'] = expression_df['maps_to'].apply(json.loads)

# ============================================================================
# Part 2: Prepare data for both visualizations
# ============================================================================
print("\n" + "=" * 80)
print("PREPARING VISUALIZATION DATA")
print("=" * 80)

# Separate drugs and targets for networks
networks_drug_df = networks_df[networks_df['pert_type'] == 'compound'].copy()
networks_target_df = networks_df[networks_df['pert_type'] == 'shRNA'].copy()

# Separate drugs and targets for expression
expression_drug_df = expression_df[expression_df['pert_type'] == 'compound'].copy()
expression_target_df = expression_df[expression_df['pert_type'] == 'shRNA'].copy()

print(f"\nNetworks - Drugs: {len(networks_drug_df):,}, Targets: {len(networks_target_df):,}")
print(f"Expression - Drugs: {len(expression_drug_df):,}, Targets: {len(expression_target_df):,}")

# Create mapping data for JavaScript - Networks
networks_id_to_coords = {}
networks_id_to_name = {}
networks_id_to_type = {}
networks_id_to_connections = {}

for _, row in networks_df.iterrows():
    row_id = row['id']
    networks_id_to_coords[row_id] = [row['umap_1'], row['umap_2']]
    networks_id_to_name[row_id] = row['name']
    networks_id_to_type[row_id] = row['pert_type']
    networks_id_to_connections[row_id] = row['maps_to_list']

# Create mapping data for JavaScript - Expression
expression_id_to_coords = {}
expression_id_to_name = {}
expression_id_to_type = {}
expression_id_to_connections = {}

for _, row in expression_df.iterrows():
    row_id = row['id']
    expression_id_to_coords[row_id] = [row['umap_1'], row['umap_2']]
    expression_id_to_name[row_id] = row['name']
    expression_id_to_type[row_id] = row['pert_type']
    expression_id_to_connections[row_id] = row['maps_to_list']

# Create sorted options list for dropdown (use networks as primary source)
options = []
for _, row in networks_drug_df.iterrows():
    n_connections = len(row['maps_to_list'])
    options.append({
        'name': row['name'],
        'id': row['id'],
        'type': 'compound',
        'label': f"[Drug] {row['name']} ({n_connections} targets)",
        'connections': n_connections
    })

for _, row in networks_target_df.iterrows():
    n_connections = len(row['maps_to_list'])
    options.append({
        'name': row['name'],
        'id': row['id'],
        'type': 'target',
        'label': f"[Target] {row['name']} ({n_connections} drugs)",
        'connections': n_connections
    })

# Sort options by name
options.sort(key=lambda x: x['name'].lower())

print(f"\nPrepared {len(options)} dropdown options")

# Manually build the JavaScript arrays from the pandas dataframes
networks_drug_coords_list = networks_drug_df[['umap_1', 'umap_2', 'id', 'name']].values.tolist()
networks_target_coords_list = networks_target_df[['umap_1', 'umap_2', 'id', 'name']].values.tolist()

expression_drug_coords_list = expression_drug_df[['umap_1', 'umap_2', 'id', 'name']].values.tolist()
expression_target_coords_list = expression_target_df[['umap_1', 'umap_2', 'id', 'name']].values.tolist()

# ============================================================================
# Part 3: Generate HTML with dual plots and JavaScript
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING INTERACTIVE HTML")
print("=" * 80)

# Generate the HTML
html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Dual Drug-Target Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 2400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .controls {
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .control-group {
            margin-bottom: 10px;
        }
        label {
            font-weight: bold;
            margin-right: 10px;
            color: #555;
        }
        select {
            padding: 8px 12px;
            font-size: 14px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: white;
            min-width: 500px;
        }
        #info {
            margin: 15px 0;
            padding: 15px;
            background-color: #e8f4f8;
            border-left: 4px solid #2196F3;
            border-radius: 4px;
            display: none;
        }
        #info h3 {
            margin-top: 0;
            color: #1976D2;
        }
        .reset-btn {
            padding: 8px 16px;
            margin-left: 10px;
            background-color: #f44336;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .reset-btn:hover {
            background-color: #d32f2f;
        }
        .plots-container {
            display: flex;
            gap: 20px;
            margin-top: 20px;
        }
        .plot-wrapper {
            flex: 1;
        }
        .plot-title {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Dual Drug-Target Visualization</h1>

        <div class="controls">
            <div class="control-group">
                <label for="selector">Select a drug or target:</label>
                <select id="selector">
                    <option value="">-- Select to show connections --</option>
""" + '\n'.join([f'                    <option value="{opt["id"]}">{opt["label"]}</option>' for opt in options]) + """
                </select>
                <button class="reset-btn" onclick="resetSelection()">Reset</button>
            </div>
        </div>

        <div id="info"></div>

        <div class="plots-container">
            <div class="plot-wrapper">
                <div class="plot-title">Networks</div>
                <div id="plot-networks"></div>
            </div>
            <div class="plot-wrapper">
                <div class="plot-title">Expression</div>
                <div id="plot-expression"></div>
            </div>
        </div>
    </div>

    <script>
        // Networks data
        const networksData = {
            drugData: """ + json.dumps(networks_drug_coords_list) + """,
            targetData: """ + json.dumps(networks_target_coords_list) + """,
            idToCoords: """ + json.dumps(networks_id_to_coords) + """,
            idToName: """ + json.dumps(networks_id_to_name) + """,
            idToType: """ + json.dumps(networks_id_to_type) + """,
            idToConnections: """ + json.dumps(networks_id_to_connections) + """
        };

        // Expression data
        const expressionData = {
            drugData: """ + json.dumps(expression_drug_coords_list) + """,
            targetData: """ + json.dumps(expression_target_coords_list) + """,
            idToCoords: """ + json.dumps(expression_id_to_coords) + """,
            idToName: """ + json.dumps(expression_id_to_name) + """,
            idToType: """ + json.dumps(expression_id_to_type) + """,
            idToConnections: """ + json.dumps(expression_id_to_connections) + """
        };

        // Function to create trace
        function createTraces(data) {
            const drugTrace = {
                x: data.drugData.map(d => d[0]),
                y: data.drugData.map(d => d[1]),
                mode: 'markers',
                type: 'scatter',
                name: 'Chemical',
                marker: {
                    color: '#1f77b4',
                    size: 8,
                    opacity: 0.7,
                    line: {color: 'white', width: 0.5}
                },
                customdata: data.drugData.map(d => [d[2], d[3]]),
                hovertemplate: '<b>Compound: %{customdata[1]}</b><br>' +
                              'SMILES: %{customdata[0]}<br>' +
                              'UMAP: (%{x:.2f}, %{y:.2f})<br>' +
                              '<extra></extra>'
            };

            const targetTrace = {
                x: data.targetData.map(d => d[0]),
                y: data.targetData.map(d => d[1]),
                mode: 'markers',
                type: 'scatter',
                name: 'shRNA Knockdown',
                marker: {
                    color: '#ff7f0e',
                    size: 8,
                    opacity: 0.7,
                    line: {color: 'white', width: 0.5}
                },
                customdata: data.targetData.map(d => [d[2], d[3]]),
                hovertemplate: '<b>Target: %{customdata[1]}</b><br>' +
                              'Ensembl: %{customdata[0]}<br>' +
                              'UMAP: (%{x:.2f}, %{y:.2f})<br>' +
                              '<extra></extra>'
            };

            return { drugTrace, targetTrace };
        }

        // Create traces for both datasets
        const networksTraces = createTraces(networksData);
        const expressionTraces = createTraces(expressionData);

        // Common layout configuration
        const commonLayout = {
            xaxis: {
                title: {text: 'UMAP 1', font: {size: 18}},
                showticklabels: false,
                showgrid: false,
                zeroline: false,
                showline: true,
                linecolor: 'black',
                mirror: false,
                ticks: ''
            },
            yaxis: {
                title: {text: 'UMAP 2', font: {size: 18}},
                showticklabels: false,
                showgrid: false,
                zeroline: false,
                showline: true,
                linecolor: 'black',
                mirror: false,
                ticks: ''
            },
            width: 750,
            height: 600,
            hovermode: 'closest',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            legend: {
                title: {text: '  Perturbation Type  ', font: {size: 16}},
                font: {size: 14},
                yanchor: 'top',
                y: 1,
                xanchor: 'left',
                x: 1.02,
                bgcolor: 'rgba(255, 255, 255, 0.9)',
                bordercolor: 'gray',
                borderwidth: 1
            }
        };

        // Create initial plots
        Plotly.newPlot('plot-networks', [networksTraces.drugTrace, networksTraces.targetTrace], commonLayout);
        Plotly.newPlot('plot-expression', [expressionTraces.drugTrace, expressionTraces.targetTrace], commonLayout);

        // Handle selection change
        document.getElementById('selector').addEventListener('change', function() {
            const selectedId = this.value;

            if (!selectedId) {
                resetSelection();
                return;
            }

            updateVisualization(selectedId);
        });

        function resetSelection() {
            // Reset dropdown
            document.getElementById('selector').value = '';

            // Hide info box
            document.getElementById('info').style.display = 'none';

            // Reset both plots to initial state
            Plotly.react('plot-networks', [networksTraces.drugTrace, networksTraces.targetTrace], commonLayout);
            Plotly.react('plot-expression', [expressionTraces.drugTrace, expressionTraces.targetTrace], commonLayout);
        }

        function updateVisualization(selectedId) {
            // Get info from networks data (primary)
            const selectedName = networksData.idToName[selectedId];
            const selectedType = networksData.idToType[selectedId];
            const typeLabel = selectedType === 'compound' ? 'Drug' : 'Target';
            const connectionLabel = selectedType === 'compound' ? 'targets' : 'drugs';

            // Update info box
            const infoDiv = document.getElementById('info');
            let infoHtml = `<h3>Selected ${typeLabel}: ${selectedName}</h3>`;
            infoDiv.innerHTML = infoHtml;
            infoDiv.style.display = 'block';

            // Update networks plot
            updatePlot('plot-networks', networksData, networksTraces, selectedId);

            // Update expression plot
            updatePlot('plot-expression', expressionData, expressionTraces, selectedId);
        }

        function updatePlot(plotId, data, traces, selectedId) {
            const connections = data.idToConnections[selectedId] || [];
            const selectedCoords = data.idToCoords[selectedId];

            if (!selectedCoords) {
                // Item not in this dataset, just show base plot
                Plotly.react(plotId, [traces.drugTrace, traces.targetTrace], commonLayout);
                return;
            }

            // Add connection lines
            const lineTraces = [];
            connections.forEach(connId => {
                if (data.idToCoords[connId]) {
                    const connCoords = data.idToCoords[connId];
                    lineTraces.push({
                        x: [selectedCoords[0], connCoords[0]],
                        y: [selectedCoords[1], connCoords[1]],
                        mode: 'lines',
                        type: 'scatter',
                        line: {
                            color: 'red',
                            width: 2
                        },
                        opacity: 0.5,
                        showlegend: false,
                        hoverinfo: 'skip'
                    });
                }
            });

            // Combine line traces with scatter traces
            const allData = [...lineTraces, traces.drugTrace, traces.targetTrace];

            // Update plot
            Plotly.react(plotId, allData, commonLayout);
        }
    </script>
</body>
</html>
"""

output_path = 'dual_visualization.html'
print(f"\nWriting HTML to: {output_path}")

with open(output_path, 'w') as f:
    f.write(html_template)

print(f"  HTML file created successfully!")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(f"\nDual interactive visualization created: {output_path}")
print(f"\nFeatures:")
print(f"  - Two side-by-side plots (Networks and Expression)")
print(f"  - Shared dropdown selector with all {len(options)} drugs and targets")
print(f"  - Selecting an item updates both plots simultaneously")
print(f"  - Red lines connect known drug-target pairs in both views")
print(f"  - Hover over points for detailed information")
print(f"  - Reset button to clear selection")
print(f"\nTo view:")
print(f"  1. Open the HTML file in any web browser")
print(f"  2. Or run: open {output_path}")
print(f"\n" + "=" * 80)
