#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive Drug-Target Visualization Generator

Creates a self-contained HTML file with:
- UMAP visualization of drugs and targets
- Dropdown selector to choose any drug or target by name
- Interactive display that shows connections when an item is selected
"""

import os
import pandas as pd
import json
import plotly.graph_objects as go

print("=" * 80)
print("INTERACTIVE DRUG-TARGET VISUALIZATION GENERATOR")
print("=" * 80)

# ============================================================================
# Part 1: Load the plot data
# ============================================================================
print("\n" + "=" * 80)
print("LOADING PLOT DATA")
print("=" * 80)

data_path = 'drug_target_mapping_viz/expr_plot_data.csv'
print(f"\nLoading plot data from: {data_path}")
plot_df = pd.read_csv(data_path)

print(f"\nLoaded plot data:")
print(f"  Total samples: {len(plot_df):,}")
print(f"  Compounds: {(plot_df['pert_type'] == 'compound').sum():,}")
print(f"  Targets: {(plot_df['pert_type'] == 'shRNA').sum():,}")
print(f"  Columns: {list(plot_df.columns)}")

# Parse the maps_to JSON strings
print("\nParsing connection mappings...")
plot_df['maps_to_list'] = plot_df['maps_to'].apply(json.loads)

# ============================================================================
# Part 2: Create interactive visualization with dropdown
# ============================================================================
print("\n" + "=" * 80)
print("CREATING INTERACTIVE VISUALIZATION")
print("=" * 80)

# Separate drugs and targets
drug_df = plot_df[plot_df['pert_type'] == 'compound'].copy()
target_df = plot_df[plot_df['pert_type'] == 'shRNA'].copy()

print(f"\nPreparing data...")
print(f"  Drugs: {len(drug_df):,}")
print(f"  Targets: {len(target_df):,}")

# Create the base figure
fig = go.Figure()

# Add drug scatter trace
fig.add_trace(go.Scatter(
    x=drug_df['umap_1'],
    y=drug_df['umap_2'],
    mode='markers',
    name='Compounds',
    marker=dict(
        color='#1f77b4',
        size=8,
        opacity=0.7,
        line=dict(color='white', width=0.5)
    ),
    customdata=drug_df[['id', 'name']].values,
    hovertemplate='<b>Compound: %{customdata[1]}</b><br>' +
                  'SMILES: %{customdata[0]}<br>' +
                  'UMAP: (%{x:.2f}, %{y:.2f})<br>' +
                  '<extra></extra>',
    text=drug_df['name']
))

# Add target scatter trace
fig.add_trace(go.Scatter(
    x=target_df['umap_1'],
    y=target_df['umap_2'],
    mode='markers',
    name='shRNA Targets',
    marker=dict(
        color='#ff7f0e',
        size=8,
        opacity=0.7,
        line=dict(color='white', width=0.5)
    ),
    customdata=target_df[['id', 'name']].values,
    hovertemplate='<b>Target: %{customdata[1]}</b><br>' +
                  'Ensembl: %{customdata[0]}<br>' +
                  'UMAP: (%{x:.2f}, %{y:.2f})<br>' +
                  '<extra></extra>',
    text=target_df['name']
))

# Update layout
fig.update_layout(
    title={
        'text': 'Drug-Target Expression UMAP Visualization<br>' +
                '<sub>Select a drug or target from the dropdown below to show connections</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 16}
    },
    xaxis_title='UMAP Dimension 1',
    yaxis_title='UMAP Dimension 2',
    width=1200,
    height=800,
    hovermode='closest',
    template='plotly_white',
    legend=dict(
        title='Perturbation Type',
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99,
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='gray',
        borderwidth=1
    ),
    margin=dict(t=120)  # Extra top margin for dropdown
)

print("  Base visualization created")

# ============================================================================
# Part 3: Create HTML with dropdown and JavaScript
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING INTERACTIVE HTML")
print("=" * 80)

# Create mapping data for JavaScript
id_to_coords = {}
id_to_name = {}
id_to_type = {}
id_to_connections = {}

for _, row in plot_df.iterrows():
    row_id = row['id']
    id_to_coords[row_id] = [row['umap_1'], row['umap_2']]
    id_to_name[row_id] = row['name']
    id_to_type[row_id] = row['pert_type']
    id_to_connections[row_id] = row['maps_to_list']

# Create sorted options list for dropdown
options = []
for _, row in drug_df.iterrows():
    n_connections = len(row['maps_to_list'])
    options.append({
        'name': row['name'],
        'id': row['id'],
        'type': 'compound',
        'label': f"[Drug] {row['name']} ({n_connections} targets)",
        'connections': n_connections
    })

for _, row in target_df.iterrows():
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
# This ensures proper data structure
drug_coords_list = drug_df[['umap_1', 'umap_2', 'id', 'name']].values.tolist()
target_coords_list = target_df[['umap_1', 'umap_2', 'id', 'name']].values.tolist()

# Generate the HTML
html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Drug-Target Expression Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
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
        #info ul {
            margin: 5px 0;
            padding-left: 20px;
        }
        #info li {
            margin: 3px 0;
            font-size: 14px;
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
        #plot {
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Drug-Target Expression Visualization</h1>

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

        <div id="plot"></div>
    </div>

    <script>
        // Data from Python - manually built arrays
        const drugData = """ + json.dumps(drug_coords_list) + """;
        const targetData = """ + json.dumps(target_coords_list) + """;

        const idToCoords = """ + json.dumps(id_to_coords) + """;
        const idToName = """ + json.dumps(id_to_name) + """;
        const idToType = """ + json.dumps(id_to_type) + """;
        const idToConnections = """ + json.dumps(id_to_connections) + """;

        // Build initial plot data
        const drugTrace = {
            x: drugData.map(d => d[0]),
            y: drugData.map(d => d[1]),
            mode: 'markers',
            type: 'scatter',
            name: 'Compounds',
            marker: {
                color: '#1f77b4',
                size: 8,
                opacity: 0.7,
                line: {color: 'white', width: 0.5}
            },
            customdata: drugData.map(d => [d[2], d[3]]),
            hovertemplate: '<b>Compound: %{customdata[1]}</b><br>' +
                          'SMILES: %{customdata[0]}<br>' +
                          'UMAP: (%{x:.2f}, %{y:.2f})<br>' +
                          '<extra></extra>'
        };

        const targetTrace = {
            x: targetData.map(d => d[0]),
            y: targetData.map(d => d[1]),
            mode: 'markers',
            type: 'scatter',
            name: 'shRNA Targets',
            marker: {
                color: '#ff7f0e',
                size: 8,
                opacity: 0.7,
                line: {color: 'white', width: 0.5}
            },
            customdata: targetData.map(d => [d[2], d[3]]),
            hovertemplate: '<b>Target: %{customdata[1]}</b><br>' +
                          'Ensembl: %{customdata[0]}<br>' +
                          'UMAP: (%{x:.2f}, %{y:.2f})<br>' +
                          '<extra></extra>'
        };

        const initialData = [drugTrace, targetTrace];

        const initialLayout = {
            title: {
                text: 'Drug-Target Expression UMAP Visualization<br>' +
                      '<sub>Select a drug or target from the dropdown to show connections</sub>',
                x: 0.5,
                xanchor: 'center',
                font: {size: 16}
            },
            xaxis: {title: 'UMAP Dimension 1'},
            yaxis: {title: 'UMAP Dimension 2'},
            width: 1200,
            height: 800,
            hovermode: 'closest',
            template: 'plotly_white',
            legend: {
                title: {text: 'Perturbation Type'},
                yanchor: 'top',
                y: 1,
                xanchor: 'left',
                x: 1.02,
                bgcolor: 'rgba(255, 255, 255, 0.9)',
                bordercolor: 'gray',
                borderwidth: 1
            }
        };

        // Create initial plot
        Plotly.newPlot('plot', initialData, initialLayout);

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

            // Reset plot to initial state
            Plotly.react('plot', initialData, initialLayout);
        }

        function updateVisualization(selectedId) {
            const selectedName = idToName[selectedId];
            const selectedType = idToType[selectedId];
            const connections = idToConnections[selectedId];
            const selectedCoords = idToCoords[selectedId];

            // Update info box
            const infoDiv = document.getElementById('info');
            const typeLabel = selectedType === 'compound' ? 'Drug' : 'Target';
            const connectionLabel = selectedType === 'compound' ? 'targets' : 'drugs';

            let infoHtml = `
                <h3>Selected ${typeLabel}: ${selectedName}</h3>
                <p><strong>ID:</strong> ${selectedId}</p>
                <p><strong>UMAP Coordinates:</strong> (${selectedCoords[0].toFixed(2)}, ${selectedCoords[1].toFixed(2)})</p>
                <p><strong>Connections:</strong> ${connections.length} ${connectionLabel}</p>
            `;

            if (connections.length > 0) {
                infoHtml += '<ul>';
                connections.forEach(connId => {
                    const connName = idToName[connId];
                    infoHtml += `<li>${connName} (${connId.substring(0, 30)}...)</li>`;
                });
                infoHtml += '</ul>';
            }

            infoDiv.innerHTML = infoHtml;
            infoDiv.style.display = 'block';

            // Add connection lines (will appear behind scatter points)
            const lineTraces = [];
            connections.forEach(connId => {
                if (idToCoords[connId]) {
                    const connCoords = idToCoords[connId];
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

            // Combine line traces with scatter traces (lines first so they appear behind)
            const allData = [...lineTraces, drugTrace, targetTrace];

            // Update title
            const newLayout = {
                ...initialLayout,
                title: {
                    ...initialLayout.title,
                    text: `Drug-Target Connections for: ${selectedName}<br><sub>${connections.length} connection${connections.length !== 1 ? 's' : ''} shown</sub>`
                }
            };

            // Update plot
            Plotly.react('plot', allData, newLayout);
        }
    </script>
</body>
</html>
"""

output_path = 'drug_target_mapping_viz/interactive_visualization.html'
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
print(f"\nInteractive visualization created: {output_path}")
print(f"\nFeatures:")
print(f"  - Dropdown selector with all {len(options)} drugs and targets")
print(f"  - Click any option to show its connections")
print(f"  - Red lines connect known drug-target pairs")
print(f"  - Hover over points for detailed information")
print(f"  - Reset button to clear selection")
print(f"\nTo view:")
print(f"  1. Open the HTML file in any web browser")
print(f"  2. Or run: open {output_path}")
print(f"\n" + "=" * 80)
