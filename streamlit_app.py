# CORRECTED STREAMLIT DASHBOARD (app.py)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="SA Crime Analytics", page_icon="🔍", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('SouthAfricaCrimeStats_v2.csv')
    year_columns = df.columns[3:]
    df_long = pd.melt(df, 
                     id_vars=['Province', 'Station', 'Category'], 
                     value_vars=year_columns,
                     var_name='Year', 
                     value_name='Incidents')
    df_long['Year'] = pd.to_datetime(df_long['Year'].str.split('-').str[0])
    return df, df_long

# Load data
df, df_long = load_data()

# Title
st.title("🔍 South Africa Crime Analytics Dashboard")
st.markdown("Machine Learning-driven crime hotspot classification and forecasting")

# Sidebar
st.sidebar.header("📊 Filters")

# Province filter
selected_province = st.sidebar.selectbox(
    "Select Province", 
    options=['All'] + df['Province'].unique().tolist()
)

# Crime category filter
selected_category = st.sidebar.selectbox(
    "Select Crime Category",
    options=['All'] + df['Category'].unique().tolist()
)

# Filter data
filtered_data = df_long.copy()
if selected_province != 'All':
    filtered_data = filtered_data[filtered_data['Province'] == selected_province]
if selected_category != 'All':
    filtered_data = filtered_data[filtered_data['Category'] == selected_category]

# Main dashboard
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Crime Trends Over Time")
    
    if not filtered_data.empty:
        # Aggregate by year
        yearly_data = filtered_data.groupby('Year')['Incidents'].sum().reset_index()
        
        fig = px.line(yearly_data, x='Year', y='Incidents',
                     title=f'Crime Trends: {selected_province} - {selected_category}')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for selected filters")

with col2:
    st.subheader("🏙️ Top Police Stations")
    
    if not filtered_data.empty:
        station_totals = filtered_data.groupby('Station')['Incidents'].sum().nlargest(10)
        
        fig = px.bar(x=station_totals.values, y=station_totals.index,
                    orientation='h', 
                    title=f"Top 10 Stations: {selected_province} - {selected_category}",
                    labels={'x': 'Total Incidents', 'y': 'Station'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select different filters to see data")

# Hotspot Analysis
st.subheader("🎯 Crime Hotspot Analysis")

# Simple hotspot calculation
hotspot_threshold = df_long.groupby(['Station', 'Category'])['Incidents'].sum().quantile(0.75)
hotspots = df_long.groupby(['Station', 'Category'])['Incidents'].sum().reset_index()
hotspots['Is_Hotspot'] = (hotspots['Incidents'] >= hotspot_threshold).astype(int)

col3, col4, col5 = st.columns(3)

with col3:
    st.metric("🔴 Hotspot Threshold", f"{hotspot_threshold:.0f}+ incidents")

with col4:
    total_hotspots = hotspots['Is_Hotspot'].sum()
    st.metric("📍 Hotspots Identified", f"{total_hotspots}")

with col5:
    hotspot_percentage = (hotspots['Is_Hotspot'].mean() * 100)
    st.metric("📊 Hotspot Percentage", f"{hotspot_percentage:.1f}%")

# Top hotspots
st.subheader("🔥 Top Crime Hotspots")
top_hotspots = hotspots[hotspots['Is_Hotspot'] == 1].nlargest(10, 'Incidents')
st.dataframe(top_hotspots[['Station', 'Category', 'Incidents']], 
             use_container_width=True)

# Forecast Section
st.subheader("🔮 Crime Forecasting")

if not filtered_data.empty:
    # Select station for forecast
    available_stations = filtered_data['Station'].unique()
    selected_station = st.selectbox("Select Station for Forecast", available_stations)
    
    station_data = filtered_data[filtered_data['Station'] == selected_station]
    
    if len(station_data) > 2:
        # Simple linear forecast
        years = station_data['Year'].dt.year.values
        incidents = station_data['Incidents'].values
        
        # Fit trend
        trend = np.polyfit(years - years[0], incidents, 1)
        
        # Forecast next 2 years
        future_years = [years[-1] + 1, years[-1] + 2]
        forecast_values = np.polyval(trend, [years[-1] - years[0] + 1, years[-1] - years[0] + 2])
        
        # Create forecast plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(x=years, y=incidents,
                               mode='lines+markers', name='Historical',
                               line=dict(color='blue', width=3)))
        
        # Forecast
        fig.add_trace(go.Scatter(x=future_years, y=forecast_values,
                               mode='lines+markers', name='Forecast',
                               line=dict(color='red', width=3, dash='dash')))
        
        fig.update_layout(title=f'2-Year Forecast for {selected_station}',
                         xaxis_title='Year', yaxis_title='Incidents')
        st.plotly_chart(fig, use_container_width=True)
        
        # Display forecast values
        st.write("**Forecast Results:**")
        for year, value in zip(future_years, forecast_values):
            st.write(f"- {year}: {value:.0f} incidents")

# Data Summary
st.subheader("📋 Dataset Summary")

col6, col7, col8, col9 = st.columns(4)

with col6:
    st.metric("Total Records", f"{len(df_long):,}")

with col7:
    st.metric("Time Period", "2005-2016")

with col8:
    st.metric("Police Stations", df['Station'].nunique())

with col9:
    st.metric("Crime Categories", df['Category'].nunique())

st.markdown("---")
st.markdown("**Built with ❤️ using Streamlit and Machine Learning**")


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class CrimeDroneSimulation:
    def __init__(self, grid_size=1000, num_drones=2):
        self.grid_size = grid_size  # 1km x 1km grid
        self.num_drones = num_drones
        self.hotspots = []
        self.drone_positions = []
        
    def generate_hotspots(self, num_hotspots=10):
        """Generate random crime hotspots within the grid"""
        np.random.seed(42)
        self.hotspots = np.random.rand(num_hotspots, 2) * self.grid_size
        return self.hotspots
    
    def initialize_drones(self):
        """Initialize drone positions at corners of the grid"""
        corners = np.array([[0, 0], [self.grid_size, 0], 
                          [0, self.grid_size], [self.grid_size, self.grid_size]])
        self.drone_positions = corners[:self.num_drones]
        return self.drone_positions
    
    def nearest_neighbor_path(self, start_position, points):
        """Calculate nearest neighbor path"""
        unvisited = points.copy()
        path = [start_position]
        current_point = start_position
        
        while len(unvisited) > 0:
            # Find nearest unvisited point
            distances = cdist([current_point], unvisited)[0]
            nearest_idx = np.argmin(distances)
            next_point = unvisited[nearest_idx]
            
            path.append(next_point)
            current_point = next_point
            unvisited = np.delete(unvisited, nearest_idx, axis=0)
        
        return np.array(path)
    
    def lawnmower_pattern(self, start_point, area_width, area_height, step_size=100):
        """Generate lawnmower pattern coverage"""
        x_coords = []
        y_coords = []
        
        x, y = start_point
        direction = 1  # 1 for right, -1 for left
        
        while y <= start_point[1] + area_height:
            while (direction == 1 and x <= start_point[0] + area_width) or \
                  (direction == -1 and x >= start_point[0]):
                x_coords.append(x)
                y_coords.append(y)
                x += step_size * direction
            
            # Move up and reverse direction
            y += step_size
            direction *= -1
            # Adjust x to stay within bounds
            x = max(0, min(x, start_point[0] + area_width))
        
        return np.column_stack([x_coords, y_coords])
    
    def simulate_flight(self, method='nearest_neighbor'):
        """Simulate drone flight pattern"""
        if method == 'nearest_neighbor':
            paths = []
            for i, start_pos in enumerate(self.drone_positions):
                # Assign hotspots to drones (simple partitioning)
                drone_hotspots = self.hotspots[i::self.num_drones]
                if len(drone_hotspots) > 0:
                    path = self.nearest_neighbor_path(start_pos, drone_hotspots)
                    paths.append(path)
            return paths
        
        elif method == 'lawnmower':
            paths = []
            for i, start_pos in enumerate(self.drone_positions):
                # Each drone covers a section of the grid
                section_width = self.grid_size / self.num_drones
                area_start = [i * section_width, 0]
                path = self.lawnmower_pattern(area_start, section_width, self.grid_size)
                paths.append(path)
            return paths
    
    def visualize_simulation(self, paths, method_name):
        """Visualize drone paths and hotspots"""
        plt.figure(figsize=(12, 8))
        
        # Plot hotspots
        plt.scatter(self.hotspots[:, 0], self.hotspots[:, 1], 
                   c='red', s=100, label='Crime Hotspots', alpha=0.7)
        
        # Plot drone paths
        colors = ['blue', 'green', 'orange', 'purple']
        for i, path in enumerate(paths):
            if len(path) > 0:
                color = colors[i % len(colors)]
                plt.plot(path[:, 0], path[:, 1], 
                        color=color, linewidth=2, marker='o', 
                        label=f'Drone {i+1} Path')
                plt.scatter(path[0, 0], path[0, 1], 
                          color=color, s=200, marker='s', edgecolors='black')
        
        plt.xlabel('X Coordinate (meters)')
        plt.ylabel('Y Coordinate (meters)')
        plt.title(f'Drone Surveillance Simulation - {method_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, self.grid_size)
        plt.ylim(0, self.grid_size)
        plt.gca().set_aspect('equal')
        plt.show()

# Run simulation
print("🚁 DRONE SURVEILLANCE SIMULATION")
print("=" * 50)

drone_sim = CrimeDroneSimulation(grid_size=1000, num_drones=2)

# Generate crime hotspots (using actual hotspot data from our analysis)
hotspots = drone_sim.generate_hotspots(num_hotspots=8)
drone_sim.initialize_drones()

print(f"Generated {len(hotspots)} crime hotspots")
print(f"Initialized {drone_sim.num_drones} drones")

# Test different path planning methods
methods = ['nearest_neighbor', 'lawnmower']

for method in methods:
    print(f"\n📡 Testing {method.replace('_', ' ').title()} Method:")
    paths = drone_sim.simulate_flight(method=method)
    
    # Calculate path statistics
    total_distance = 0
    for i, path in enumerate(paths):
        if len(path) > 1:
            distance = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
            total_distance += distance
            print(f"  Drone {i+1}: {len(path)} waypoints, {distance:.0f} meters")
    
    print(f"  Total distance: {total_distance:.0f} meters")
    
    # Visualize
    drone_sim.visualize_simulation(paths, method.replace('_', ' ').title())
