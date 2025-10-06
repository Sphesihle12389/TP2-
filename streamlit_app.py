import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="SA Crime Analytics", page_icon="🔍", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('SouthAfricaCrimeStats_v2.csv')
        
        # Get year columns (all columns from index 3 onwards)

        import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import io
import base64

class CrimeDroneSimulation:
    def __init__(self, grid_size=1000, num_drones=2):
        self.grid_size = grid_size
        self.num_drones = num_drones
        self.hotspots = []
        self.drone_positions = []
        
    def generate_hotspots(self, num_hotspots=10):
        np.random.seed(42)
        self.hotspots = np.random.rand(num_hotspots, 2) * self.grid_size
        return self.hotspots
    
    def initialize_drones(self):
        self.drone_positions = np.random.rand(self.num_drones, 2) * self.grid_size
        return self.drone_positions
    
    def nearest_neighbor_path(self, start_position, points):
        if len(points) == 0:
            return np.array([start_position])
        unvisited = points.copy()
        path = [start_position]
        current_point = start_position
        while len(unvisited) > 0:
            distances = cdist([current_point], unvisited)[0]
            nearest_idx = np.argmin(distances)
            next_point = unvisited[nearest_idx]
            path.append(next_point)
            current_point = next_point
            unvisited = np.delete(unvisited, nearest_idx, axis=0)
        return np.array(path)
    
    def simulate_flight(self, method='nearest_neighbor'):
        paths = []
        for i, start_pos in enumerate(self.drone_positions):
            drone_hotspots = self.hotspots[i::self.num_drones]
            if len(drone_hotspots) > 0:
                path = self.nearest_neighbor_path(start_pos, drone_hotspots)
                paths.append(path)
            else:
                paths.append(np.array([start_pos]))
        return paths

def create_drone_visualization(paths, hotspots, drone_positions, grid_size, method_name):
    """Create and return a matplotlib figure as base64 string"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot hotspots
    ax.scatter(hotspots[:, 0], hotspots[:, 1], 
               c='red', s=100, label='Crime Hotspots', alpha=0.7)
    
    # Plot drone paths
    colors = ['blue', 'green', 'orange', 'purple']
    for i, path in enumerate(paths):
        color = colors[i % len(colors)]
        ax.plot(path[:, 0], path[:, 1], 
                color=color, linewidth=2, marker='o', 
                label=f'Drone {i+1} Path')
        ax.scatter(path[0, 0], path[0, 1], 
                  color=color, s=200, marker='s', edgecolors='black')
    
    ax.set_xlabel('X Coordinate (meters)')
    ax.set_ylabel('Y Coordinate (meters)')
    ax.set_title(f'Drone Surveillance - {method_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    
    # Convert plot to image for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# Add to your existing Streamlit app
def drone_simulation_page():
    st.header("🚁 Drone Surveillance Simulation")
    st.markdown("Simulate drone patrol routes for crime hotspot monitoring")
    
    # Simulation controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_drones = st.slider("Number of Drones", 1, 4, 2)
    
    with col2:
        num_hotspots = st.slider("Number of Hotspots", 5, 20, 8)
    
    with col3:
        grid_size = st.slider("Grid Size (meters)", 500, 2000, 1000)
    
    # Run simulation button
    if st.button("🚀 Run Drone Simulation", type="primary"):
        with st.spinner("Running drone simulation..."):
            # Initialize simulation
            drone_sim = CrimeDroneSimulation(grid_size=grid_size, num_drones=num_drones)
            hotspots = drone_sim.generate_hotspots(num_hotspots=num_hotspots)
            drone_sim.initialize_drones()
            
            # Run simulation
            paths = drone_sim.simulate_flight(method='nearest_neighbor')
            
            # Calculate statistics
            total_distance = 0
            drone_stats = []
            for i, path in enumerate(paths):
                if len(path) > 1:
                    distance = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
                    total_distance += distance
                    drone_stats.append({
                        'drone': i+1,
                        'waypoints': len(path),
                        'distance': f"{distance:.0f} meters"
                    })
            
            # Display statistics
            st.subheader("📊 Simulation Results")
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            
            with stats_col1:
                st.metric("Total Hotspots", len(hotspots))
            
            with stats_col2:
                st.metric("Total Drones", num_drones)
            
            with stats_col3:
                st.metric("Total Distance", f"{total_distance:.0f} meters")
            
            # Show drone details
            st.write("**Drone Patrol Details:**")
            for stat in drone_stats:
                st.write(f"- Drone {stat['drone']}: {stat['waypoints']} waypoints, {stat['distance']}")
            
            # Display visualization
            st.subheader("🗺️ Drone Patrol Routes")
            image_buf = create_drone_visualization(
                paths, hotspots, drone_sim.drone_positions, grid_size, 'Nearest Neighbor Path'
            )
            st.image(image_buf, use_column_width=True)
            
            # Additional insights
            st.subheader("💡 Patrol Insights")
            avg_distance_per_drone = total_distance / num_drones
            efficiency = (len(hotspots) / total_distance) * 1000 if total_distance > 0 else 0
            
            insight_col1, insight_col2 = st.columns(2)
            with insight_col1:
                st.metric("Avg Distance per Drone", f"{avg_distance_per_drone:.0f} meters")
            
            with insight_col2:
                st.metric("Patrol Efficiency", f"{efficiency:.2f} hotspots/km")

# Integrate into your main app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Crime Analytics", "Drone Simulation"])
    
    if page == "Crime Analytics":
        # Your existing crime analytics code here
        st.title("🔍 South Africa Crime Analytics Dashboard")
        # ... your existing crime analysis code
        
    elif page == "Drone Simulation":
        drone_simulation_page()

if __name__ == "__main__":
    main()
        year_columns = [col for col in df.columns if '-' in col]
        
        # Reshape data
        df_long = pd.melt(df, 
                         id_vars=['Province', 'Station', 'Category'], 
                         value_vars=year_columns,
                         var_name='Year', 
                         value_name='Incidents')
        
        # Convert year to datetime
        df_long['Year'] = pd.to_datetime(df_long['Year'].str.split('-').str[0])
        df_long['Year_Num'] = df_long['Year'].dt.year
        
        return df, df_long
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Load data
df, df_long = load_data()

if df is None:
    st.stop()

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

try:
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

except Exception as e:
    st.error(f"Error in hotspot analysis: {e}")

# Simple Forecasting Section
st.subheader("🔮 Crime Forecasting")

if not filtered_data.empty:
    try:
        # Select station for forecast
        available_stations = filtered_data['Station'].unique()
        selected_station = st.selectbox("Select Station for Forecast", available_stations)
        
        station_data = filtered_data[filtered_data['Station'] == selected_station]
        
        if len(station_data) > 2:
            # Simple linear forecast
            years = station_data['Year_Num'].values
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
                
    except Exception as e:
        st.error(f"Error in forecasting: {e}")

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
