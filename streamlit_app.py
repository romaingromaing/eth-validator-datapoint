import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import subprocess
import sys
import threading
import queue
import time
import json

# Load environment variables from .env file
load_dotenv()

# Configuration
class Config:
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        self.table_name = os.getenv('SUPABASE_TABLE_NAME', 'validator_data')
        
        # Validate required environment variables
        if not self.supabase_url or not self.supabase_key:
            st.error("Missing SUPABASE_URL or SUPABASE_KEY in environment variables")
            st.stop()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data_from_supabase():
    """
    Load validator data from Supabase
    
    Returns:
        pd.DataFrame: Validator data
    """
    config = Config()
    
    try:
        # Initialize Supabase client
        supabase: Client = create_client(config.supabase_url, config.supabase_key)
        
        # Fetch data from Supabase
        response = supabase.table(config.table_name).select("*").execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            
            # Convert datetime columns
            if 'last_transaction_time' in df.columns:
                df['last_transaction_time'] = pd.to_datetime(df['last_transaction_time'], errors='coerce')
            
            return df
        else:
            st.error("No data found in the table.")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error loading data from Supabase: {e}")
        return pd.DataFrame()

def run_validator_analysis():
    """
    Run the validator analysis script and capture output in real-time
    """
    try:
        # Create a process to run the validator analysis
        process = subprocess.Popen(
            [sys.executable, "validator_analysis.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Read output line by line
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            if line:
                output_lines.append(line.strip())
                yield line.strip()
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            yield f"❌ Process finished with error code: {return_code}"
        else:
            yield "✅ Validator analysis completed successfully!"
            
    except Exception as e:
        yield f"❌ Error running validator analysis: {str(e)}"

def analysis_tab():
    """
    Tab for running the validator analysis
    """
    st.header("🚀 Run Validator Analysis")
    
    st.markdown("""
    This will run the complete validator analysis pipeline:
    1. Load validator data from JSON file
    2. Fetch deposit addresses from BeaconChain API
    3. Get transaction data from Dune API
    4. Check for smart contract deployments
    5. Identify DEX addresses
    6. Save results to Supabase
    """)
    
    # Check if required files exist
    required_files = ["validator_analysis.py", "0x00-validators.json"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        st.error(f"Missing required files: {', '.join(missing_files)}")
        st.info("Please ensure all required files are in the same directory as this Streamlit app.")
        return
    
    # Environment variables check
    env_vars = ['DUNE_SIM_API_KEY', 'DUNE_CLIENT_API_KEY', 'SUPABASE_URL', 'SUPABASE_KEY']
    missing_env = [var for var in env_vars if not os.getenv(var)]
    
    if missing_env:
        st.error(f"Missing environment variables: {', '.join(missing_env)}")
        st.info("Please set all required environment variables in your .env file.")
        return
    
    st.success("✅ All requirements met. Ready to run analysis!")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🔄 Run Analysis", type="primary", use_container_width=True):
            st.session_state.run_analysis = True
    
    with col2:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Run analysis if button was clicked
    if st.session_state.get('run_analysis', False):
        st.markdown("---")
        st.subheader("📊 Analysis Progress")
        
        # Create containers for different types of output
        progress_container = st.container()
        output_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        with output_container:
            output_text = st.empty()
        
        # Run the analysis
        output_lines = []
        total_steps = 6  # Approximate number of major steps
        current_step = 0
        
        try:
            for line in run_validator_analysis():
                output_lines.append(line)
                
                # Update progress based on key milestones
                if "All required environment variables loaded successfully" in line:
                    current_step = 1
                    status_text.text("✓ Environment variables loaded")
                elif "Processing batch" in line:
                    current_step = 2
                    status_text.text("🔄 Fetching deposit addresses...")
                elif "Fetching and analyzing transaction data" in line:
                    current_step = 3
                    status_text.text("🔄 Analyzing transaction data...")
                elif "ethereum-dex-addresses" in line:
                    current_step = 4
                    status_text.text("🔄 Checking DEX addresses...")
                elif "Saved to" in line and "csv" in line:
                    current_step = 5
                    status_text.text("✓ Data saved to CSV")
                elif "Successfully overwritten Supabase table" in line:
                    current_step = 6
                    status_text.text("✅ Data uploaded to Supabase")
                
                # Update progress bar
                progress = min(current_step / total_steps, 1.0)
                progress_bar.progress(progress)
                
                # Show recent output (last 20 lines)
                recent_output = output_lines[-20:] if len(output_lines) > 20 else output_lines
                output_text.text_area(
                    "Real-time Output",
                    value="\n".join(recent_output),
                    height=300,
                    key=f"output_{len(output_lines)}"
                )
                
                # Small delay to make updates visible
                time.sleep(0.1)
        
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
        
        finally:
            # Reset the run analysis flag
            st.session_state.run_analysis = False
            
            # Show completion message
            if current_step >= 6:
                st.success("🎉 Analysis completed successfully! Data has been updated in Supabase.")
                st.balloons()
            else:
                st.warning("⚠️ Analysis may not have completed successfully. Check the output above.")

def dashboard_tab():
    """
    Original dashboard functionality
    """
    st.header("🔍 Validator Analysis Dashboard")
    
    # Load data
    with st.spinner("Loading data from Supabase..."):
        df = load_data_from_supabase()
    
    if df.empty:
        st.warning("No data available. Please run the validator analysis first or check your Supabase connection.")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Smart contract filter
    contract_filter = st.sidebar.selectbox(
        "Smart Contract Status",
        ["All", "Smart Contract Deployers", "Non-Deployers"]
    )
    
    # DEX filter
    dex_filter = st.sidebar.selectbox(
        "DEX Status", 
        ["All", "DEX Addresses", "Non-DEX Addresses"]
    )
    
    # Date range filter (if last_transaction_time exists)
    if 'last_transaction_time' in df.columns and df['last_transaction_time'].notna().any():
        min_date = df['last_transaction_time'].min()
        max_date = df['last_transaction_time'].max()
        
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.sidebar.date_input(
                "Last Transaction Date Range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
    
    # Apply filters
    filtered_df = df.copy()
    
    if contract_filter == "Smart Contract Deployers":
        filtered_df = filtered_df[filtered_df['is_smart_contract'] == True]
    elif contract_filter == "Non-Deployers":
        filtered_df = filtered_df[filtered_df['is_smart_contract'] == False]
    
    if dex_filter == "DEX Addresses":
        filtered_df = filtered_df[filtered_df['is_dex'] == True]
    elif dex_filter == "Non-DEX Addresses":
        filtered_df = filtered_df[filtered_df['is_dex'] == False]
    
    # Date filter
    if 'last_transaction_time' in df.columns and 'date_range' in locals():
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df['last_transaction_time'].dt.date >= start_date) &
                (filtered_df['last_transaction_time'].dt.date <= end_date)
            ]
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Validators", len(filtered_df))
    
    with col2:
        smart_contract_count = filtered_df['is_smart_contract'].sum() if 'is_smart_contract' in filtered_df.columns else 0
        st.metric("Smart Contract Deployers", smart_contract_count)
    
    with col3:
        dex_count = filtered_df['is_dex'].sum() if 'is_dex' in filtered_df.columns else 0
        st.metric("DEX Addresses", dex_count)
    
    with col4:
        active_validators = filtered_df['last_transaction_time'].notna().sum() if 'last_transaction_time' in filtered_df.columns else 0
        st.metric("With Transaction History", active_validators)
    
    st.markdown("---")
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Smart Contract Deployment Distribution")
        if 'is_smart_contract' in filtered_df.columns:
            contract_counts = filtered_df['is_smart_contract'].value_counts()
            fig_pie = px.pie(
                values=contract_counts.values,
                names=['Non-Deployers' if not x else 'Smart Contract Deployers' for x in contract_counts.index],
                title="Smart Contract Deployment Status"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No smart contract data available")
    
    with col2:
        st.subheader("DEX Address Distribution")
        if 'is_dex' in filtered_df.columns:
            dex_counts = filtered_df['is_dex'].value_counts()
            fig_dex = px.pie(
                values=dex_counts.values,
                names=['Non-DEX' if not x else 'DEX Addresses' for x in dex_counts.index],
                title="DEX Address Status"
            )
            st.plotly_chart(fig_dex, use_container_width=True)
        else:
            st.info("No DEX data available")
    
    # Transaction timeline
    if 'last_transaction_time' in filtered_df.columns:
        st.subheader("Transaction Timeline")
        timeline_df = filtered_df.dropna(subset=['last_transaction_time'])
        
        if not timeline_df.empty:
            # Create bins for the timeline
            timeline_df['transaction_date'] = timeline_df['last_transaction_time'].dt.date
            daily_counts = timeline_df['transaction_date'].value_counts().sort_index()
            
            fig_timeline = px.line(
                x=daily_counts.index,
                y=daily_counts.values,
                title="Last Transactions Over Time",
                labels={'x': 'Date', 'y': 'Number of Validators'}
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("No transaction timeline data available")
    
    # Data table
    st.subheader("Validator Data")
    
    # Display options
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"Showing {len(filtered_df)} of {len(df)} total records")
    
    with col2:
        show_all_columns = st.checkbox("Show all columns", value=False)
    
    # Select columns to display
    if show_all_columns:
        display_df = filtered_df
    else:
        # Show only key columns
        key_columns = []
        for col in ['index', 'pubkey', 'deposit_address', 'last_transaction_time', 'is_smart_contract', 'is_dex']:
            if col in filtered_df.columns:
                key_columns.append(col)
        display_df = filtered_df[key_columns] if key_columns else filtered_df
    
    # Format the display
    if 'last_transaction_time' in display_df.columns:
        display_df = display_df.copy()
        display_df['last_transaction_time'] = display_df['last_transaction_time'].dt.strftime('%Y-%m-%d %H:%M')
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    # Download section
    st.subheader("Export Data")
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"validator_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def main():
    st.set_page_config(
        page_title="Validator Analysis Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 Validator Analysis Platform")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🚀 Run Analysis", "📊 Dashboard"])
    
    with tab1:
        analysis_tab()
    
    with tab2:
        dashboard_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("*Use the 'Run Analysis' tab to update data, then view results in the 'Dashboard' tab*")

if __name__ == "__main__":
    main()