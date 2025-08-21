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
        # Helper function to get secrets from either Streamlit secrets or environment variables
        def get_secret(key, default=None):
            # Try Streamlit secrets first with proper error handling
            try:
                if hasattr(st, 'secrets') and st.secrets is not None:
                    return st.secrets.get(key)
            except Exception:
                # If secrets.toml doesn't exist or has issues, continue to env vars
                pass
            
            # Fall back to environment variables
            import os
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass  # dotenv might not be installed
            return os.getenv(key, default)
        
        self.supabase_url = get_secret('SUPABASE_URL')
        self.supabase_key = get_secret('SUPABASE_KEY')
        self.table_name = get_secret('SUPABASE_TABLE_NAME', 'validator_data')
        
        # Validate required environment variables
        if not self.supabase_url or not self.supabase_key:
            st.error("Missing SUPABASE_URL or SUPABASE_KEY in configuration")
            st.info("""
            **For local development:** Make sure your `.env` file contains:
            ```
            SUPABASE_URL=your_supabase_url
            SUPABASE_KEY=your_supabase_key
            SUPABASE_DATABASE_URL=your_database_connection_string
            ```
            
            **For Streamlit Cloud:** Add these to your app's Secrets in the dashboard.
            """)
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

def check_environment_variables():
    """
    Check if all required environment variables/secrets are available
    Returns: (bool, list) - (all_present, missing_vars)
    """
    def get_secret(key):
        # Try Streamlit secrets first, with proper error handling
        try:
            if hasattr(st, 'secrets') and st.secrets is not None:
                return st.secrets.get(key)
        except Exception:
            # If secrets.toml doesn't exist or has issues, continue to env vars
            pass
        
        # Fall back to environment variables
        import os
        return os.getenv(key)
    
    required_vars = [
        'DUNE_SIM_API_KEY',
        'DUNE_CLIENT_API_KEY', 
        'SUPABASE_URL',
        'SUPABASE_KEY',
        'SUPABASE_DATABASE_URL'  # Added this for direct CSV upload
    ]
    
    missing_vars = []
    for var in required_vars:
        value = get_secret(var)
        if not value:
            missing_vars.append(var)
    
    return len(missing_vars) == 0, missing_vars

def run_validator_analysis():
    """
    Run the validator analysis script and capture output in real-time with enhanced monitoring
    """
    try:
        # Create a process to run the validator analysis with unbuffered output
        process = subprocess.Popen(
            [sys.executable, "-u", "validator_analysis.py"],  # -u for unbuffered output
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=0,  # Unbuffered
            env=os.environ.copy()  # Pass all environment variables
        )
        
        # Read output line by line in real-time
        for line in iter(process.stdout.readline, ''):
            if line.strip():  # Only yield non-empty lines
                yield line.strip()
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            yield f"❌ Process finished with error code: {return_code}"
        else:
            yield "✅ Validator analysis completed successfully!"
            
    except Exception as e:
        yield f"❌ Error running validator analysis: {str(e)}"

def parse_progress_from_output(line):
    """
    Parse progress information from output line
    Returns: (step_name, percentage, description)
    """
    progress_patterns = {
        "environment variables loaded": ("Environment Setup", 5, "Loading configuration..."),
        "Starting processing": ("Data Loading", 10, "Loading validator data..."),
        "Processing batch": ("Deposit Addresses", 30, "Fetching deposit addresses..."),
        "Fetching and analyzing transaction data": ("Transaction Analysis", 50, "Analyzing transactions..."),
        "ethereum-dex-addresses": ("DEX Analysis", 70, "Checking DEX addresses..."),
        "Saved to": ("CSV Export", 85, "Saving data to CSV..."),
        "Clearing existing data": ("Database Cleanup", 90, "Preparing database..."),
        "Successfully uploaded CSV": ("Upload Complete", 100, "Data uploaded to Supabase!")
    }
    
    for pattern, (step, percentage, desc) in progress_patterns.items():
        if pattern.lower() in line.lower():
            return step, percentage, desc
    
    # Extract batch progress if available
    if "Processing batch" in line and "/" in line:
        try:
            # Extract numbers from "Processing batch X/Y"
            parts = line.split("Processing batch")[1].split("(")[0].strip()
            current, total = parts.split("/")
            current, total = int(current), int(total)
            batch_progress = (current / total) * 20 + 10  # Scale to 10-30% range
            return "Deposit Addresses", batch_progress, f"Processing batch {current}/{total}"
        except:
            pass
    
    # Extract transaction progress if available
    if "[" in line and "]" in line and "Processing:" in line:
        try:
            # Extract numbers from "[X/Y] Processing:"
            bracket_content = line.split("[")[1].split("]")[0]
            current, total = bracket_content.split("/")
            current, total = int(current), int(total)
            tx_progress = (current / total) * 20 + 50  # Scale to 50-70% range
            return "Transaction Analysis", tx_progress, f"Analyzing transaction {current}/{total}"
        except:
            pass
    
    return None, None, None

def analysis_tab():
    """
    Enhanced tab for running the validator analysis with real-time progress
    """
    st.header("🚀 Run Validator Analysis")
    
    st.markdown("""
    This will run the complete validator analysis pipeline:
    1. **Environment Setup** - Load configuration and validate settings
    2. **Data Loading** - Load validator data from JSON file
    3. **Deposit Addresses** - Fetch deposit addresses from BeaconChain API
    4. **Transaction Analysis** - Get transaction data from Dune API and check for smart contracts
    5. **DEX Analysis** - Identify DEX addresses using Dune query
    6. **CSV Export** - Save processed data to CSV file
    7. **Database Upload** - Upload data directly to Supabase using PostgreSQL COPY
    """)
    
    # Check if required files exist
    required_files = ["validator_analysis.py", "0x00-validators.json"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        st.error(f"Missing required files: {', '.join(missing_files)}")
        st.info("Please ensure all required files are in the same directory as this Streamlit app.")
        return
    
    # Environment variables check with enhanced validation
    all_vars_present, missing_vars = check_environment_variables()
    
    if not all_vars_present:
        st.error(f"Missing environment variables: {', '.join(missing_vars)}")
        
        with st.expander("📋 Environment Variables Setup Guide"):
            st.markdown("""
            **Required Environment Variables:**
            
            ```
            # Dune API Keys
            DUNE_SIM_API_KEY=your_dune_sim_api_key
            DUNE_CLIENT_API_KEY=your_dune_client_api_key
            
            # Supabase Configuration
            SUPABASE_URL=your_supabase_url
            SUPABASE_KEY=your_supabase_anon_key
            SUPABASE_DATABASE_URL=postgresql://postgres.xxx:password@aws-x-region.pooler.supabase.com:5432/postgres
            SUPABASE_TABLE_NAME=validator_data
            
            # Optional Configuration
            BATCH_SIZE=100
            DELAY_SECONDS=6
            API_DELAY=0.2
            ```
            
            **For Streamlit Cloud:** Add these to your app's Secrets in the dashboard.
            **For local development:** Add these to your `.env` file.
            """)
        return
    
    # Show current configuration
    with st.expander("⚙️ Current Configuration"):
        config_info = {
            "Table Name": os.getenv('SUPABASE_TABLE_NAME', 'validator_data'),
            "Batch Size": os.getenv('BATCH_SIZE', '100'),
            "API Delay": f"{os.getenv('API_DELAY', '0.2')}s",
            "Request Delay": f"{os.getenv('DELAY_SECONDS', '6')}s"
        }
        
        for key, value in config_info.items():
            st.text(f"{key}: {value}")
    
    st.success("✅ All requirements met. Ready to run analysis!")
    
    # Control buttons
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            st.session_state.run_analysis = True
            st.session_state.analysis_stopped = False
    
    with col2:
        if st.button("⏹️ Stop Analysis", use_container_width=True):
            st.session_state.analysis_stopped = True
            st.session_state.run_analysis = False
    
    with col3:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Run analysis if button was clicked
    if st.session_state.get('run_analysis', False) and not st.session_state.get('analysis_stopped', False):
        st.markdown("---")
        st.subheader("📊 Analysis Progress")
        
        # Create enhanced progress tracking containers
        progress_container = st.container()
        metrics_container = st.container()
        output_container = st.container()
        
        with progress_container:
            # Main progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step-by-step progress
            col1, col2, col3, col4 = st.columns(4)
            step_indicators = [col1.empty(), col2.empty(), col3.empty(), col4.empty()]
        
        with metrics_container:
            # Real-time metrics
            metric_cols = st.columns(4)
            metrics = {
                'processed': metric_cols[0].empty(),
                'success_rate': metric_cols[1].empty(),
                'elapsed_time': metric_cols[2].empty(),
                'eta': metric_cols[3].empty()
            }
        
        with output_container:
            # Output display options
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("📝 Real-time Output")
            with col2:
                show_all_output = st.checkbox("Show all output", value=False)
            
            output_text = st.empty()
        
        # Initialize tracking variables
        output_lines = []
        start_time = time.time()
        current_progress = 0
        current_step = "Starting..."
        
        try:
            for line in run_validator_analysis():
                if st.session_state.get('analysis_stopped', False):
                    st.warning("⚠️ Analysis stopped by user")
                    break
                
                output_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")
                
                # Parse progress from output
                step_name, progress, description = parse_progress_from_output(line)
                if step_name:
                    current_progress = progress
                    current_step = description or step_name
                
                # Update progress indicators
                progress_bar.progress(min(current_progress / 100, 1.0))
                status_text.text(f"🔄 {current_step} ({current_progress:.0f}%)")
                
                # Update step indicators
                steps = ["Environment", "Data Loading", "API Calls", "Upload"]
                for i, indicator in enumerate(step_indicators):
                    if current_progress > (i + 1) * 25:
                        indicator.success(f"✅ {steps[i]}")
                    elif current_progress > i * 25:
                        indicator.info(f"🔄 {steps[i]}")
                    else:
                        indicator.empty()
                
                # Update metrics
                elapsed = time.time() - start_time
                metrics['elapsed_time'].metric("⏱️ Elapsed", f"{elapsed:.0f}s")
                
                # Estimate completion time based on progress
                if current_progress > 5:
                    estimated_total = elapsed * (100 / current_progress)
                    eta = max(0, estimated_total - elapsed)
                    metrics['eta'].metric("⏳ ETA", f"{eta:.0f}s")
                
                # Count successful operations
                success_count = len([l for l in output_lines if "✓" in l])
                error_count = len([l for l in output_lines if ("✗" in l or "Error" in l)])
                
                if success_count + error_count > 0:
                    success_rate = (success_count / (success_count + error_count)) * 100
                    metrics['success_rate'].metric("✅ Success Rate", f"{success_rate:.1f}%")
                
                # Update output display
                if show_all_output:
                    display_lines = output_lines
                else:
                    # Show last 25 lines for better readability
                    display_lines = output_lines[-25:] if len(output_lines) > 25 else output_lines
                
                # Format output with color coding
                formatted_output = []
                for output_line in display_lines:
                    if "✓" in output_line:
                        formatted_output.append(f"✅ {output_line}")
                    elif "✗" in output_line or "Error" in output_line:
                        formatted_output.append(f"❌ {output_line}")
                    elif "Processing batch" in output_line:
                        formatted_output.append(f"🔄 {output_line}")
                    elif "Found" in output_line and "transactions" in output_line:
                        formatted_output.append(f"📊 {output_line}")
                    else:
                        formatted_output.append(f"   {output_line}")
                
                output_text.text_area(
                    "Console Output",
                    value="\n".join(formatted_output),
                    height=400,
                    key=f"output_{len(output_lines)}"
                )
                
                # Small delay to prevent UI freezing
                time.sleep(0.05)
        
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            output_lines.append(f"[ERROR] {str(e)}")
        
        finally:
            # Final status update
            elapsed_total = time.time() - start_time
            
            if current_progress >= 100:
                st.success(f"🎉 Analysis completed successfully in {elapsed_total:.1f} seconds!")
                st.balloons()
                
                # Show completion summary
                with st.expander("📈 Analysis Summary"):
                    total_lines = len(output_lines)
                    success_operations = len([l for l in output_lines if "✓" in l])
                    error_operations = len([l for l in output_lines if ("✗" in l or "Error" in l)])
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Operations", total_lines)
                    col2.metric("Successful", success_operations)
                    col3.metric("Errors", error_operations)
                
            elif st.session_state.get('analysis_stopped', False):
                st.warning(f"⚠️ Analysis stopped after {elapsed_total:.1f} seconds")
            else:
                st.error(f"❌ Analysis may not have completed successfully after {elapsed_total:.1f} seconds")
            
            # Reset flags
            st.session_state.run_analysis = False
            st.session_state.analysis_stopped = False
            
            # Option to download log
            if output_lines:
                log_content = "\n".join(output_lines)
                st.download_button(
                    label="📥 Download Analysis Log",
                    data=log_content,
                    file_name=f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

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