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

def output_reader_thread(process, output_queue, stop_event):
    """
    Thread function to continuously read process output
    This runs in background and feeds output to the queue
    """
    try:
        while not stop_event.is_set() and process.poll() is None:
            line = process.stdout.readline()
            if line:
                timestamp = datetime.now().strftime('%H:%M:%S')
                formatted_line = f"[{timestamp}] {line.strip()}"
                output_queue.put(formatted_line)
            else:
                time.sleep(0.1)  # Small delay if no output
        
        # Process finished - read any remaining output
        while True:
            line = process.stdout.readline()
            if not line:
                break
            timestamp = datetime.now().strftime('%H:%M:%S')
            formatted_line = f"[{timestamp}] {line.strip()}"
            output_queue.put(formatted_line)
        
        # Signal completion
        exit_code = process.poll()
        if exit_code == 0:
            output_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] Process completed successfully!")
        else:
            output_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] Process failed with exit code: {exit_code}")
            
    except Exception as e:
        output_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] Output reader error: {str(e)}")

def start_analysis_with_threading():
    """
    Start the validator analysis with proper threading for real-time output
    Returns (process, output_queue, stop_event, reader_thread)
    """
    try:
        # Start the subprocess
        process = subprocess.Popen(
            [sys.executable, "-u", "validator_analysis.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,  # Line buffered
            env=os.environ.copy(),
            cwd=os.getcwd()
        )
        
        # Create queue and stop event for thread communication
        output_queue = queue.Queue()
        stop_event = threading.Event()
        
        # Start the output reader thread
        reader_thread = threading.Thread(
            target=output_reader_thread,
            args=(process, output_queue, stop_event),
            daemon=True
        )
        reader_thread.start()
        
        return process, output_queue, stop_event, reader_thread
        
    except Exception as e:
        st.error(f"Failed to start analysis process: {e}")
        return None, None, None, None

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
    Enhanced tab for running the validator analysis with FULL real-time output
    """
    st.header("Run Validator Analysis")
    
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
    
    # Environment variables check
    all_vars_present, missing_vars = check_environment_variables()
    
    if not all_vars_present:
        st.error(f"Missing environment variables: {', '.join(missing_vars)}")
        with st.expander("Environment Variables Setup Guide"):
            st.markdown("""
            **Required Environment Variables:**
            ```
            DUNE_SIM_API_KEY=your_dune_sim_api_key
            DUNE_CLIENT_API_KEY=your_dune_client_api_key
            SUPABASE_URL=your_supabase_url
            SUPABASE_KEY=your_supabase_anon_key
            SUPABASE_DATABASE_URL=postgresql://postgres.xxx:password@aws-x-region.pooler.supabase.com:5432/postgres
            SUPABASE_TABLE_NAME=validator_data
            ```
            """)
        return
    
    st.success("All requirements met. Ready to run analysis!")
    
    # Initialize session state
    if 'analysis_process' not in st.session_state:
        st.session_state.analysis_process = None
    if 'output_queue' not in st.session_state:
        st.session_state.output_queue = None
    if 'stop_event' not in st.session_state:
        st.session_state.stop_event = None
    if 'reader_thread' not in st.session_state:
        st.session_state.reader_thread = None
    if 'analysis_output' not in st.session_state:
        st.session_state.analysis_output = []
    if 'analysis_status' not in st.session_state:
        st.session_state.analysis_status = "idle"  # idle, running, completed, failed
    if 'analysis_start_time' not in st.session_state:
        st.session_state.analysis_start_time = None
    if 'current_progress' not in st.session_state:
        st.session_state.current_progress = 0
    if 'current_step' not in st.session_state:
        st.session_state.current_step = "Ready"
    
    # Control buttons
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        start_disabled = (st.session_state.analysis_status == "running")
        if st.button("Start Analysis", type="primary", use_container_width=True, disabled=start_disabled):
            # Clean up any existing process
            if st.session_state.stop_event:
                st.session_state.stop_event.set()
            if st.session_state.analysis_process:
                try:
                    st.session_state.analysis_process.terminate()
                except:
                    pass
            
            # Start new analysis with threading
            process, output_queue, stop_event, reader_thread = start_analysis_with_threading()
            
            if process:
                st.session_state.analysis_process = process
                st.session_state.output_queue = output_queue
                st.session_state.stop_event = stop_event
                st.session_state.reader_thread = reader_thread
                st.session_state.analysis_output = []
                st.session_state.analysis_status = "running"
                st.session_state.analysis_start_time = time.time()
                st.session_state.current_progress = 0
                st.session_state.current_step = "Starting analysis..."
    
    with col2:
        stop_disabled = (st.session_state.analysis_status != "running")
        if st.button("Stop Analysis", use_container_width=True, disabled=stop_disabled):
            if st.session_state.stop_event:
                st.session_state.stop_event.set()
            if st.session_state.analysis_process:
                try:
                    st.session_state.analysis_process.terminate()
                    st.session_state.analysis_process.wait(timeout=5)
                except:
                    try:
                        st.session_state.analysis_process.kill()
                    except:
                        pass
            
            st.session_state.analysis_status = "failed"
            st.session_state.current_step = "Stopped by user"
    
    with col3:
        if st.button("Reset & Clear", use_container_width=True):
            # Clean up everything
            if st.session_state.stop_event:
                st.session_state.stop_event.set()
            if st.session_state.analysis_process:
                try:
                    st.session_state.analysis_process.terminate()
                except:
                    pass
            
            # Reset all state
            for key in ['analysis_process', 'output_queue', 'stop_event', 'reader_thread', 
                       'analysis_output', 'analysis_start_time', 'current_progress', 'current_step']:
                st.session_state[key] = None if 'time' in key or 'progress' in key else ([] if 'output' in key else ("idle" if key == 'analysis_status' else None))
            
            st.session_state.analysis_status = "idle"
            st.session_state.current_progress = 0
            st.session_state.current_step = "Ready"
            st.cache_data.clear()
    
    # Real-time processing and display
    if st.session_state.analysis_status == "running":
        st.info("Analysis is currently running...")
        
        # Create display containers
        progress_container = st.container()
        metrics_container = st.container() 
        output_container = st.container()
        
        # Process new output from queue
        new_output_received = False
        if st.session_state.output_queue:
            # Read ALL available output from queue
            while True:
                try:
                    line = st.session_state.output_queue.get_nowait()
                    st.session_state.analysis_output.append(line)
                    new_output_received = True
                    
                    # Parse progress from this line
                    step_name, progress, description = parse_progress_from_output(line)
                    if step_name and progress:
                        st.session_state.current_progress = progress
                        st.session_state.current_step = description or step_name
                    
                    # Check for completion indicators
                    if "completed successfully" in line.lower():
                        st.session_state.analysis_status = "completed"
                    elif "failed" in line.lower() and "exit code" in line.lower():
                        st.session_state.analysis_status = "failed"
                        
                except queue.Empty:
                    break  # No more output available right now
        
        # Check if process finished
        if st.session_state.analysis_process and st.session_state.analysis_process.poll() is not None:
            exit_code = st.session_state.analysis_process.poll()
            if st.session_state.analysis_status == "running":  # Only update if not already set
                if exit_code == 0:
                    st.session_state.analysis_status = "completed"
                    st.session_state.current_progress = 100
                    st.session_state.current_step = "Analysis completed!"
                else:
                    st.session_state.analysis_status = "failed"
                    st.session_state.current_step = f"Process failed with exit code: {exit_code}"
        
        # Display progress
        with progress_container:
            progress_bar = st.progress(min(st.session_state.current_progress / 100, 1.0))
            status_text = st.text(f"{st.session_state.current_step} ({st.session_state.current_progress:.0f}%)")
            
            # Step indicators
            col1, col2, col3, col4 = st.columns(4)
            steps = ["Environment", "Data Loading", "API Calls", "Upload"]
            step_indicators = [col1, col2, col3, col4]
            
            for i, (indicator, step_name) in enumerate(zip(step_indicators, steps)):
                if st.session_state.current_progress > (i + 1) * 25:
                    indicator.success(f"✓ {step_name}")
                elif st.session_state.current_progress > i * 25:
                    indicator.info(f"• {step_name}")
                else:
                    indicator.empty()
        
        # Display metrics
        with metrics_container:
            col1, col2, col3, col4 = st.columns(4)
            
            # Elapsed time
            if st.session_state.analysis_start_time:
                elapsed = time.time() - st.session_state.analysis_start_time
                col1.metric("Elapsed Time", f"{elapsed:.0f}s")
                
                # ETA calculation
                if st.session_state.current_progress > 5:
                    estimated_total = elapsed * (100 / st.session_state.current_progress)
                    eta = max(0, estimated_total - elapsed)
                    col2.metric("ETA", f"{eta:.0f}s")
            
            # Success/error counts
            success_count = len([l for l in st.session_state.analysis_output if "✓" in l or "completed successfully" in l.lower()])
            error_count = len([l for l in st.session_state.analysis_output if "✗" in l or "error" in l.lower()])
            
            col3.metric("Successful Operations", success_count)
            col4.metric("Errors", error_count)
        
        # Display real-time output
        with output_container:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("Real-time Output")
            with col2:
                show_all = st.checkbox("Show all output", value=True, key="show_all_running")
            
            if st.session_state.analysis_output:
                # Determine what to show
                if show_all:
                    display_lines = st.session_state.analysis_output
                else:
                    display_lines = st.session_state.analysis_output[-50:] if len(st.session_state.analysis_output) > 50 else st.session_state.analysis_output
                
                # Format output with status indicators
                formatted_output = []
                for line in display_lines:
                    if any(success_indicator in line for success_indicator in ["✓", "completed successfully", "Successfully uploaded"]):
                        formatted_output.append(f"✅ {line}")
                    elif any(error_indicator in line for error_indicator in ["✗", "error", "Error", "failed"]):
                        formatted_output.append(f"❌ {line}")
                    elif any(progress_indicator in line for progress_indicator in ["Processing batch", "Fetching", "Analyzing"]):
                        formatted_output.append(f"🔄 {line}")
                    elif "Found" in line and "transactions" in line:
                        formatted_output.append(f"📊 {line}")
                    else:
                        formatted_output.append(f"   {line}")
                
                # Display with scrolling
                st.code("\n".join(formatted_output), language=None)
        
        # Auto-refresh while running
        if st.session_state.analysis_status == "running":
            time.sleep(0.5)  # Refresh every 0.5 seconds for real-time feel
            st.rerun()
    
    elif st.session_state.analysis_status == "completed":
        st.success("Analysis completed successfully!")
        if st.session_state.analysis_start_time:
            total_time = time.time() - st.session_state.analysis_start_time
            st.text(f"Total time: {total_time:.0f} seconds")
            
        # Show completion summary
        with st.expander("Analysis Summary", expanded=True):
            total_operations = len(st.session_state.analysis_output)
            success_operations = len([l for l in st.session_state.analysis_output if "✓" in l or "completed successfully" in l.lower()])
            error_operations = len([l for l in st.session_state.analysis_output if "✗" in l or "error" in l.lower()])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Operations", total_operations)
            col2.metric("Successful", success_operations)
            col3.metric("Errors", error_operations)
    
    elif st.session_state.analysis_status == "failed":
        st.error("Analysis failed or was stopped")
    
    # Show output if available
    if st.session_state.analysis_output:
        with st.expander("Analysis Output", expanded=(st.session_state.analysis_status in ["completed", "failed"])):
            show_all = st.checkbox("Show all output", value=True, key="show_all_final")
            
            if show_all:
                display_lines = st.session_state.analysis_output
            else:
                display_lines = st.session_state.analysis_output[-100:] if len(st.session_state.analysis_output) > 100 else st.session_state.analysis_output
            
            # Format output
            formatted_output = []
            for line in display_lines:
                if any(success_indicator in line for success_indicator in ["✓", "completed successfully", "Successfully uploaded"]):
                    formatted_output.append(f"✅ {line}")
                elif any(error_indicator in line for error_indicator in ["✗", "error", "Error", "failed"]):
                    formatted_output.append(f"❌ {line}")
                elif any(progress_indicator in line for progress_indicator in ["Processing batch", "Fetching", "Analyzing"]):
                    formatted_output.append(f"🔄 {line}")
                elif "Found" in line and "transactions" in line:
                    formatted_output.append(f"📊 {line}")
                else:
                    formatted_output.append(f"   {line}")
            
            st.code("\n".join(formatted_output), language=None)
        
        # Download option
        log_content = "\n".join(st.session_state.analysis_output)
        st.download_button(
            label="Download Analysis Log",
            data=log_content,
            file_name=f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key="download_final_log"
        )

def dashboard_tab():
    """
    Dashboard functionality
    """
    st.header("Validator Analysis Dashboard")
    
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
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Smart Contract Distribution")
        if 'is_smart_contract' in filtered_df.columns:
            contract_counts = filtered_df['is_smart_contract'].value_counts()
            fig_pie = px.pie(
                values=contract_counts.values,
                names=['Non-Deployers' if not x else 'Smart Contract Deployers' for x in contract_counts.index],
                title="Smart Contract Status"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
 = px.pie(
                values=dex_counts.values,
                names=['Non-DEX' if not x else 'DEX Addresses' for x in dex_counts.index],
                title="DEX Status"
            )
            st.plotly_chart(fig_dex, use_container_width=True)
    
    # Data table
    st.subheader("Validator Data")
    
    # Show key columns by default
    key_columns = []
    for col in ['index', 'pubkey', 'deposit_address', 'last_transaction_time', 'is_smart_contract', 'is_dex']:
        if col in filtered_df.columns:
            key_columns.append(col)
    
    display_df = filtered_df[key_columns] if key_columns else filtered_df
    
    # Format datetime
    if 'last_transaction_time' in display_df.columns:
        display_df = display_df.copy()
        display_df['last_transaction_time'] = pd.to_datetime(display_df['last_transaction_time']).dt.strftime('%Y-%m-%d %H:%M')
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Download section
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
    
    st.title("Validator Analysis Platform")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Run Analysis", "Dashboard"])
    
    with tab1:
        analysis_tab()
    
    with tab2:
        dashboard_tab()

if __name__ == "__main__":
    main()