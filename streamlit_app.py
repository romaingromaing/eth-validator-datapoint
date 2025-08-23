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

def start_analysis_subprocess():
    """
    Start the validator analysis as a background subprocess
    Returns the process object
    """
    try:
        # Set environment variable to force unbuffered output
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            [sys.executable, "-u", "validator_analysis.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=0,  # Unbuffered for real-time output
            env=env,
            cwd=os.getcwd()
        )
        return process
    except Exception as e:
        st.error(f"Failed to start analysis process: {e}")
        return None

def read_process_output_thread(process, output_queue):
    """
    Thread function to read process output and put it in a queue
    """
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                # Strip the line but preserve it even if it's just whitespace/formatting
                clean_line = line.rstrip('\n\r')
                if clean_line or line.strip():  # Keep lines with content or meaningful whitespace
                    output_queue.put(clean_line)
        
        # Signal that the process has finished
        output_queue.put("__PROCESS_FINISHED__")
    except Exception as e:
        output_queue.put(f"Error reading output: {e}")
        output_queue.put("__PROCESS_FINISHED__")

def get_new_output_lines(output_queue):
    """
    Get all available lines from the output queue without blocking
    """
    lines = []
    process_finished = False
    
    try:
        while True:
            try:
                line = output_queue.get_nowait()
                if line == "__PROCESS_FINISHED__":
                    process_finished = True
                    break
                else:  # Add all lines, even empty ones for formatting
                    lines.append(line)
            except queue.Empty:
                break
    except Exception:
        pass
    
    return lines, process_finished

def analysis_tab():
    """
    Enhanced tab for running the validator analysis with real-time output
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
    
    # Add manual dashboard access button
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📊 Go to Dashboard", use_container_width=True):
            # Clear cache and indicate dashboard should be viewed
            st.cache_data.clear()
            st.session_state.force_dashboard = True
            st.rerun()
    with col2:
        # Refresh data button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.success("Data cache cleared. Next dashboard view will fetch fresh data.")
    
    st.markdown("---")
    
    # Initialize session state
    if 'analysis_process' not in st.session_state:
        st.session_state.analysis_process = None
    if 'analysis_output' not in st.session_state:
        st.session_state.analysis_output = []
    if 'analysis_status' not in st.session_state:
        st.session_state.analysis_status = "idle"  # idle, running, completed, failed
    if 'analysis_start_time' not in st.session_state:
        st.session_state.analysis_start_time = None
    if 'output_queue' not in st.session_state:
        st.session_state.output_queue = None
    if 'output_thread' not in st.session_state:
        st.session_state.output_thread = None
    
    # Current status
    process = st.session_state.analysis_process
    status = st.session_state.analysis_status
    
    # Control buttons
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        start_disabled = (status == "running")
        if st.button("Start Analysis", type="primary", use_container_width=True, disabled=start_disabled):
            # Clean up any existing process
            if st.session_state.analysis_process:
                try:
                    st.session_state.analysis_process.terminate()
                    st.session_state.analysis_process.wait(timeout=3)
                except:
                    try:
                        st.session_state.analysis_process.kill()
                    except:
                        pass
            
            # Clean up existing thread
            if st.session_state.output_thread and st.session_state.output_thread.is_alive():
                # Thread will stop when process terminates
                pass
            
            # Start new analysis
            st.session_state.analysis_process = start_analysis_subprocess()
            st.session_state.analysis_output = []
            st.session_state.analysis_status = "running"
            st.session_state.analysis_start_time = time.time()
            
            # Create output queue and thread
            if st.session_state.analysis_process:
                st.session_state.output_queue = queue.Queue()
                st.session_state.output_thread = threading.Thread(
                    target=read_process_output_thread,
                    args=(st.session_state.analysis_process, st.session_state.output_queue),
                    daemon=True
                )
                st.session_state.output_thread.start()
            
            st.rerun()
    
    with col2:
        stop_disabled = (status != "running")
        if st.button("Stop Analysis", use_container_width=True, disabled=stop_disabled):
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
            st.session_state.analysis_process = None
            st.rerun()
    
    with col3:
        if st.button("Reset & Clear", use_container_width=True):
            # Clean up process
            if st.session_state.analysis_process:
                try:
                    st.session_state.analysis_process.terminate()
                    st.session_state.analysis_process.wait(timeout=3)
                except:
                    try:
                        st.session_state.analysis_process.kill()
                    except:
                        pass
            
            # Reset all state
            st.session_state.analysis_process = None
            st.session_state.analysis_output = []
            st.session_state.analysis_status = "idle"
            st.session_state.analysis_start_time = None
            st.session_state.output_queue = None
            st.session_state.output_thread = None
            st.cache_data.clear()
            st.rerun()
    
    # Status display and real-time output handling
    if status == "running":
        st.info("Analysis is currently running...")
        
        # Create containers for status and output
        progress_container = st.container()
        output_container = st.container()
        
        # Check if we have new output
        if st.session_state.output_queue:
            new_lines, process_finished = get_new_output_lines(st.session_state.output_queue)
            
            # Add new lines to output
            if new_lines:
                for line in new_lines:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    formatted_line = f"[{timestamp}] {line}"
                    st.session_state.analysis_output.append(formatted_line)
                
                # Keep output manageable (last 2000 lines)
                if len(st.session_state.analysis_output) > 2000:
                    st.session_state.analysis_output = st.session_state.analysis_output[-1500:]
            
            # Check if process finished
            if process_finished or (process and process.poll() is not None):
                if process and process.poll() == 0:
                    st.session_state.analysis_status = "completed"
                    completion_time = datetime.now().strftime('%H:%M:%S')
                    st.session_state.analysis_output.append(f"[{completion_time}] ✅ Analysis completed successfully!")
                else:
                    st.session_state.analysis_status = "failed"
                    error_time = datetime.now().strftime('%H:%M:%S')
                    exit_code = process.poll() if process else "Unknown"
                    st.session_state.analysis_output.append(f"[{error_time}] ❌ Analysis failed with exit code: {exit_code}")
                
                st.session_state.analysis_process = None
                st.rerun()
        
        # Show progress info
        with progress_container:
            if st.session_state.analysis_start_time:
                elapsed = time.time() - st.session_state.analysis_start_time
                st.text(f"⏱️ Elapsed time: {elapsed:.0f} seconds")
            
            # Parse current step from recent output
            recent_output = st.session_state.analysis_output[-10:] if st.session_state.analysis_output else []
            current_step = "Starting analysis..."
            
            for line in reversed(recent_output):
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in [
                    "processing batch", "fetching", "analyzing", "uploading", 
                    "saving", "loading", "connecting", "querying"
                ]):
                    # Extract the step description
                    if "] " in line:
                        current_step = line.split("] ", 1)[-1]
                    else:
                        current_step = line
                    break
            
            st.text(f"🔄 Current step: {current_step}")
        
        # Auto-refresh every 0.5 seconds while running for more responsive updates
        time.sleep(0.5)
        st.rerun()
    
    elif status == "completed":
        st.success("✅ Analysis completed successfully!")
        if st.session_state.analysis_start_time:
            total_time = time.time() - st.session_state.analysis_start_time
            st.text(f"⏱️ Total time: {total_time:.0f} seconds")
        
        # Add dashboard button when analysis is complete
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📊 View Dashboard", type="primary", use_container_width=True):
                # Clear data cache to force refresh
                st.cache_data.clear()
                # Set flag to show dashboard message
                st.session_state.force_dashboard = True
                st.rerun()
    
    elif status == "failed":
        st.error("❌ Analysis failed or was stopped")
    
            # Show output if available
    if st.session_state.analysis_output:
        with st.expander("Analysis Output", expanded=(status == "running")):
            # Show last 200 lines to keep it manageable but comprehensive
            display_lines = st.session_state.analysis_output[-200:] if len(st.session_state.analysis_output) > 200 else st.session_state.analysis_output
            
            # Format output with better status indicators
            formatted_output = []
            for line in display_lines:
                line_lower = line.lower()
                if any(success_word in line_lower for success_word in ["✓", "completed successfully", "successfully uploaded", "success", "saved", "finished"]):
                    formatted_output.append(f"✅ {line}")
                elif any(error_word in line_lower for error_word in ["✗", "error", "failed", "exception", "timeout"]):
                    formatted_output.append(f"❌ {line}")
                elif any(progress_word in line_lower for progress_word in ["processing", "fetching", "analyzing", "loading", "getting", "found"]):
                    formatted_output.append(f"🔄 {line}")
                elif any(info_word in line_lower for info_word in ["starting", "total", "batch", "query"]):
                    formatted_output.append(f"ℹ️  {line}")
                else:
                    formatted_output.append(f"   {line}")
            
            # Create a scrollable text area for better viewing
            output_text = "\n".join(formatted_output)
            st.text_area(
                "Live Output",
                value=output_text,
                height=500,
                key=f"output_display_{status}_{len(st.session_state.analysis_output)}",
                disabled=True
            )
            
            # Show total lines info
            st.caption(f"Showing last {len(display_lines)} of {len(st.session_state.analysis_output)} total lines")
        
        # Download option
        if st.session_state.analysis_output:
            log_content = "\n".join(st.session_state.analysis_output)
            st.download_button(
                label="📥 Download Analysis Log",
                data=log_content,
                file_name=f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_log"
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
    
    with col2:
        st.subheader("DEX Address Distribution")
        if 'is_dex' in filtered_df.columns:
            dex_counts = filtered_df['is_dex'].value_counts()
            fig_dex = px.pie(
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
    
    st.title("🔍 Validator Analysis Platform")
    
    # Initialize active tab in session state  
    if 'force_dashboard' not in st.session_state:
        st.session_state.force_dashboard = False
    
    # Show message if user clicked dashboard button
    if st.session_state.force_dashboard:
        st.info("💡 **Dashboard Updated!** Click on the 'Dashboard' tab above to view the latest data.")
        st.session_state.force_dashboard = False
    
    # Create tabs normally
    tab1, tab2 = st.tabs(["🚀 Run Analysis", "📊 Dashboard"])
    
    with tab1:
        analysis_tab()
    
    with tab2:
        dashboard_tab()

if __name__ == "__main__":
    main()