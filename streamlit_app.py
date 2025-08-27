import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from supabase import create_client, Client
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
        tuple: (pd.DataFrame, dict) - (Validator data, metadata)
    """
    config = Config()
    
    try:
        # Initialize Supabase client
        supabase: Client = create_client(config.supabase_url, config.supabase_key)
        
        # Paginated approach
        all_data = []
        page_size = 5000
        offset = 0
        
        while True:
            response = supabase.table(config.table_name).select("*").range(offset, offset + page_size - 1).execute()
            
            if not response.data:
                break
                
            all_data.extend(response.data)
            
            # If we got less than page_size records, we've reached the end
            if len(response.data) < page_size:
                break
                
            offset += page_size
        
        # Create a mock response object for consistency
        class MockResponse:
            def __init__(self, data):
                self.data = data
        
        response = MockResponse(all_data)
        
        metadata = {
            'table_name': config.table_name,
            'record_count': len(response.data) if response.data else 0,
            'success': True,
            'error': None
        }
        
        if response.data:
            df = pd.DataFrame(response.data)
            
            # Convert datetime columns
            datetime_columns = ['last_transaction_time', 'created_at', 'updated_at']
            for col in datetime_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            return df, metadata
        else:
            return pd.DataFrame(), metadata
            
    except Exception as e:
        metadata = {
            'table_name': config.table_name,
            'record_count': 0,
            'success': False,
            'error': str(e)
        }
        return pd.DataFrame(), metadata

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
        
        # Use the fixed script name from the artifact
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
    if st.button("Go to Dashboard", use_container_width=True):
        # Clear cache and indicate dashboard should be viewed
        st.cache_data.clear()
        st.session_state.force_dashboard = True
        st.rerun()
    
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
    col1, col2 = st.columns([1, 1])
    
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
                    st.session_state.analysis_output.append(f"[{completion_time}] Analysis completed successfully!")
                else:
                    st.session_state.analysis_status = "failed"
                    error_time = datetime.now().strftime('%H:%M:%S')
                    exit_code = process.poll() if process else "Unknown"
                    st.session_state.analysis_output.append(f"[{error_time}] Analysis failed with exit code: {exit_code}")
                
                st.session_state.analysis_process = None
                st.rerun()
        
        # Show progress info
        with progress_container:
            if st.session_state.analysis_start_time:
                elapsed = time.time() - st.session_state.analysis_start_time
                st.text(f"Elapsed time: {elapsed:.0f} seconds")
            
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
            
            st.text(f"Current step: {current_step}")
        
        # Auto-refresh every 0.5 seconds while running for more responsive updates
        time.sleep(0.5)
        st.rerun()
    
    elif status == "completed":
        st.success("Analysis completed successfully!")
        if st.session_state.analysis_start_time:
            total_time = time.time() - st.session_state.analysis_start_time
            st.text(f"Total time: {total_time:.0f} seconds")
        
        # Add dashboard button when analysis is complete
        st.markdown("---")
        if st.button("View Dashboard", type="primary", use_container_width=True):
            # Clear data cache to force refresh
            st.cache_data.clear()
            # Set flag to show dashboard message
            st.session_state.force_dashboard = True
            st.rerun()
    
    elif status == "failed":
        st.error("Analysis failed or was stopped")
    
    # Show output if available
    if st.session_state.analysis_output:
        with st.expander("Analysis Output", expanded=(status == "running")):
            # Show last 200 lines to keep it manageable but comprehensive
            display_lines = st.session_state.analysis_output[-200:] if len(st.session_state.analysis_output) > 200 else st.session_state.analysis_output
            
            # Format output with better status indicators
            formatted_output = []
            for line in display_lines:
                line_lower = line.lower()
                if any(success_word in line_lower for success_word in ["completed successfully", "successfully uploaded", "success", "saved", "finished"]):
                    formatted_output.append(f"✅ {line}")
                elif any(error_word in line_lower for error_word in ["error", "failed", "exception", "timeout"]):
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
                label="Download Analysis Log",
                data=log_content,
                file_name=f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_log"
            )

def dashboard_tab():
    """
    Dashboard functionality with refresh button
    """
    st.header("Validator Analysis Dashboard")
    
    # Add refresh button at the top of dashboard
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.success("Data refreshed! Loading latest data from database...")
            st.rerun()
    
    # Load data
    with st.spinner("Loading data from Supabase..."):
        df, metadata = load_data_from_supabase()
    
    # Show connection status
    if metadata['success']:
        st.success(f"Successfully loaded {metadata['record_count']} records from table '{metadata['table_name']}'")
    else:
        st.error(f"Failed to load data: {metadata['error']}")
        
        # Show debugging information
        with st.expander("Debug Information"):
            config = Config()
            st.write("**Configuration:**")
            st.write(f"- URL: {config.supabase_url[:50]}..." if config.supabase_url else "- URL: Not set")
            st.write(f"- Key: {config.supabase_key[:20]}..." if config.supabase_key else "- Key: Not set") 
            st.write(f"- Table: {metadata['table_name']}")
            st.write("**Error Details:**")
            st.code(metadata['error'])
    
    if df.empty:
        st.warning("No data available. Please run the validator analysis first or check your Supabase connection.")
        
        # Show connection help
        with st.expander("Troubleshooting"):
            st.markdown("""
            **Possible issues:**
            1. **No analysis has been run yet** - Go to the 'Run Analysis' tab and complete an analysis first
            2. **Database connection issue** - Check your Supabase credentials in environment variables
            3. **Empty table** - The analysis may not have uploaded data successfully
            
            **Environment variables needed:**
            - `SUPABASE_URL`
            - `SUPABASE_KEY`
            - `SUPABASE_TABLE_NAME` (defaults to 'validator_data')
            """)
        return
    
    # Show data summary
    if not df.empty:
        st.info(f"Displaying data from {len(df)} validator records from database")
    
    # Sidebar filters - Combined Activity Filter
    st.sidebar.header("Filters")
    
    activity_filter = st.sidebar.selectbox(
        "Validator Activity",
        ["All", "Active after Shapella (Apr 12, 2023)", "Active after Merge (Sep 15, 2022)"]
    )
    
    # Apply activity filters
    filtered_df = df.copy()
    
    if 'last_transaction_time' in df.columns:
        if activity_filter == "Active after Shapella (Apr 12, 2023)":
            shapella_date = pd.Timestamp('2023-04-12')
            filtered_df = filtered_df[filtered_df['last_transaction_time'] > shapella_date]
        elif activity_filter == "Active after Merge (Sep 15, 2022)":
            merge_date = pd.Timestamp('2022-09-15')
            filtered_df = filtered_df[filtered_df['last_transaction_time'] > merge_date]
    
    # Calculate validator status (active/inactive)
    if 'state' in filtered_df.columns:
        active_validators = len(filtered_df[filtered_df['state'].isin(['active_ongoing', 'active_exiting', 'active_slashed'])])
        inactive_validators = len(filtered_df[~filtered_df['state'].isin(['active_ongoing', 'active_exiting', 'active_slashed'])])
    else:
        # Fallback if status column doesn't exist
        active_validators = len(filtered_df)
        inactive_validators = 0
    
    # Calculate unique deposit addresses statistics
    unique_deposit_addresses = filtered_df['deposit_address'].nunique() if 'deposit_address' in filtered_df.columns else 0
    
    # Get unique deposit address breakdown
    if 'deposit_address' in filtered_df.columns:
        unique_addresses_df = filtered_df[['deposit_address', 'is_smart_contract', 'is_dex']].drop_duplicates(subset=['deposit_address'])
        
        # Count by category
        if 'is_smart_contract' in unique_addresses_df.columns and 'is_dex' in unique_addresses_df.columns:
            from_dex = unique_addresses_df['is_dex'].sum()
            from_smart_contract = unique_addresses_df[
                (unique_addresses_df['is_smart_contract'] == True) & 
                (unique_addresses_df['is_dex'] == False)
            ].shape[0]
            from_wallet = unique_addresses_df[
                (unique_addresses_df['is_smart_contract'] == False) & 
                (unique_addresses_df['is_dex'] == False)
            ].shape[0]
        else:
            from_dex = 0
            from_smart_contract = 0
            from_wallet = unique_deposit_addresses
    else:
        from_dex = 0
        from_smart_contract = 0
        from_wallet = 0
    
    # Validator Status Score Cards (3 cards)
    st.subheader("Validator Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Validators", len(filtered_df))
    
    with col2:
        st.metric("Active Validators", active_validators)
    
    with col3:
        st.metric("Inactive Validators", inactive_validators)
    
    st.markdown("---")
    
    # Deposit Address Source Score Cards (4 cards)
    st.subheader("Deposit Address Sources")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Unique Addresses", unique_deposit_addresses)
    
    with col2:
        st.metric("From Wallet", from_wallet)
    
    with col3:
        st.metric("From Smart Contract", from_smart_contract)
    
    with col4:
        st.metric("From DEX", from_dex)
    
    st.markdown("---")
    
    # Data table
    st.subheader("Validator Data")
    
    # Show available columns
    available_columns = list(filtered_df.columns)
    st.caption(f"Available columns: {', '.join(available_columns)}")
    
    # Show key columns by default if they exist
    key_columns = []
    preferred_columns = ['index', 'pubkey', 'status', 'deposit_address', 'last_transaction_time', 'is_smart_contract', 'is_dex']
    
    for col in preferred_columns:
        if col in filtered_df.columns:
            key_columns.append(col)
    
    # Handle both 'status' and 'state' column names (fallback for different API responses)
    if 'status' not in key_columns:
        if 'state' in filtered_df.columns:
            key_columns.insert(2, 'state')  # Insert at position where 'status' would be
        elif 'status' in filtered_df.columns:
            key_columns.insert(2, 'status')
    
    # If no preferred columns found, show all
    if not key_columns:
        key_columns = available_columns
    
    display_df = filtered_df[key_columns] if key_columns else filtered_df
    
    # Format datetime columns
    datetime_columns = ['last_transaction_time', 'created_at', 'updated_at']
    for col in datetime_columns:
        if col in display_df.columns:
            display_df = display_df.copy()
            display_df[col] = pd.to_datetime(display_df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
    
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
    
    # Show message if user clicked dashboard button
    if 'force_dashboard' in st.session_state and st.session_state.force_dashboard:
        st.info("**Dashboard Updated!** Click on the 'Dashboard' tab above to view the latest data.")
        st.session_state.force_dashboard = False
    
    # Create tabs
    tab1, tab2 = st.tabs(["Run Analysis", "Dashboard"])
    
    with tab1:
        analysis_tab()
    
    with tab2:
        dashboard_tab()

if __name__ == "__main__":
    main()