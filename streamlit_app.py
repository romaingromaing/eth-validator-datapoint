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
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import atexit

# Load environment variables from .env file
load_dotenv()

# Import verification functions
from verify_keystore_signature import validate_bls_to_execution_change_keystore
from verify_signatures import verify_json

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
        
        # Store validation status instead of stopping the app
        self.is_valid = bool(self.supabase_url and self.supabase_key)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data_from_supabase():
    """
    Load validator data from Supabase
    
    Returns:
        tuple: (pd.DataFrame, dict) - (Validator data, metadata)
    """
    config = Config()
    
    # Check if config is valid first
    if not config.is_valid:
        metadata = {
            'table_name': config.table_name,
            'record_count': 0,
            'success': False,
            'error': 'Missing SUPABASE_URL or SUPABASE_KEY in configuration'
        }
        return pd.DataFrame(), metadata
    
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

def initialize_admin_auth():
    """
    Initialize admin authentication state
    Returns: bool - True if user has admin access
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
    
    # Initialize session state
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    # Method 1: Simple password check
    admin_password = get_secret('ADMIN_PASSWORD')
    if admin_password:
        return st.session_state.admin_authenticated
    
    # Method 2: IP-based access (fallback)
    allowed_ips = get_secret('ALLOWED_IPS')  # Comma-separated list of IPs
    if allowed_ips:
        # Note: This is limited in cloud deployments due to proxies
        try:
            import socket
            user_ip = st.context.headers.get('x-forwarded-for', '').split(',')[0].strip()
            if not user_ip:
                user_ip = st.context.headers.get('x-real-ip', '')
            allowed_ip_list = [ip.strip() for ip in allowed_ips.split(',')]
            return user_ip in allowed_ip_list
        except:
            pass
    
    # Method 3: Environment-based (default to admin if no restrictions set)
    restrict_access = get_secret('RESTRICT_ADMIN_ACCESS', 'false').lower() == 'true'
    return not restrict_access

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

def get_last_refresh_date(df):
    """
    Get the last data refresh date from the created_at column
    Returns: (datetime, str) - (datetime object, formatted string)
    """
    if df.empty or 'created_at' not in df.columns:
        return None, "Unknown"
    
    try:
        # Get the most recent created_at timestamp
        last_refresh = df['created_at'].max()
        if pd.isna(last_refresh):
            return None, "Unknown"
        
        # Format for display
        formatted_date = last_refresh.strftime('%Y-%m-%d %H:%M:%S UTC')
        return last_refresh, formatted_date
    except Exception:
        return None, "Unknown"

def normalize_state(state_value):
    """
    Normalize state values to simplified categories
    """
    if pd.isna(state_value):
        return "Inactive"
    
    state_str = str(state_value).lower()
    
    # Map to simplified states
    if state_str in ['active_online', 'exiting_online', 'active_exiting']:
        return "Active"
    elif state_str in ['exited', 'exited_unslashed']:
        return "Exited"
    elif state_str == 'exited_slashed':
        return "Slashed"
    elif state_str == 'confirmed_lost' or 'lost' in state_str:
        return "Confirmed Lost"
    else:
        return "Inactive"

def dashboard_tab():
    """
    Dashboard functionality with refresh button and last refresh date
    """
    # Header with last refresh on same line
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Validator Analysis Dashboard")
    
    # Load data first to get refresh date
    with st.spinner("Loading data from Supabase..."):
        df, metadata = load_data_from_supabase()
    
    # Get last refresh date
    last_refresh_dt, last_refresh_str = get_last_refresh_date(df)
    
    with col2:
        st.info(f"📅 Last refresh: {last_refresh_str}")
    
    # Check admin access using session state (no duplicate widgets)
    has_admin_access = st.session_state.get('admin_authenticated', False)
    
    # Check if we have configuration issues
    config = Config()

    if not config.is_valid:
        st.error("⚠️ Missing SUPABASE_URL or SUPABASE_KEY in configuration")
        st.info("""
        **For local development:** Make sure your `.env` file contains:
        ```
        SUPABASE_URL=your_supabase_url
        SUPABASE_KEY=your_supabase_key
        SUPABASE_DATABASE_URL=your_database_connection_string
        ```
        
        **For Streamlit Cloud:** Add these to your app's Secrets in the dashboard.
        """)
        return
    
    # Admin refresh button
    if has_admin_access:
        if st.button("Refresh Data", width=False):
            st.cache_data.clear()
            st.success("Data refreshed!")
            st.rerun()
    
    # Show connection error details if needed
    if not metadata['success']:
        st.error(f"Failed to load data: {metadata['error']}")
        
        # Show debugging information
        with st.expander("Debug Information"):
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
    
    # Normalize state column
    if 'state' in df.columns:
        df['state'] = df['state'].apply(normalize_state)
    elif 'status' in df.columns:
        df['state'] = df['status'].apply(normalize_state)
    
    # Sidebar filters - Combined Activity Filter
    st.sidebar.header("Filters")

    # Search functionality
    st.sidebar.markdown("---")
    st.sidebar.subheader("Search")
    search_term = st.sidebar.text_input(
        "Search by Public Key or Deposit Address",
        placeholder="Enter pubkey or deposit address...",
        help="Search for specific validators by their public key or deposit address"
    )

    if search_term:
        if st.sidebar.button("Clear Search", width=True):
            st.rerun()
    
    activity_filter = st.sidebar.selectbox(
        "Validator Activity",
        ["All", "Active after Shapella (Apr 12, 2023)", "Active after Merge (Sep 15, 2022)"]
    )
    
    # Add validator status filter
    status_filter = st.sidebar.selectbox(
        "Validator Status",
        ["All", "Active", "Inactive", "Exited", "Slashed", "Confirmed Lost"]
    )
    
    # Apply activity filters
    filtered_df = df.copy()

    # Apply search filter
    if search_term:
        search_term_lower = search_term.lower()
        search_mask = False

        # Search in pubkey column
        if 'pubkey' in filtered_df.columns:
            search_mask |= filtered_df['pubkey'].str.lower().str.contains(search_term_lower, na=False)

        # Search in deposit_address column
        if 'deposit_address' in filtered_df.columns:
            search_mask |= filtered_df['deposit_address'].str.lower().str.contains(search_term_lower, na=False)

        filtered_df = filtered_df[search_mask]

        # Show search results info
        if len(filtered_df) == 0:
            st.warning(f"No results found for '{search_term}'")
        else:
            st.info(f"Found {len(filtered_df)} validator(s) matching '{search_term}'")
    
    if 'last_transaction_time' in df.columns:
        if activity_filter == "Active after Shapella (Apr 12, 2023)":
            shapella_date = pd.Timestamp('2023-04-12')
            filtered_df = filtered_df[filtered_df['last_transaction_time'] > shapella_date]
        elif activity_filter == "Active after Merge (Sep 15, 2022)":
            merge_date = pd.Timestamp('2022-09-15')
            filtered_df = filtered_df[filtered_df['last_transaction_time'] > merge_date]
    
    # Apply status filter
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df['state'] == status_filter]

    # Calculate validator status (active/inactive)
    active_validators = len(filtered_df[filtered_df['state'] == 'Active'])
    inactive_validators = len(filtered_df[filtered_df['state'] == 'Inactive'])
    
    # Calculate unique deposit addresses statistics
    unique_deposit_addresses = filtered_df['deposit_address'].nunique() if 'deposit_address' in filtered_df.columns else 0
    
    # Overview section
    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Validators", len(filtered_df))
    
    with col2:
        st.metric("Active Validators", active_validators)
    
    with col3:
        st.metric("Inactive Validators", inactive_validators)
    
    with col4:
        st.metric("Total Unique Addresses", unique_deposit_addresses)
    
    st.markdown("---")
    
    # Data table
    st.subheader("Validator Data")
    
    # Show available columns
    available_columns = list(filtered_df.columns)
    st.caption(f"Available columns: {', '.join(available_columns)}")
    
    # Show key columns by default if they exist
    key_columns = []
    preferred_columns = ['index', 'pubkey', 'state', 'deposit_address', 'operator', 'last_transaction_time', 'to_execution_address']
    
    for col in preferred_columns:
        if col in filtered_df.columns:
            key_columns.append(col)
    
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
    
    st.dataframe(display_df, width=True, height=400)
    
    # Download section
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"validator_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


class SchedulerManager:
    def __init__(self):
        self.scheduler = None
        self.is_running = False
    
    def get_secret(self, key, default=None):
        """Helper to get secrets from Streamlit or environment"""
        try:
            if hasattr(st, 'secrets') and st.secrets is not None:
                return st.secrets.get(key, default)
        except Exception:
            pass
        return os.getenv(key, default)
    
    def should_enable_scheduler(self):
        """Check if scheduler should be enabled based on configuration"""
        enable_scheduler = self.get_secret('ENABLE_AUTO_ANALYSIS', 'false').lower() == 'true'
        has_required_vars = all([
            self.get_secret('SUPABASE_URL'),
            self.get_secret('SUPABASE_KEY'),
            self.get_secret('DUNE_SIM_API_KEY'),
            self.get_secret('DUNE_CLIENT_API_KEY')
        ])
        return enable_scheduler and has_required_vars
    
    def run_scheduled_analysis(self):
        """Run the analysis in background"""
        try:
            # Log the scheduled run
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Store the run status
            if 'scheduled_runs' not in st.session_state:
                st.session_state.scheduled_runs = []
            
            run_info = {
                'timestamp': timestamp,
                'status': 'started'
            }
            
            # Start the analysis subprocess
            import subprocess
            import sys
            
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['SCHEDULED_RUN'] = 'true'  # Flag to indicate this is a scheduled run
            env['CLEAR_EXISTING_DATA'] = 'true'
            
            process = subprocess.Popen(
                [sys.executable, "-u", "validator_analysis.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env,
                cwd=os.getcwd()
            )
            
            # Wait for completion (with timeout)
            try:
                stdout, stderr = process.communicate(timeout=3600)  # 1 hour timeout
                if process.returncode == 0:
                    run_info['status'] = 'completed'
                    run_info['message'] = 'Analysis completed successfully'
                else:
                    run_info['status'] = 'failed'
                    run_info['message'] = f'Analysis failed with code {process.returncode}'
            except subprocess.TimeoutExpired:
                process.kill()
                run_info['status'] = 'timeout'
                run_info['message'] = 'Analysis timed out after 1 hour'
            
            # Store the result
            st.session_state.scheduled_runs.append(run_info)
            
            # Keep only last 10 runs
            if len(st.session_state.scheduled_runs) > 10:
                st.session_state.scheduled_runs = st.session_state.scheduled_runs[-10:]
                
        except Exception as e:
            run_info['status'] = 'error'
            run_info['message'] = f'Error: {str(e)}'
            if 'scheduled_runs' not in st.session_state:
                st.session_state.scheduled_runs = []
            st.session_state.scheduled_runs.append(run_info)
    
    def start_scheduler(self):
        """Start the background scheduler"""
        if not self.should_enable_scheduler():
            return False
        
        if self.scheduler is None:
            self.scheduler = BackgroundScheduler()
            
            # Get cron settings from configuration
            cron_day = int(self.get_secret('CRON_DAY', '1'))  # 1st of month
            cron_hour = int(self.get_secret('CRON_HOUR', '2'))  # 2 AM
            cron_minute = int(self.get_secret('CRON_MINUTE', '0'))  # 0 minutes
            
            # Add monthly job
            self.scheduler.add_job(
                func=self.run_scheduled_analysis,
                trigger=CronTrigger(
                    day=cron_day,
                    hour=cron_hour, 
                    minute=cron_minute
                ),
                id='monthly_analysis',
                name='Monthly Validator Analysis',
                replace_existing=True
            )
            
            self.scheduler.start()
            self.is_running = True
            
            # Register cleanup
            atexit.register(lambda: self.scheduler.shutdown() if self.scheduler else None)
        
        return True
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        if self.scheduler and self.is_running:
            self.scheduler.shutdown()
            self.scheduler = None
            self.is_running = False
    
    def get_next_run_time(self):
        """Get the next scheduled run time"""
        if self.scheduler and self.is_running:
            jobs = self.scheduler.get_jobs()
            if jobs:
                return jobs[0].next_run_time
        return None
    
    def trigger_manual_run(self):
        """Manually trigger a scheduled analysis"""
        if self.should_enable_scheduler():
            # Run in a separate thread to avoid blocking
            thread = threading.Thread(target=self.run_scheduled_analysis, daemon=True)
            thread.start()
            return True
        return False

# Global scheduler instance
if 'scheduler_manager' not in st.session_state:
    st.session_state.scheduler_manager = SchedulerManager()

def admin_tab():
    """
    Combined Admin tab with analysis and scheduler functionality
    """
    st.header("Admin Panel")
    
    # Admin login section - now in the tab itself
    def get_secret(key):
        try:
            if hasattr(st, 'secrets') and st.secrets is not None:
                return st.secrets.get(key)
        except Exception:
            pass
        import os
        return os.getenv(key)
    
    # Initialize session state
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    admin_password = get_secret('ADMIN_PASSWORD')
    
    # Show login form if not authenticated
    if not st.session_state.admin_authenticated and admin_password:
        st.subheader("🔒 Admin Access Required")
        password_input = st.text_input("Admin Password", type="password", key="admin_password_input")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Login", key="admin_login_button"):
                if password_input == admin_password:
                    st.session_state.admin_authenticated = True
                    st.success("Admin access granted!")
                    st.rerun()
                else:
                    st.error("Invalid password")
        st.info("Admin access required for running analysis and managing the scheduler.")
        return
    
    # Show logout button if authenticated
    if st.session_state.admin_authenticated:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success("🔓 Admin Access Granted")
        with col2:
            if st.button("Logout", key="admin_logout_button"):
                st.session_state.admin_authenticated = False
                st.rerun()
    
    # If no password is set or user is authenticated, show admin content
    if not admin_password or st.session_state.admin_authenticated:
        # Create sub-sections using expanders
        with st.expander("📊 Run Validator Analysis", expanded=True):
            run_analysis_section()
        
        st.markdown("---")
        
        with st.expander("⏰ Automated Scheduler", expanded=False):
            scheduler_section()

def run_analysis_section():
    """
    Analysis section (formerly analysis_tab content)
    """
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
    if st.button("Go to Dashboard", width=True):
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
        if st.button("Start Analysis", type="primary", width=True, disabled=start_disabled):
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
        if st.button("Stop Analysis", width=True, disabled=stop_disabled):
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
        if st.button("View Dashboard", type="primary", width=True):
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

def scheduler_section():
    """
    Scheduler section (formerly scheduler_tab content)
    """
    scheduler = st.session_state.scheduler_manager
    
    # Check if scheduler can be enabled
    can_enable = scheduler.should_enable_scheduler()
    
    if not can_enable:
        missing_items = []
        if not scheduler.get_secret('ENABLE_AUTO_ANALYSIS') == 'true':
            missing_items.append("ENABLE_AUTO_ANALYSIS=true")
        
        required_vars = ['SUPABASE_URL', 'SUPABASE_KEY', 'DUNE_SIM_API_KEY', 'DUNE_CLIENT_API_KEY']
        for var in required_vars:
            if not scheduler.get_secret(var):
                missing_items.append(var)
        
        st.error("❌ Scheduler cannot be enabled. Missing configuration:")
        for item in missing_items:
            st.write(f"- {item}")
        
        with st.expander("Configuration Guide"):
            st.markdown("""
            **Required Environment Variables/Secrets:**
            ```
            ENABLE_AUTO_ANALYSIS=true
            CRON_DAY=1          # Day of month (1-28, default: 1)
            CRON_HOUR=2         # Hour (0-23, default: 2 AM)
            CRON_MINUTE=0       # Minute (0-59, default: 0)
            
            # Plus all existing API keys:
            DUNE_SIM_API_KEY=your_key
            DUNE_CLIENT_API_KEY=your_key
            SUPABASE_URL=your_url
            SUPABASE_KEY=your_key
            ```
            """)
        return
    
    # Scheduler status
    st.success("✅ Scheduler configuration is valid")
    
    # Current settings
    cron_day = scheduler.get_secret('CRON_DAY', '1')
    cron_hour = scheduler.get_secret('CRON_HOUR', '2')
    cron_minute = scheduler.get_secret('CRON_MINUTE', '0')
    
    st.info(f"📅 **Schedule:** Monthly on day {cron_day} at {cron_hour}:{cron_minute:0>2}")
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not scheduler.is_running:
            if st.button("Start Scheduler", type="primary", width=True):
                if scheduler.start_scheduler():
                    st.success("Scheduler started!")
                    st.rerun()
                else:
                    st.error("Failed to start scheduler")
        else:
            if st.button("Stop Scheduler", width=True):
                scheduler.stop_scheduler()
                st.success("Scheduler stopped!")
                st.rerun()
    
    with col2:
        if scheduler.is_running:
            if st.button("Manual Run", width=True):
                if scheduler.trigger_manual_run():
                    st.success("Manual analysis started!")
                else:
                    st.error("Failed to start manual run")
    
    with col3:
        if st.button("Refresh Status", width=True):
            st.rerun()
    
    # Status display
    st.markdown("---")
    st.subheader("Status")
    
    if scheduler.is_running:
        st.success("🟢 Scheduler is running")
        
        next_run = scheduler.get_next_run_time()
        if next_run:
            st.info(f"⏰ **Next run:** {next_run.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    else:
        st.error("🔴 Scheduler is stopped")
    
    # Recent runs
    if 'scheduled_runs' in st.session_state and st.session_state.scheduled_runs:
        st.markdown("---")
        st.subheader("Recent Scheduled Runs")
        
        runs_df = pd.DataFrame(st.session_state.scheduled_runs)
        runs_df = runs_df.sort_values('timestamp', ascending=False)
        
        # Format the display
        for _, run in runs_df.iterrows():
            col1, col2, col3 = st.columns([1, 1, 3])
            
            with col1:
                st.text(run['timestamp'])
            
            with col2:
                status = run['status']
                if status == 'completed':
                    st.success(f"✅ {status}")
                elif status == 'failed' or status == 'error':
                    st.error(f"❌ {status}")
                elif status == 'timeout':
                    st.warning(f"⏰ {status}")
                else:
                    st.info(f"🔄 {status}")
            
            with col3:
                if 'message' in run:
                    st.text(run['message'])
    
    # Important notes
    st.markdown("---")
    st.subheader("Important Notes")
    
    st.warning("""
    **Deployment Considerations:**
    - ⚠️ **Streamlit Cloud**: Free tier may have limitations for long-running processes
    - 🔄 **App Restarts**: Scheduler will restart automatically when the app restarts
    - 💾 **Persistence**: Scheduled run history is stored in session state (temporary)
    - 🕐 **Timezone**: All times are in UTC
    - ⏱️ **Timeout**: Analysis jobs timeout after 1 hour
    """)

def flag_validator_as_lost(validator_index, to_execution_address):
    """
    Flag a validator as lost in the database
    """
    config = Config()
    if not config.is_valid:
        return False, "Invalid Supabase configuration"
    
    supabase: Client = create_client(config.supabase_url, config.supabase_key)
    
    # Try both string and int versions
    for idx in [validator_index, int(validator_index) if str(validator_index).isdigit() else validator_index]:
        response = supabase.table(config.table_name).update({
            'designation': 'lost',
            'to_execution_address': to_execution_address,
            'updated_at': datetime.now().isoformat()
        }).eq('index', idx).execute()
        
        if response.data and len(response.data) > 0:
            return True, f"Updated validator {validator_index}"
    
    return False, f"Validator {validator_index} not found or update failed"

def vote_tab():
    """
    New Vote tab for Ethereum Validator Signature Generator
    """
    st.header("Ethereum Validator Signature Generator")
    
    st.markdown("---")
    st.subheader("Step 1: Create Keystore Signature")
    
    # Instructions
    st.markdown("#### Follow these steps to generate your keystore signature using ethstaker-deposit-cli")
    
    st.markdown("**a. Download ethstaker-deposit-cli**")
    st.markdown("Download the latest version (at least v0.1.3) from the official repository")
    
    if st.button("📥 Download ethstaker-deposit-cli", width=True, type="primary"):
        st.markdown("[Click here to open download page](https://github.com/eth-educators/ethstaker-deposit-cli/releases/tag/v1.0.0)")
    
    st.markdown("**b. Run the CLI Command**")
    st.markdown("Open a terminal and run the generate-bls-to-execution-change-keystore command")
    st.code("./deposit generate-bls-to-execution-change-keystore --keystore=PATH_TO_FILE", language="bash")
    
    st.markdown("**c. Locate the Generated File**")
    st.markdown("""
    The command will generate a `bls_to_execution_change_keystore_transaction-*-*.json` file 
    in the `bls_to_execution_changes_keystore` directory. You'll need this file for step 2.
    """)
    
    st.info("💡 Need help? Check out the [detailed documentation](https://deposit-cli.ethstaker.cc/generate_bls_to_execution_change_keystore.html) for more information.")
    
    st.markdown("---")
    st.subheader("Step 2: Upload and Verify Signature")
    
    # Initialize session states
    if 'verification_complete' not in st.session_state:
        st.session_state.verification_complete = False
    if 'verified_validator_index' not in st.session_state:
        st.session_state.verified_validator_index = None
    if 'verified_execution_address' not in st.session_state:
        st.session_state.verified_execution_address = None
    if 'flag_result' not in st.session_state:
        st.session_state.flag_result = None
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your keystore signature JSON file",
        type=['json'],
        help="Upload the bls_to_execution_change_keystore_transaction JSON file"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            file_contents = uploaded_file.read()
            uploaded_data = json.loads(file_contents)
            
            # Display file contents
            with st.expander("View Uploaded File Contents"):
                st.json(uploaded_data)
            
            # Verify button
            if st.button("🔍 Verify Signature", type="primary", width=True):
                with st.spinner("Verifying signature..."):
                    # Save temporary file for verification
                    temp_file_path = f"temp_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    
                    try:
                        # Write temporary file
                        with open(temp_file_path, 'w') as f:
                            json.dump(uploaded_data, f)
                        
                        # Verify using the verify_json function
                        is_valid = verify_json(temp_file_path)
                        
                        if is_valid:
                            st.session_state.verification_complete = True
                            st.session_state.verified_validator_index = uploaded_data.get("validator_index")
                            st.session_state.verified_execution_address = uploaded_data.get("to_execution_address")
                            st.rerun()
                        else:
                            st.error("❌ Signature verification failed!")
                            st.warning("""
                            **Possible reasons for failure:**
                            - Invalid signature format
                            - Validator not found in the database
                            - Mismatched validator index or public key
                            - Incorrect execution address
                            """)
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
            
            # Show verification success and flag option
            if st.session_state.verification_complete and st.session_state.verified_validator_index:
                st.success("✅ Signature verified successfully!")
                
                validator_index = st.session_state.verified_validator_index
                to_execution_address = st.session_state.verified_execution_address
                
                st.info(f"""
                **Verified Information:**
                - Validator Index: {validator_index}
                - To Execution Address: {to_execution_address}
                """)
                
                # Flag validator section
                st.markdown("---")
                st.subheader("Step 3: Flag Validator as Lost")
                
                st.warning(f"""
                ⚠️ **Important**: This will update the database with:
                - Validator Index: {validator_index}
                - To Execution Address: {to_execution_address}
                - Designation: 'lost'
                """)
                
                # Show result if action was taken
                if st.session_state.flag_result:
                    if st.session_state.flag_result['success']:
                        st.success(f"✅ {st.session_state.flag_result['message']}")
                        st.balloons()
                        if st.button("Flag Another Validator"):
                            st.session_state.verification_complete = False
                            st.session_state.flag_result = None
                            st.rerun()
                    else:
                        st.error(f"❌ {st.session_state.flag_result['message']}")
                        with st.expander("Troubleshooting"):
                            st.write(f"Validator Index: {validator_index}")
                            st.write(f"Type: {type(validator_index)}")
                            st.write(f"Execution Address: {to_execution_address}")
                        if st.button("Try Again"):
                            st.session_state.flag_result = None
                            st.rerun()
                else:
                    # Show flag button
                    if st.button("🚩 Flag Validator as Lost", type="primary", width=True):
                        with st.spinner("Updating database..."):
                            success, message = flag_validator_as_lost(validator_index, to_execution_address)
                            st.session_state.flag_result = {'success': success, 'message': message}
                            if success:
                                st.cache_data.clear()
                            st.rerun()
        
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON file. Please upload a valid keystore signature file.")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        st.info("👆 Please upload your keystore signature JSON file to begin verification")
    
    st.markdown("---")
    st.subheader("Additional Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Documentation:**
        - [ethstaker-deposit-cli Guide](https://deposit-cli.ethstaker.cc/)
        - [BLS to Execution Change](https://deposit-cli.ethstaker.cc/generate_bls_to_execution_change_keystore.html)
        """)
    
    with col2:
        st.markdown("""
        **Support:**
        - [GitHub Issues](https://github.com/eth-educators/ethstaker-deposit-cli/issues)
        - [EthStaker Discord](https://discord.io/ethstaker)
        """)

def main():
    st.set_page_config(
        page_title="Validator Analysis Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Validator Analysis Platform")
    
    # Start scheduler if configured
    has_admin_access = initialize_admin_auth()
    if has_admin_access:
        scheduler = st.session_state.scheduler_manager
        if scheduler.should_enable_scheduler() and not scheduler.is_running:
            scheduler.start_scheduler()
    
    # Show message if user clicked dashboard button
    if 'force_dashboard' in st.session_state and st.session_state.force_dashboard:
        st.info("**Dashboard Updated!** Click on the 'Dashboard' tab above to view the latest data.")
        st.session_state.force_dashboard = False
    
    # Create tabs - Updated structure
    tab1, tab2, tab3 = st.tabs(["Admin", "Dashboard", "Vote"])
    
    with tab1:
        admin_tab()
    
    with tab2:
        dashboard_tab()
    
    with tab3:
        vote_tab()

if __name__ == "__main__":
    main()