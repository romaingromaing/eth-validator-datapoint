import pandas as pd
import requests
import json
from datetime import datetime
import time
import os
from dune_client.client import DuneClient
from supabase import create_client, Client
import psycopg2
from urllib.parse import urlparse

# Configuration - Modularized API keys using environment variables
class Config:
    """Configuration class to load API keys and settings from environment variables"""
    def __init__(self):
        # Load environment variables from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            print("Warning: python-dotenv not installed. Using system environment variables only.")
        
        # Helper function to get configuration values
        def get_config_value(key, default=None):
            # First try environment variables (works for both standalone and Streamlit)
            import os
            value = os.getenv(key)
            if value:
                return value
            
            # Only try Streamlit secrets if we're in a Streamlit context
            try:
                # Check if we're running in Streamlit context
                import streamlit as st
                # This will only work if we're actually in a Streamlit app
                if hasattr(st, 'secrets') and st.secrets is not None:
                    return st.secrets.get(key, default)
            except (ImportError, Exception):
                # Not in Streamlit context or secrets not available
                pass
            
            return default
        
        # Load API keys from configuration
        self.dune_sim_api_key = get_config_value('DUNE_SIM_API_KEY')
        self.dune_client_api_key = get_config_value('DUNE_CLIENT_API_KEY')
        
        # Supabase configuration
        self.supabase_url = get_config_value('SUPABASE_URL')
        self.supabase_key = get_config_value('SUPABASE_KEY')
        self.database_url = get_config_value('SUPABASE_DATABASE_URL')
        self.table_name = get_config_value('SUPABASE_TABLE_NAME', 'validator_data')
        
        # CSV upload configuration
        self.connection_timeout = 30
        self.clear_existing_data = True
        self.encoding = 'utf-8'
        self.csv_delimiter = ','
        
        # Other configuration
        self.batch_size = int(get_config_value('BATCH_SIZE', 100))
        self.delay_seconds = int(get_config_value('DELAY_SECONDS', 6))
        self.api_delay = float(get_config_value('API_DELAY', 0.25))
        
        # Validate required environment variables
        self._validate_config()
    
    def _validate_config(self):
        """Validate that all required environment variables are set"""
        required_vars = [
            ('DUNE_SIM_API_KEY', self.dune_sim_api_key),
            ('DUNE_CLIENT_API_KEY', self.dune_client_api_key),
            ('SUPABASE_URL', self.supabase_url),
            ('SUPABASE_KEY', self.supabase_key),
            ('SUPABASE_DATABASE_URL', self.database_url)
        ]
        
        missing_vars = []
        for var_name, var_value in required_vars:
            if not var_value:
                missing_vars.append(var_name)
        
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            print(f"Error: {error_msg}")
            print("\nPlease ensure your .env file contains:")
            for var in missing_vars:
                print(f"  {var}=your_value_here")
            raise ValueError(error_msg)
        
        print("✓ All required environment variables loaded successfully")

def upload_csv_to_supabase(csv_file_path, config):
    """
    Upload CSV file directly to Supabase table using PostgreSQL COPY command
    
    Args:
        csv_file_path (str): Path to CSV file
        config (Config): Configuration object with Supabase database URL
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Parse the Supabase database connection string
        # Get this from: Supabase Dashboard > Settings > Database > Connection String
        url = urlparse(config.database_url)
        
        # Connect directly to PostgreSQL with production settings
        conn = psycopg2.connect(
            host=url.hostname,
            port=url.port or 5432,
            user=url.username,
            password=url.password,
            database=url.path[1:],  # Remove leading '/'
            connect_timeout=getattr(config, 'connection_timeout', 30),
            sslmode='require'  # Required for Supabase connections
        )
        
        cur = conn.cursor()
        
        # Clear existing data (configurable)
        if getattr(config, 'clear_existing_data', True):
            print(f"Clearing existing data from {config.table_name}...")
            cur.execute(f"DELETE FROM {config.table_name}")
        
        # Use COPY command to upload CSV directly
        print(f"Uploading CSV file: {csv_file_path}")
        encoding = getattr(config, 'encoding', 'utf-8')
        delimiter = getattr(config, 'csv_delimiter', ',')
        
        with open(csv_file_path, 'r', encoding=encoding) as f:
            cur.copy_expert(
                f"COPY {config.table_name} FROM STDIN WITH (FORMAT CSV, HEADER true, DELIMITER '{delimiter}')",
                f
            )
        
        # Commit the transaction
        conn.commit()
        
        # Get row count for confirmation
        cur.execute(f"SELECT COUNT(*) FROM {config.table_name}")
        row_count = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        print(f"✓ Successfully uploaded CSV to table '{config.table_name}' - {row_count} records")
        return True
        
    except Exception as e:
        print(f"✗ Error uploading CSV: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

# Initialize configuration
config = Config()

input_file = '0x00-validators.json'
output_file = 'last_datapoint.csv'  # Changed to CSV format

with open(input_file, 'r') as f:
    validators = json.load(f)

def get_deposit_addresses(pubkeys_batch):
    pubkeys_str = ','.join(pubkeys_batch)
    url = f"https://beaconcha.in/api/v1/validator/{pubkeys_str}/deposits"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('status') != 'OK':
            print(f"API error: {data}")
            return {}
        deposits = data.get('data', [])
    except Exception as e:
        print(f"Request failed: {e}")
        return {}

    pubkey_to_address = {}
    for deposit in deposits:
        pubkey = deposit.get('publickey', '').lower()
        from_address = deposit.get('from_address')
        pubkey_to_address[pubkey] = from_address
    return pubkey_to_address

pubkey_to_validator = {val['pubkey'].lower(): val for val in validators if 'pubkey' in val}
all_pubkeys = list(pubkey_to_validator.keys())

# Calculate total batches upfront
total_batches = (len(all_pubkeys) + config.batch_size - 1) // config.batch_size
print(f"Starting processing of {len(all_pubkeys):,} pubkeys in {total_batches} batches")
print(f"Estimated completion time: {(total_batches * config.delay_seconds) / 3600:.1f} hours")

for i in range(0, len(all_pubkeys), config.batch_size):
    batch = all_pubkeys[i:i + config.batch_size]
    current_batch = i // config.batch_size + 1
    
    print(f"Processing batch {current_batch}/{total_batches} ({current_batch/total_batches*100:.1f}%): {len(batch)} pubkeys")
    
    deposit_map = get_deposit_addresses(batch)
    for pubkey, address in deposit_map.items():
        if pubkey in pubkey_to_validator:
            pubkey_to_validator[pubkey]['deposit_address'] = address
    
    if current_batch < total_batches:  # Don't sleep after the last batch
        time.sleep(config.delay_seconds)

enriched_validators = list(pubkey_to_validator.values())
for validator in enriched_validators:
    if 'index' in validator:
        try:
            validator['index'] = int(validator['index'])
        except ValueError:
            print(f"Failed to convert index: {validator.get('pubkey', 'unknown')}: {validator['index']}")

# Convert to DataFrame for further processing
df = pd.DataFrame(enriched_validators)

# Strategy: Extract unique deposit_addresses, perform operations, merge back

# Step 1: Get unique deposit_addresses
unique_addresses = df['deposit_address'].drop_duplicates()
# Step 2: Create a DataFrame with unique addresses for operations
unique_df = pd.DataFrame({'deposit_address': unique_addresses.values})

# Dune Sim API Calls & Data Extraction
def get_transaction_data(deposit_address, api_key):
    """
    Fetch transaction data and analyze for smart contract deployment
    
    Args:
        deposit_address (str): The wallet address to query
        api_key (str): Your Dune API key
    
    Returns:
        dict: Contains last_transaction_time and is_smart_contract, or None values if error
    """
    url = f"https://api.sim.dune.com/v1/evm/transactions/{deposit_address}"
    querystring = {"chain_ids": "1"}
    headers = {"X-Sim-Api-Key": api_key}
    
    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=30)
        response.raise_for_status()  # Raise exception for bad status codes
        
        data = response.json()
        transactions = data.get('transactions', [])
        
        if not transactions:
            print(f"No transactions found for {deposit_address}")
            return {
                'last_transaction_time': None,
                'is_smart_contract': False
            }
        
        # Find the most recent transaction
        # Assuming transactions are sorted by time, take the first one
        # If not sorted, we need to find the max block_time
        last_transaction = max(transactions, key=lambda x: x.get('block_time', ''))
        last_time = last_transaction.get('block_time')
        
        # Check for smart contract deployment
        is_smart_contract = check_smart_contract_deployment(deposit_address, transactions)
        
        print(f"✓ Found {len(transactions)} transactions for {deposit_address}")
        print(f"  Last transaction: {last_time}")
        print(f"  Smart contract deployed: {is_smart_contract}")
        
        return {
            'last_transaction_time': last_time,
            'is_smart_contract': is_smart_contract
        }
        
    except requests.exceptions.RequestException as e:
        print(f"✗ API error for {deposit_address}: {e}")
        return {
            'last_transaction_time': None,
            'is_smart_contract': False
        }
    except Exception as e:
        print(f"✗ Unexpected error for {deposit_address}: {e}")
        return {
            'last_transaction_time': None,
            'is_smart_contract': False
        }

def check_smart_contract_deployment(deposit_address, transactions):
    """
    Check if the address has deployed a smart contract based on transaction history
    
    Args:
        deposit_address (str): The wallet address to check
        transactions (list): List of transactions from the API
    
    Returns:
        bool: True if smart contract deployment detected, False otherwise
    """
    for tx in transactions:
        # Check all conditions for smart contract deployment
        from_address = tx.get('from', '').lower()
        to_address = tx.get('to')
        success = tx.get('success', False)
        data = tx.get('data', '0x')
        
        # Condition 1: from = deposit_address (the address initiated the transaction)
        condition1 = from_address == deposit_address.lower()
        
        # Condition 2: to IS NULL (contract creation transactions have no recipient)
        condition2 = to_address is None
        
        # Condition 3: success = true (the deployment was successful)
        condition3 = success is True
        
        # Condition 4: data != '0x' (contains actual contract bytecode)
        condition4 = data != '0x' and data != '' and data is not None and len(data) > 2
        
        # Debug output for troubleshooting
        if condition1 and condition2:  # Only show debug for potential contract deployments
            print(f"    Checking tx {tx.get('hash', 'unknown')[:10]}...")
            print(f"      from={from_address} == {deposit_address.lower()}: {condition1}")
            print(f"      to={to_address} (is None): {condition2}")
            print(f"      success={success}: {condition3}")
            print(f"      data length={len(data) if data else 0} (>2): {condition4}")
        
        # All conditions must be met
        if condition1 and condition2 and condition3 and condition4:
            print(f"    ✓ Smart contract deployment detected in tx {tx.get('hash', 'unknown')[:10]}")
            return True
    
    return False

def fetch_transaction_data_with_analysis(unique_addresses, api_key, delay=0.25):
    """
    Fetch transaction data and analyze for smart contract deployment for all unique addresses
    
    Args:
        unique_addresses (pd.Series or list): Unique deposit addresses
        api_key (str): Your Dune API key
        delay (float): Delay between API calls to avoid rate limiting
    
    Returns:
        pd.DataFrame: DataFrame with deposit_address, last_transaction_time, and is_smart_contract columns
    """
    results = []
    
    print(f"Fetching and analyzing transaction data for {len(unique_addresses)} unique addresses...")
    
    for i, address in enumerate(unique_addresses, 1):
        print(f"\n[{i}/{len(unique_addresses)}] Processing: {address}")
        
        transaction_data = get_transaction_data(address, api_key)
        
        results.append({
            'deposit_address': address,
            'last_transaction_time': transaction_data['last_transaction_time'],
            'is_smart_contract': transaction_data['is_smart_contract']
        })
        
        # Add delay to avoid hitting rate limits
        if delay > 0 and i < len(unique_addresses):
            print(f"Waiting {delay}s before next request...")
            time.sleep(delay)
    
    return pd.DataFrame(results)

# function call using modularized API key
unique_results = fetch_transaction_data_with_analysis(unique_addresses.values, config.dune_sim_api_key, delay=config.api_delay)

# Adjust the time format
# First convert to datetime, then format
unique_results['last_transaction_time'] = pd.to_datetime(unique_results['last_transaction_time']).dt.strftime('%Y-%m-%d %H:%M')

# check if any address is a dex
dune = DuneClient(config.dune_client_api_key)
query_result = dune.get_latest_result(5644376)  # ethereum-dex-addresses

# Extract the rows data from the query_result
rows_data = query_result.result.rows

# Create DataFrame from the rows
dex_addresses = pd.DataFrame(rows_data)

# Create the is_dex column by checking if deposit_address exists in dex_addresses
unique_results['is_dex'] = unique_results['deposit_address'].isin(dex_addresses['address'])

# Step 4: Merge back to original DataFrame
df_with_transaction_data = df.merge(unique_results, on='deposit_address', how='left')

# Save to CSV with semicolon separator as requested
df_with_transaction_data.to_csv(output_file, index=False)

print(f"Saved to {output_file}")

# Save to Supabase using direct CSV upload
save_success = upload_csv_to_supabase(output_file, config)
if save_success:
    print("Data successfully saved to Supabase!")
else:
    print("Failed to save data to Supabase.")