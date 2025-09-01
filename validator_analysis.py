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
        self.delay_seconds = int(get_config_value('DELAY_SECONDS', 7))
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
    Explicitly specifies column names to handle auto-generated columns (id, created_at)
    """
    try:
        # Parse the Supabase database connection string
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
        clear_data = getattr(config, 'clear_existing_data', True)
        # Also check environment variable for scheduled runs
        if os.getenv('CLEAR_EXISTING_DATA', '').lower() == 'true':
            clear_data = True

        if clear_data:
            print(f"Clearing existing data from {config.table_name}...")
            cur.execute(f"DELETE FROM {config.table_name}")
            deleted_count = cur.rowcount
            print(f"Deleted {deleted_count} existing records")
        
        # Read CSV headers to verify column structure
        import pandas as pd
        df_sample = pd.read_csv(csv_file_path, nrows=0)  # Just get headers
        csv_columns = list(df_sample.columns)
        print(f"CSV columns ({len(csv_columns)}): {csv_columns}")
        
        # Define the expected CSV columns (excluding auto-generated database columns)
        expected_csv_columns = [
            'index', 
            'pubkey', 
            'state', 
            'withdrawal_credentials', 
            'deposit_address', 
            'last_transaction_time', 
            'is_smart_contract', 
            'is_dex'
        ]
        
        # Verify CSV has expected columns
        missing_cols = [col for col in expected_csv_columns if col not in csv_columns]
        extra_cols = [col for col in csv_columns if col not in expected_csv_columns]
        
        if missing_cols:
            print(f"Warning: Missing expected columns: {missing_cols}")
        if extra_cols:
            print(f"Info: Extra columns in CSV (will be ignored): {extra_cols}")
        
        # Use explicit column names in COPY command (only the non-auto-generated ones)
        # This tells PostgreSQL exactly which columns we're providing data for
        columns_str = ', '.join(expected_csv_columns)
        
        copy_sql = f"""
            COPY {config.table_name} ({columns_str}) 
            FROM STDIN WITH (FORMAT CSV, HEADER true, DELIMITER '{getattr(config, 'csv_delimiter', ',')}')
        """
        
        print(f"Using COPY command with explicit columns:")
        print(f"COPY {config.table_name} ({columns_str}) FROM STDIN...")
        print(f"Uploading CSV file: {csv_file_path}")
        
        # Execute the COPY command
        with open(csv_file_path, 'r', encoding=getattr(config, 'encoding', 'utf-8')) as f:
            cur.copy_expert(copy_sql, f)
        
        # Commit the transaction
        conn.commit()
        print("Transaction committed successfully")
        
        # Get row count for confirmation
        cur.execute(f"SELECT COUNT(*) FROM {config.table_name}")
        row_count = cur.fetchone()[0]
        
        # Get a sample of uploaded data to verify
        cur.execute(f"SELECT id, index, pubkey, deposit_address FROM {config.table_name} LIMIT 3")
        sample_rows = cur.fetchall()
        
        print(f"\nSample uploaded data:")
        for row in sample_rows:
            print(f"  id: {row[0]}, index: {row[1]}, pubkey: {row[2][:20]}..., address: {row[3][:20]}...")
        
        cur.close()
        conn.close()
        
        print(f"\n✓ Successfully uploaded CSV to table '{config.table_name}' - {row_count} records")
        print(f"✓ Auto-generated 'id' and 'created_at' columns handled automatically by database")
        return True
        
    except psycopg2.Error as e:
        print(f"✗ PostgreSQL Error: {e}")
        print(f"✗ Error code: {e.pgcode}")
        if hasattr(e, 'pgerror'):
            print(f"✗ Detailed error: {e.pgerror}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False
        
    except Exception as e:
        print(f"✗ Unexpected error uploading CSV: {e}")
        print(f"✗ Error type: {type(e).__name__}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

def get_deposit_addresses(pubkeys_batch):
    """Get deposit addresses for a batch of pubkeys"""
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

def get_validator_info(pubkeys_batch):
    """Get validator status and withdrawal credentials for a batch of pubkeys"""
    pubkeys_str = ','.join(pubkeys_batch)
    url = f"https://beaconcha.in/api/v1/validator/{pubkeys_str}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('status') != 'OK':
            print(f"API error: {data}")
            return {}
        validators_data = data.get('data', [])
    except Exception as e:
        print(f"Request failed: {e}")
        return {}

    pubkey_to_info = {}
    # Handle both single validator and batch responses
    if not isinstance(validators_data, list):
        validators_data = [validators_data]
    
    for validator in validators_data:
        pubkey = validator.get('pubkey', '').lower()
        pubkey_to_info[pubkey] = {
            'status': validator.get('status'),
            'withdrawal_credentials': validator.get('withdrawalcredentials')
        }
    return pubkey_to_info

def load_validators_from_json(input_file):
    """
    Load validators from JSON file and convert to the expected format
    """
    print(f"Loading validators from {input_file}")
    
    try:
        with open(input_file, 'r') as f:
            validators_data = json.load(f)
        
        # Convert JSON data to DataFrame format expected by the rest of the pipeline
        # Assuming the JSON structure contains validator information
        if isinstance(validators_data, list):
            # If it's a list of validators, convert to DataFrame
            df = pd.DataFrame(validators_data)
            
            # Ensure we have the required columns - if not, create them
            if 'pubkey' not in df.columns and 'pubkeys' in df.columns:
                df['pubkey'] = df['pubkeys']
            
            if 'index' not in df.columns:
                # Create index if it doesn't exist
                df['index'] = range(len(df))
            
            # Select only the columns we need
            required_columns = ['index', 'pubkey']
            available_columns = [col for col in required_columns if col in df.columns]
            df_filtered = df[available_columns].copy()
            
            print(f"Loaded {len(df_filtered)} validators from JSON")
            return df_filtered
        else:
            raise ValueError("JSON file does not contain a list of validators")
            
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {input_file}: {e}")
        raise
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        raise

def process_validators(input_file, first_output_file, config):
    """
    Process validators from JSON file to get deposit addresses, withdrawal credentials, and status
    Filter for only 0x00 withdrawal credentials
    """
    # Load the JSON file and convert to expected format
    df = load_validators_from_json(input_file)
    
    # Create validator dictionary with lowercase pubkeys
    validators = df.to_dict('records')
    pubkey_to_validator = {val['pubkey'].lower(): val for val in validators if 'pubkey' in val}
    all_pubkeys = list(pubkey_to_validator.keys())

    print(f"Loaded {len(all_pubkeys):,} validators from {input_file}")

    # Calculate total batches upfront
    total_batches = (len(all_pubkeys) + config.batch_size - 1) // config.batch_size
    print(f"Starting processing of {len(all_pubkeys):,} pubkeys in {total_batches} batches")
    print(f"Estimated completion time: {(total_batches * config.delay_seconds) / 3600:.1f} hours")

    for i in range(0, len(all_pubkeys), config.batch_size):
        batch = all_pubkeys[i:i + config.batch_size]
        current_batch = i // config.batch_size + 1
        
        print(f"Processing batch {current_batch}/{total_batches} ({current_batch/total_batches*100:.1f}%): {len(batch)} pubkeys")
        
        # Get deposit addresses
        deposit_map = get_deposit_addresses(batch)
        
        # Get validator info (status and withdrawal credentials)
        validator_info_map = get_validator_info(batch)
        
        # Update validators with deposit addresses and validator info
        for pubkey in batch:
            if pubkey in pubkey_to_validator:
                # Add deposit address
                if pubkey in deposit_map:
                    pubkey_to_validator[pubkey]['deposit_address'] = deposit_map[pubkey]
                
                # Add validator status and withdrawal credentials
                if pubkey in validator_info_map:
                    pubkey_to_validator[pubkey]['status'] = validator_info_map[pubkey]['status']
                    pubkey_to_validator[pubkey]['withdrawal_credentials'] = validator_info_map[pubkey]['withdrawal_credentials']
        
        if current_batch < total_batches:  # Don't sleep after the last batch
            time.sleep(config.delay_seconds)

    # Convert to list
    enriched_validators = list(pubkey_to_validator.values())
    
    # Convert index to int where possible
    for validator in enriched_validators:
        if 'index' in validator:
            try:
                validator['index'] = int(validator['index'])
            except ValueError:
                print(f"Failed to convert index: {validator.get('pubkey', 'unknown')}: {validator['index']}")

    # Filter for only validators with withdrawal credentials starting with "0x00"
    print(f"Filtering validators with withdrawal credentials starting with '0x00'...")
    validators_0x00 = []
    total_validators = len(enriched_validators)
    
    for validator in enriched_validators:
        withdrawal_creds = validator.get('withdrawal_credentials', '')
        if withdrawal_creds and withdrawal_creds.lower().startswith('0x00'):
            validators_0x00.append(validator)
    
    print(f"Found {len(validators_0x00):,} validators with 0x00 withdrawal credentials out of {total_validators:,} total validators")
    print(f"Filtered out {total_validators - len(validators_0x00):,} validators")

    # Convert filtered validators to DataFrame
    df_filtered = pd.DataFrame(validators_0x00)
    
    # Save the filtered dataset
    print(f"Saving filtered dataset to {first_output_file}")
    df_filtered.to_csv(first_output_file, index=False)
    
    print(f"Processing complete! Saved {len(validators_0x00):,} validators to {first_output_file}")
    
    return validators_0x00

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

def main():
    """Main execution function"""
    try:
        # Initialize configuration
        config = Config()
        
        # File configuration
        input_file = '0x00-validators.json'
        first_output_file = 'enriched_validators_0x00.csv'
        output_file = 'last_datapoint.csv'  # Changed to CSV format
        
        print("="*50)
        print("VALIDATOR PROCESSING PIPELINE")
        print("="*50)
        
        # Step 1: Process validators from JSON and filter for 0x00 withdrawal credentials
        print("\n1. Processing validators and filtering for 0x00 withdrawal credentials...")
        validators_0x00 = process_validators(input_file, first_output_file, config)
        
        if not validators_0x00:
            print("No validators found with 0x00 withdrawal credentials. Exiting.")
            return
        
        # Step 2: Convert to DataFrame for further processing
        print("\n2. Converting to DataFrame for transaction analysis...")
        df = pd.DataFrame(validators_0x00)
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Step 3: Extract unique deposit addresses
        print("\n3. Extracting unique deposit addresses...")
        unique_addresses = df['deposit_address'].drop_duplicates()
        unique_df = pd.DataFrame({'deposit_address': unique_addresses.values})
        print(f"Found {len(unique_addresses)} unique deposit addresses")
        
        # Step 4: Fetch transaction data from Dune Sim API
        print("\n4. Fetching transaction data from Dune Sim API...")
        unique_results = fetch_transaction_data_with_analysis(
            unique_addresses.values, 
            config.dune_sim_api_key, 
            delay=config.api_delay
        )
        
        # Step 5: Format timestamp data
        print("\n5. Formatting timestamp data...")
        unique_results['last_transaction_time'] = pd.to_datetime(
            unique_results['last_transaction_time']
        ).dt.strftime('%Y-%m-%d %H:%M')
        
        # Step 6: Check for DEX addresses
        print("\n6. Checking for DEX addresses...")
        dune = DuneClient(config.dune_client_api_key)
        query_result = dune.get_latest_result(5644376)  # ethereum-dex-addresses
        
        # Extract the rows data from the query_result
        rows_data = query_result.result.rows
        dex_addresses = pd.DataFrame(rows_data)
        
        # Create the is_dex column
        unique_results['is_dex'] = unique_results['deposit_address'].isin(dex_addresses['address'])
        print(f"Found {unique_results['is_dex'].sum()} DEX addresses")
        
        # Step 7: Merge transaction data back to original DataFrame
        print("\n7. Merging transaction data with validator data...")
        df_with_transaction_data = df.merge(unique_results, on='deposit_address', how='left')
        
        # Step 8: Prepare final dataset
        print("\n8. Preparing final dataset...")
        csv_columns = [
            'index', 
            'pubkey', 
            'status',  # Fixed: was 'state', should be 'status'
            'withdrawal_credentials', 
            'deposit_address', 
            'last_transaction_time', 
            'is_smart_contract', 
            'is_dex'
        ]
        
        # Check for missing columns
        missing_columns = [col for col in csv_columns if col not in df_with_transaction_data.columns]
        if missing_columns:
            print(f"Warning: Missing columns in DataFrame: {missing_columns}")
            print(f"Available columns: {list(df_with_transaction_data.columns)}")
            # Add missing columns with default values
            for col in missing_columns:
                if col == 'status':
                    df_with_transaction_data[col] = df_with_transaction_data.get('state', None)
        
        # Reorder DataFrame columns to match database schema
        available_columns = [col for col in csv_columns if col in df_with_transaction_data.columns]
        df_ordered = df_with_transaction_data[available_columns]
        
        # Step 9: Data type formatting
        print("\n9. Formatting data types...")
        
        # Ensure index column is integer
        if 'index' in df_ordered.columns:
            df_ordered['index'] = pd.to_numeric(df_ordered['index'], errors='coerce').astype('Int64')
            null_indices = df_ordered['index'].isna().sum()
            if null_indices > 0:
                print(f"Warning: {null_indices} rows have invalid index values")
        
        # Convert boolean columns
        bool_columns = ['is_smart_contract', 'is_dex']
        for col in bool_columns:
            if col in df_ordered.columns:
                df_ordered[col] = df_ordered[col].astype(bool)
        
        # Step 10: Save to CSV
        print(f"\n10. Saving to CSV: {output_file}")
        df_ordered.to_csv(output_file, index=False)
        print(f"Saved {len(df_ordered)} records to {output_file}")
        print(f"Columns: {list(df_ordered.columns)}")
        
        # Display sample data
        print("\nFirst 3 rows of final data:")
        print(df_ordered.head(3).to_string())
        
        # Step 11: Upload to Supabase
        print(f"\n11. Uploading to Supabase...")
        save_success = upload_csv_to_supabase(output_file, config)
        
        if save_success:
            print("\n" + "="*50)
            print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"✓ Processed {len(validators_0x00)} validators")
            print(f"✓ Analyzed {len(unique_addresses)} unique addresses")
            print(f"✓ Saved to {output_file}")
            print(f"✓ Uploaded to Supabase table: {config.table_name}")
            print("="*50)
        else:
            print("\n" + "="*50)
            print("⚠️ PIPELINE COMPLETED WITH WARNINGS")
            print(f"✓ Data saved to {output_file}")
            print("✗ Failed to upload to Supabase")
            print("="*50)
            
    except Exception as e:
        print(f"\n✗ PIPELINE FAILED: {e}")
        print(f"✗ Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

# Run the main pipeline
if __name__ == "__main__":
    main()