import json
import sys


# At the top of your script
DEBUG_MODE = '--debug' in sys.argv

from verify_keystore_signature import validate_bls_to_execution_change_keystore

validator_file = "./0x00-validators.json"

def verify_json(json_file):
    try:
        with open(validator_file, 'r') as f:
            validators = json.load(f)

        with open(json_file, 'r') as f:
            data = json.load(f)

        # grab contents from signature file
        to_execution_address = data.get("to_execution_address")
        validator_index = data.get("validator_index")
        keystore_signature = data.get("keystore_signature")

        # Validator 0 has 0x01 credentials so this won't accidentally trigger
        if not validator_index:
            print("Missing validator index")
            return False

        if not to_execution_address:
            print("Missing desired execution address")
            return False

        # Strip 0x prefix from address if present
        if to_execution_address.startswith('0x'):
            to_execution_address = to_execution_address[2:]

        # search for validator by index and grab contents (convert to string for comparison)
        validator_data = next(filter(lambda item: item.get("index") == str(validator_index), validators), None)

        if validator_data == None:
            print(f"Unable to find validator of index {validator_index}")
            return False

        validator_pubkey = validator_data.get("pubkey")

        if not validator_pubkey:
            print("Missing validator pubkey in validator data")
            return False

        # Strip 0x prefix from pubkey if present
        if validator_pubkey.startswith('0x'):
            validator_pubkey = validator_pubkey[2:]

        if not keystore_signature:
            print("Missing keystore signature")
            return False

        # Strip 0x prefix from signature if present
        if keystore_signature.startswith('0x'):
            keystore_signature = keystore_signature[2:]

        # Verify keystore signature with debug enabled
        valid_keystore_signature = validate_bls_to_execution_change_keystore(
            validator_index=str(validator_index),
            to_execution_address=to_execution_address,
            signature=keystore_signature,
            pubkey=validator_pubkey,
            debug=DEBUG_MODE
        )

        if not valid_keystore_signature:
            print("Invalid keystore signature")
            return False

        print("Valid signatures")
        return True
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    json_file = sys.argv[1]
    if verify_json(json_file):
        sys.exit(0)
    else:
        sys.exit(1)