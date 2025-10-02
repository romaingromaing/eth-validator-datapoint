from eth_typing import (
    BLSPubkey,
    BLSSignature,
)
from eth_utils import decode_hex
from py_ecc.bls import G2ProofOfPossession as bls
from ssz import (
    ByteVector,
    Serializable,
    bytes4,
    bytes32,
    uint64,
)

bytes20 = ByteVector(20)

class BLSToExecutionChangeKeystore(Serializable):
    """
    Ref: https://github.com/ethereum/consensus-specs/blob/dev/specs/capella/beacon-chain.md#blstoexecutionchange
    """
    fields = [
        ('validator_index', uint64),
        ('to_execution_address', bytes20),
    ]

class ForkData(Serializable):
    fields = [
        ('current_version', bytes4),
        ('genesis_validators_root', bytes32),
    ]

class SigningData(Serializable):
    fields = [
        ('object_root', bytes32),
        ('domain', bytes32)
    ]


def compute_fork_data_root(current_version: bytes, genesis_validators_root: bytes) -> bytes:
    """
    Return the appropriate ForkData root for a given deposit version.
    """
    if len(current_version) != 4:
        raise ValueError(f"Fork version should be in 4 bytes. Got {len(current_version)}.")
    return ForkData(  # type: ignore[no-untyped-call]
        current_version=current_version,
        genesis_validators_root=genesis_validators_root,
    ).hash_tree_root


def compute_signing_root(ssz_object: Serializable, domain: bytes) -> bytes:
    if len(domain) != 32:
        raise ValueError(f"Domain should be in 32 bytes. Got {len(domain)}.")
    domain_wrapped_object = SigningData(  # type: ignore[no-untyped-call]
        object_root=ssz_object.hash_tree_root,
        domain=domain,
    )
    return domain_wrapped_object.hash_tree_root


def compute_bls_change_keystore(fork_version: bytes, genesis_validators_root: bytes) -> bytes:
    """
    VOLUNTARY_EXIT-only `compute_domain`
    """
    if len(fork_version) != 4:
        raise ValueError(f"Fork version should be in 4 bytes. Got {len(fork_version)}.")
    # Temporary constant until agreement
    domain_type = bytes.fromhex('0F000000')
    fork_data_root = compute_fork_data_root(fork_version, genesis_validators_root)
    return domain_type + fork_data_root[:28]


def validate_bls_to_execution_change_keystore(validator_index: str,
                                              to_execution_address: str,
                                              signature: str,
                                              pubkey: str) -> bool:
    
    bls_pubkey = BLSPubkey(bytes.fromhex(pubkey))
    bls_signature = BLSSignature(decode_hex(signature))
    message = BLSToExecutionChangeKeystore(  # type: ignore[no-untyped-call]
        to_execution_address=decode_hex(to_execution_address),
        validator_index=int(validator_index)
    )

    domain = compute_bls_change_keystore(
        fork_version=bytes.fromhex('00000000'),
        genesis_validators_root=bytes.fromhex('4b363db94e286120d76eb905340fdd4e54bfe9f06bf33ff6cf5ad27f511bfe95')
    )

    signing_root = compute_signing_root(message, domain)
    return bls.Verify(bls_pubkey, signing_root, bls_signature)