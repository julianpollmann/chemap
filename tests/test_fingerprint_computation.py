from typing import Sequence
import numpy as np
import pytest
from chemap import FingerprintConfig, compute_fingerprints


# =============================================================================
# Test doubles / fakes
# =============================================================================

class _FakeSparseCountFP:
    """Mimics RDKit's SparseCountFingerprint-like object."""
    def __init__(self, elems: dict[int, int]):
        self._elems = dict(elems)

    def GetNonzeroElements(self):
        return dict(self._elems)


class _FakeCountVector:
    """Mimics RDKit SparseIntVect returned by GetCountFingerprint()."""
    def __init__(self, length: int, nz: dict[int, int]):
        self._length = int(length)
        self._nz = dict(nz)

    def GetLength(self):
        return self._length

    def GetNonzeroElements(self):
        return dict(self._nz)


class _FakeBitVector:
    """Mimics RDKit ExplicitBitVect returned by GetFingerprint()."""
    def __init__(self, nbits: int, on_bits: Sequence[int]):
        self._nbits = int(nbits)
        self._on_bits = sorted(set(int(b) for b in on_bits))

    def GetNumBits(self):
        return self._nbits

    def GetOnBits(self):
        return list(self._on_bits)


class FakeRDKitFPGen:
    """
    RDKit rdFingerprintGenerator-shaped fake.

    Provides:
      - GetFingerprintAsNumPy(mol)
      - GetCountFingerprintAsNumPy(mol)
      - GetSparseCountFingerprint(mol).GetNonzeroElements()
      - GetCountFingerprint(mol).GetLength()/GetNonzeroElements()
      - GetFingerprint(mol).GetNumBits()/GetOnBits()
    """
    def __init__(
        self,
        dense_fp: np.ndarray | None = None,
        dense_count_fp: np.ndarray | None = None,
        sparse_dict: dict[int, int] | None = None,
    ):
        self._dense_fp = np.array([0, 1, 1, 0], dtype=np.int32) if dense_fp is None else np.asarray(dense_fp)
        self._dense_count_fp = np.array([0, 2, 5, 0],
                                        dtype=np.int32) if dense_count_fp is None else np.asarray(dense_count_fp)
        self._sparse = {1: 2, 2: 5} if sparse_dict is None else dict(sparse_dict)

    def GetFingerprintAsNumPy(self, mol):
        return self._dense_fp.copy()

    def GetCountFingerprintAsNumPy(self, mol):
        return self._dense_count_fp.copy()

    def GetSparseCountFingerprint(self, mol):
        return _FakeSparseCountFP(self._sparse)

    def GetCountFingerprint(self, mol):
        length = int(self._dense_count_fp.shape[0])
        return _FakeCountVector(length=length, nz=self._sparse)

    def GetFingerprint(self, mol):
        nbits = int(self._dense_fp.shape[0])
        on_bits = [i for i, v in enumerate(self._dense_fp) if v != 0]
        return _FakeBitVector(nbits=nbits, on_bits=on_bits)


class DummyUnsupported:
    """No required methods."""
    pass


# -----------------------------------------------------------------------------
# Clone-safe sklearn/scikit-fingerprints transformer fakes
# -----------------------------------------------------------------------------

class FakeTransformer:
    """
    Clone-safe sklearn/scikit-fingerprints-like transformer fake.

    IMPORTANT:
    Your implementation clones transformers via:
        fpgen.__class__(**fpgen.get_params(deep=False))
    so any test behavior MUST be encoded in get_params() keys.

    Parameters (all included in get_params):
      - sparse: bool       (skfp: dense vs CSR; your code forces sparse=False)
      - verbose: int       (your code sets based on show_progress)
      - variant: str|None  (required for ragged mode; your code sets to "raw_bits")
      - mode: str          (controls deterministic output in transform)
      - n_features: int    (output feature dimension for most modes)
    """

    def __init__(
        self,
        *,
        sparse: bool = True,
        verbose: int = 0,
        variant: str | None = None,
        mode: str = "onehot",
        n_features: int = 6,
    ):
        self._params = {
            "sparse": sparse,
            "verbose": verbose,
            "variant": variant,
            "mode": mode,
            "n_features": int(n_features),
        }

    def get_params(self, deep: bool = False):
        return dict(self._params)

    def transform(self, X):
        n = len(X)
        mode = self._params["mode"]
        d = int(self._params["n_features"])
        verbose = int(self._params["verbose"])
        sparse = bool(self._params["sparse"])
        variant = self._params.get("variant", None)

        if mode == "check_sparse_verbose":
            # Used to verify your adapter forced sparse=False and set verbose properly.
            if (sparse is False) and (verbose == 0):
                return np.array([[1, 2], [3, 4]], dtype=np.float32)
            return np.array([[9, 9], [9, 9]], dtype=np.float32)

        if mode == "check_verbose":
            # Used to verify your adapter sets verbose=1 when show_progress=True.
            if verbose == 1:
                return np.array([[1, 0], [0, 1]], dtype=np.float32)
            return np.array([[0, 0], [0, 0]], dtype=np.float32)

        if mode == "fixed_02_30":
            # Used for scaling test
            assert n == 2
            return np.array([[0, 2], [3, 0]], dtype=np.float32)

        if mode == "fixed_123":
            # Used for dense weights test
            assert n == 1
            return np.array([[1, 2, 3]], dtype=np.float32)

        if mode == "ragged_binary":
            # Used for ragged binary; only correct when variant has been set to raw_bits by adapter.
            if variant != "raw_bits":
                # distinguish failure mode clearly
                return np.zeros((n, d), dtype=np.float32)
            M = np.zeros((n, d), dtype=np.float32)
            M[0, 1] = 1.0
            M[0, 4] = 1.0
            return M

        if mode == "ragged_count":
            if variant != "raw_bits":
                return np.zeros((n, d), dtype=np.float32)
            M = np.zeros((n, d), dtype=np.float32)
            M[0, 2] = 4.0
            M[0, 5] = 2.0
            return M

        # default: one-hot
        M = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            M[i, i % d] = 1.0
        return M


class NoVariantTransformer(FakeTransformer):
    """Transformer fake without `variant` in params (to test NotImplementedError)."""
    def get_params(self, deep: bool = False):
        p = super().get_params(deep)
        p.pop("variant", None)
        return p


# =============================================================================
# Config validation / routing
# =============================================================================

def test_validate_scaling_rejects_unknown():
    cfg = FingerprintConfig(scaling="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scaling"):
        compute_fingerprints(["CCO"], DummyUnsupported(), cfg)


def test_validate_dense_rejects_ragged_weights():
    cfg = FingerprintConfig(sparse=False, ragged_weights={1: 2.0})
    with pytest.raises(ValueError, match="ragged_weights"):
        compute_fingerprints(["CCO"], DummyUnsupported(), cfg)


def test_validate_ragged_rejects_dense_weights():
    cfg = FingerprintConfig(sparse=True, dense_weights=np.ones(10, dtype=np.float32))
    with pytest.raises(ValueError, match="dense_weights"):
        compute_fingerprints(["CCO"], DummyUnsupported(), cfg)


def test_validate_ragged_rejects_ragged_weights_when_binary():
    cfg = FingerprintConfig(sparse=True, count=False, ragged_weights={1: 2.0})
    with pytest.raises(ValueError, match="ragged_weights"):
        compute_fingerprints(["CCO"], DummyUnsupported(), cfg)


def test_validate_dense_weights_must_be_1d():
    cfg = FingerprintConfig(sparse=False, dense_weights=np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="1D"):
        compute_fingerprints(["CCO"], DummyUnsupported(), cfg)


def test_unsupported_fpgen_typeerror():
    with pytest.raises(TypeError, match="Unsupported fpgen"):
        compute_fingerprints(["CCO"], DummyUnsupported(), FingerprintConfig())


# =============================================================================
# RDKit backend via fake fpgen (deterministic numerics)
# =============================================================================

def test_rdkit_dense_binary_dtype_and_values():
    fpgen = FakeRDKitFPGen(dense_fp=np.array([0, 1, 0, 1], dtype=np.int32))
    cfg = FingerprintConfig(count=False, sparse=False)

    X = compute_fingerprints(["CCO", "CCC"], fpgen, cfg)
    assert isinstance(X, np.ndarray)
    assert X.shape == (2, 4)
    assert X.dtype == np.float32
    np.testing.assert_array_equal(X[0], np.array([0, 1, 0, 1], dtype=np.float32))


def test_rdkit_dense_count_scaling_log():
    fpgen = FakeRDKitFPGen(dense_count_fp=np.array([0, 2, 5, 1], dtype=np.int32))
    cfg = FingerprintConfig(count=True, sparse=False, scaling="log")

    X = compute_fingerprints(["CCO"], fpgen, cfg)
    expected = np.log1p(np.array([0, 2, 5, 1], dtype=np.float32))
    assert X.shape == (1, 4)
    np.testing.assert_allclose(X[0], expected, rtol=1e-6, atol=1e-6)


def test_rdkit_dense_count_dense_weights():
    fpgen = FakeRDKitFPGen(dense_count_fp=np.array([1, 2, 3, 4], dtype=np.int32))
    w = np.array([1.0, 2.0, 0.5, 10.0], dtype=np.float32)
    cfg = FingerprintConfig(count=True, sparse=False, dense_weights=w)

    X = compute_fingerprints(["CCO"], fpgen, cfg)
    expected = np.array([1, 2, 3, 4], dtype=np.float32) * w
    np.testing.assert_allclose(X[0], expected, rtol=1e-6, atol=1e-6)


def test_rdkit_dense_weights_shape_mismatch_raises():
    fpgen = FakeRDKitFPGen(dense_count_fp=np.array([1, 2, 3, 4], dtype=np.int32))
    cfg = FingerprintConfig(count=True, sparse=False, dense_weights=np.ones(3, dtype=np.float32))

    with pytest.raises(ValueError, match="dense_weights length"):
        _ = compute_fingerprints(["CCO"], fpgen, cfg)


def test_rdkit_ragged_binary_sorted_int64():
    fpgen = FakeRDKitFPGen(sparse_dict={10: 1, 3: 2, 7: 5})
    cfg = FingerprintConfig(count=False, sparse=True)

    out = compute_fingerprints(["CCO"], fpgen, cfg)
    assert isinstance(out, list)
    assert len(out) == 1
    keys = out[0]
    assert keys.dtype == np.int64
    assert list(keys) == [3, 7, 10]


def test_rdkit_ragged_count_scaling_and_weights():
    fpgen = FakeRDKitFPGen(sparse_dict={5: 2, 2: 4})
    cfg = FingerprintConfig(
        count=True,
        sparse=True,
        scaling="log",
        ragged_weights={2: 10.0},  # 5 defaults to 1.0
    )

    out = compute_fingerprints(["CCO"], fpgen, cfg)
    keys, vals = out[0]
    assert list(keys) == [2, 5]
    expected_base = np.array([4.0, 2.0], dtype=np.float32)
    expected = np.log1p(expected_base) * np.array([10.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(vals, expected, rtol=1e-6, atol=1e-6)
    assert vals.dtype == np.float32


# =============================================================================
# Invalid SMILES handling (uses real RDKit SMILES parsing)
# =============================================================================

def test_rdkit_invalid_policy_drop_dense():
    fpgen = FakeRDKitFPGen(dense_fp=np.array([1, 0, 0, 0], dtype=np.int32))
    cfg = FingerprintConfig(count=False, sparse=False, invalid_policy="drop")

    X = compute_fingerprints(["this_is_not_a_smiles", "CCO"], fpgen, cfg)
    assert X.shape == (1, 4)  # invalid dropped


def test_rdkit_invalid_policy_keep_dense_backfills_zeros():
    fpgen = FakeRDKitFPGen(dense_fp=np.array([1, 0, 0, 0], dtype=np.int32))
    cfg = FingerprintConfig(count=False, sparse=False, invalid_policy="keep")

    X = compute_fingerprints(["this_is_not_a_smiles", "CCO"], fpgen, cfg)
    assert X.shape == (2, 4)
    np.testing.assert_array_equal(X[0], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(X[1], np.array([1, 0, 0, 0], dtype=np.float32))


def test_rdkit_invalid_policy_keep_all_invalid_dense():
    fpgen = FakeRDKitFPGen()
    cfg = FingerprintConfig(count=False, sparse=False, invalid_policy="keep")

    X = compute_fingerprints(["bad1", "bad2"], fpgen, cfg)
    assert X.shape == (2, 0)  # all invalid -> (N, 0)


def test_rdkit_invalid_policy_keep_ragged():
    fpgen = FakeRDKitFPGen(sparse_dict={1: 2})
    cfg = FingerprintConfig(count=True, sparse=True, invalid_policy="keep")

    out = compute_fingerprints(["this_is_not_a_smiles", "CCO"], fpgen, cfg)
    assert len(out) == 2
    k0, v0 = out[0]
    assert k0.size == 0 and v0.size == 0
    k1, v1 = out[1]
    assert list(k1) == [1]
    assert list(v1) == [2.0]


def test_rdkit_invalid_policy_raise():
    fpgen = FakeRDKitFPGen()
    cfg = FingerprintConfig(count=False, sparse=False, invalid_policy="raise")

    with pytest.raises(ValueError, match="Invalid SMILES"):
        _ = compute_fingerprints(["this_is_not_a_smiles"], fpgen, cfg)


# =============================================================================
# sklearn/scikit-fingerprints backend via clone-safe fakes
# =============================================================================

def test_sklearn_dense_forces_sparse_false_and_sets_verbose_show_progress_false():
    fp = FakeTransformer(
        sparse=True,
        verbose=123,
        mode="check_sparse_verbose",
        n_features=2,
    )
    cfg = FingerprintConfig(count=True, sparse=False)

    X = compute_fingerprints(["A", "B"], fp, cfg, show_progress=False)
    np.testing.assert_array_equal(X, np.array([[1, 2], [3, 4]], dtype=np.float32))


def test_sklearn_dense_sets_verbose_show_progress_true():
    fp = FakeTransformer(
        sparse=False,
        verbose=0,
        mode="check_verbose",
        n_features=2,
    )
    cfg = FingerprintConfig(count=True, sparse=False)

    X = compute_fingerprints(["A", "B"], fp, cfg, show_progress=True)
    np.testing.assert_array_equal(X, np.array([[1, 0], [0, 1]], dtype=np.float32))


def test_sklearn_dense_scaling_log_applies_when_count_true():
    fp = FakeTransformer(
        sparse=False,
        mode="fixed_02_30",
        n_features=2,
    )
    cfg = FingerprintConfig(count=True, sparse=False, scaling="log")

    X = compute_fingerprints(["A", "B"], fp, cfg)
    expected = np.log1p(np.array([[0, 2], [3, 0]], dtype=np.float32)).astype(np.float32)
    np.testing.assert_allclose(X, expected, rtol=1e-6, atol=1e-6)


def test_sklearn_dense_weights_applies():
    fp = FakeTransformer(
        sparse=False,
        mode="fixed_123",
        n_features=3,
    )
    w = np.array([1.0, 10.0, 0.5], dtype=np.float32)
    cfg = FingerprintConfig(count=True, sparse=False, dense_weights=w)

    X = compute_fingerprints(["A"], fp, cfg)
    np.testing.assert_allclose(X[0], np.array([1, 20, 1.5], dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_sklearn_ragged_requires_variant():
    fp = NoVariantTransformer(sparse=False, mode="ragged_binary", n_features=6)
    cfg = FingerprintConfig(count=False, sparse=True)

    with pytest.raises(NotImplementedError, match="variant"):
        _ = compute_fingerprints(["A"], fp, cfg)


def test_sklearn_ragged_sets_variant_raw_bits_and_returns_ragged_binary():
    fp = FakeTransformer(
        sparse=False,
        variant="folded",
        mode="ragged_binary",
        n_features=6,
    )
    cfg = FingerprintConfig(count=False, sparse=True)

    out = compute_fingerprints(["A"], fp, cfg)
    assert isinstance(out, list)
    assert out[0].dtype == np.int64
    assert list(out[0]) == [1, 4]


def test_sklearn_ragged_count_scaling_and_ragged_weights():
    fp = FakeTransformer(
        sparse=False,
        variant="folded",
        mode="ragged_count",
        n_features=6,
    )
    cfg = FingerprintConfig(count=True, sparse=True, scaling="log", ragged_weights={2: 10.0})

    out = compute_fingerprints(["A"], fp, cfg)
    keys, vals = out[0]
    assert list(keys) == [2, 5]
    expected = np.log1p(np.array([4.0, 2.0], dtype=np.float32)) * np.array([10.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(vals, expected, rtol=1e-6, atol=1e-6)


# =============================================================================
# Optional real-RDKit integration smoke tests (skip if not available)
# =============================================================================

@pytest.mark.filterwarnings("ignore")
def test_rdkit_morgan_dense_smoke():
    rdFingerprintGenerator = pytest.importorskip("rdkit.Chem.rdFingerprintGenerator")

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    cfg = FingerprintConfig(count=False, sparse=False, invalid_policy="drop")

    X = compute_fingerprints(["CCO", "c1ccccc1"], fpgen, cfg)
    assert isinstance(X, np.ndarray)
    assert X.shape == (2, 1024)
    assert X.dtype == np.float32
    assert (X >= 0).all()


@pytest.mark.filterwarnings("ignore")
def test_rdkit_morgan_ragged_smoke_sorted_keys():
    rdFingerprintGenerator = pytest.importorskip("rdkit.Chem.rdFingerprintGenerator")

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    cfg = FingerprintConfig(count=True, sparse=True, invalid_policy="drop")

    out = compute_fingerprints(["CCO", "c1ccccc1"], fpgen, cfg)
    assert isinstance(out, list)
    assert len(out) == 2

    k0, v0 = out[0]
    assert k0.dtype == np.int64
    assert v0.dtype == np.float32
    assert np.all(k0[:-1] <= k0[1:])
