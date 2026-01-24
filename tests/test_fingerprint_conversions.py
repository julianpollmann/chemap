import numpy as np
import pytest
import scipy.sparse as sp
from chemap.fingerprint_conversions import (
    fingerprints_to_csr,
    fingerprints_to_csr_folded,
    fingerprints_to_tfidf,
    fingerprints_to_tfidf_folded,
    fold_csr_mod,
    idf_normalized,
)


# ----------------------------
# Small helpers
# ----------------------------

def _assert_csr_shape_dtype(X: sp.csr_matrix, shape, dtype):
    assert sp.isspmatrix_csr(X)
    assert X.shape == shape
    assert X.dtype == np.dtype(dtype)


# ----------------------------
# idf_normalized
# ----------------------------

def test_idf_normalized_basic_properties():
    # N=10, df=[1,2,5,10]
    df = np.array([1, 2, 5, 10], dtype=np.int32)
    N = 10
    idf = idf_normalized(df, N)

    assert idf.shape == (4,)
    # df=1 -> max idf -> normalized to 1
    assert idf[0] == pytest.approx(1.0)
    # df=N -> log(1)=0 -> normalized to 0
    assert idf[-1] == pytest.approx(0.0)
    # monotonic decreasing with df
    assert idf[0] >= idf[1] >= idf[2] >= idf[3]


def test_idf_normalized_empty():
    df = np.array([], dtype=np.int32)
    out = idf_normalized(df, 5)
    assert out.size == 0


def test_idf_normalized_invalid_N():
    with pytest.raises(ValueError):
        _ = idf_normalized(np.array([1, 2], dtype=np.int32), 0)


# ----------------------------
# Case 1: fingerprints_to_csr
# ----------------------------

def test_fingerprints_to_csr_count_basic_sorted_vocab():
    # Two unfolded count fingerprints (bits, counts)
    fp_A = (np.array([2, 5], dtype=np.int64), np.array([1.0, 1.0], dtype=np.float32))
    fp_B = (np.array([3], dtype=np.int64), np.array([2.0], dtype=np.float32))

    out = fingerprints_to_csr([fp_A, fp_B], sort_bits=True, return_bit_to_col=True)
    X, vocab = out.X, out.vocab

    _assert_csr_shape_dtype(X, (2, 3), np.float32)
    # Sorted vocabulary by original bit id
    np.testing.assert_array_equal(vocab.col_bits, np.array([2, 3, 5], dtype=np.int64))
    np.testing.assert_array_equal(vocab.df, np.array([1, 1, 1], dtype=np.int32))
    assert vocab.bit_to_col == {2: 0, 3: 1, 5: 2}

    expected = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(X.toarray(), expected)


def test_fingerprints_to_csr_binary_basic():
    # Two unfolded binary fingerprints: list of bit arrays
    fp_A = np.array([2, 5], dtype=np.int64)
    fp_B = np.array([3], dtype=np.int64)

    out = fingerprints_to_csr([fp_A, fp_B], sort_bits=True)
    X, vocab = out.X, out.vocab

    _assert_csr_shape_dtype(X, (2, 3), np.float32)
    np.testing.assert_array_equal(vocab.col_bits, np.array([2, 3, 5], dtype=np.int64))
    np.testing.assert_array_equal(vocab.df, np.array([1, 1, 1], dtype=np.int32))

    expected = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(X.toarray(), expected)


def test_fingerprints_to_csr_consolidates_duplicates_in_row():
    # Duplicate bit 7 appears twice in row 0, should sum to 3.0
    fp = (np.array([7, 7, 2], dtype=np.int64), np.array([1.0, 2.0, 5.0], dtype=np.float32))
    out = fingerprints_to_csr([fp], sort_bits=True, consolidate_duplicates_within_rows=True)
    X, vocab = out.X, out.vocab

    np.testing.assert_array_equal(vocab.col_bits, np.array([2, 7], dtype=np.int64))
    row = X.toarray().ravel()
    # bit 2 -> 5.0 ; bit 7 -> 3.0
    np.testing.assert_allclose(row, np.array([5.0, 3.0], dtype=np.float32))


def test_fingerprints_to_csr_tf_transform_applied_counts_only():
    fp = (np.array([1, 2], dtype=np.int64), np.array([1.0, 3.0], dtype=np.float32))
    out = fingerprints_to_csr([fp], sort_bits=True, tf_transform=np.log1p)
    X = out.X.toarray().ravel()
    # log1p([1,3]) = [log2, log4]
    np.testing.assert_allclose(X, np.log1p(np.array([1.0, 3.0], dtype=np.float32)), rtol=1e-6)


def test_fingerprints_to_csr_raises_on_mixed_input_types():
    fps = [
        (np.array([1], dtype=np.int64), np.array([1.0], dtype=np.float32)),  # count
        np.array([2], dtype=np.int64),  # binary
    ]
    with pytest.raises(TypeError):
        _ = fingerprints_to_csr(fps)


# ----------------------------
# Occurrence filtering
# ----------------------------

def test_fingerprints_to_csr_min_occurrence_filters_rare_bits():
    # 3 documents:
    # doc0: bits {1,2}
    # doc1: bits {2}
    # doc2: bits {3}
    # df(1)=1, df(2)=2, df(3)=1
    fps = [
        (np.array([1, 2], np.int64), np.array([1.0, 1.0], np.float32)),
        (np.array([2], np.int64), np.array([2.0], np.float32)),
        (np.array([3], np.int64), np.array([3.0], np.float32)),
    ]
    out = fingerprints_to_csr(fps, min_occurrence=2, sort_bits=True)
    X, vocab = out.X, out.vocab

    # Only bit 2 remains
    np.testing.assert_array_equal(vocab.col_bits, np.array([2], dtype=np.int64))
    np.testing.assert_array_equal(vocab.df, np.array([2], dtype=np.int32))
    assert X.shape == (3, 1)

    np.testing.assert_allclose(X.toarray(), np.array([[1.0], [2.0], [0.0]], dtype=np.float32))


def test_fingerprints_to_csr_max_occurrence_int_filters_frequent_bits():
    # bit 2 appears in all 3 documents -> df=3
    fps = [
        np.array([1, 2], np.int64),
        np.array([2], np.int64),
        np.array([2, 3], np.int64),
    ]
    out = fingerprints_to_csr(fps, max_occurrence=2, sort_bits=True)
    X, vocab = out.X, out.vocab

    # bit 2 removed; remaining bits 1 and 3
    np.testing.assert_array_equal(vocab.col_bits, np.array([1, 3], dtype=np.int64))
    np.testing.assert_array_equal(vocab.df, np.array([1, 1], dtype=np.int32))
    np.testing.assert_allclose(X.toarray(), np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32))


def test_fingerprints_to_csr_max_occurrence_fraction_filters_frequent_bits():
    # N=4, max_occ=0.5 => floor(2) => remove bits with df>2
    fps = [
        np.array([1, 2], np.int64),
        np.array([2], np.int64),
        np.array([2, 3], np.int64),
        np.array([2], np.int64),
    ]
    # df(2)=4 -> removed; df(1)=1 kept; df(3)=1 kept
    out = fingerprints_to_csr(fps, max_occurrence=0.5, sort_bits=True)
    X, vocab = out.X, out.vocab
    np.testing.assert_array_equal(vocab.col_bits, np.array([1, 3], dtype=np.int64))
    assert X.shape == (4, 2)


def test_occurrence_threshold_validation():
    fps = [np.array([1], np.int64)]
    with pytest.raises(ValueError):
        _ = fingerprints_to_csr(fps, min_occurrence=0)
    with pytest.raises(ValueError):
        _ = fingerprints_to_csr(fps, max_occurrence=0)
    with pytest.raises(ValueError):
        _ = fingerprints_to_csr(fps, max_occurrence=1.0)  # must be in (0,1)
    with pytest.raises(ValueError):
        _ = fingerprints_to_csr(fps, max_occurrence=-0.1)
    with pytest.raises(ValueError):
        _ = fingerprints_to_csr(fps, min_occurrence=3, max_occurrence=2)


# ----------------------------
# Case 2: fingerprints_to_tfidf
# ----------------------------

def test_fingerprints_to_tfidf_binary_idf_only():
    # Two docs:
    # doc0: bits {1,2}
    # doc1: bits {2}
    # df(1)=1, df(2)=2, N=2
    fps = [
        np.array([1, 2], np.int64),
        np.array([2], np.int64),
    ]
    out = fingerprints_to_tfidf(fps, sort_bits=True)
    X, vocab, idf = out.X, out.vocab, out.idf

    assert idf is not None
    np.testing.assert_array_equal(vocab.col_bits, np.array([1, 2], dtype=np.int64))
    # idf_norm(df=1,N=2)=1 ; idf_norm(df=2,N=2)=0
    np.testing.assert_allclose(idf, np.array([1.0, 0.0], dtype=np.float32), rtol=1e-6)

    # doc0 has 1 and 2 -> values [1,0]; doc1 only 2 -> [0]
    np.testing.assert_allclose(X.toarray(), np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32))


def test_fingerprints_to_tfidf_count_tf_times_idf():
    # Two docs (counts):
    # doc0: bit1=2, bit2=1
    # doc1: bit2=3
    # df(1)=1, df(2)=2, N=2 -> idf=[1,0]
    fps = [
        (np.array([1, 2], np.int64), np.array([2.0, 1.0], np.float32)),
        (np.array([2], np.int64), np.array([3.0], np.float32)),
    ]
    out = fingerprints_to_tfidf(fps, sort_bits=True)
    X, idf = out.X.toarray(), out.idf
    assert idf is not None
    np.testing.assert_allclose(idf, np.array([1.0, 0.0], dtype=np.float32), rtol=1e-6)

    # After multiplying by idf: bit1 survives, bit2 becomes 0 everywhere
    expected = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(X, expected)


def test_fingerprints_to_tfidf_respects_min_occurrence_before_idf():
    # 3 docs counts:
    # doc0: bit1
    # doc1: bit2
    # doc2: bit2
    # df(1)=1 df(2)=2 N=3
    fps = [
        (np.array([1], np.int64), np.array([1.0], np.float32)),
        (np.array([2], np.int64), np.array([1.0], np.float32)),
        (np.array([2], np.int64), np.array([1.0], np.float32)),
    ]
    out = fingerprints_to_tfidf(fps, min_occurrence=2, sort_bits=True)
    # Only bit2 remains -> df=[2], N=3 -> idf_norm = log(3/2)/log(3)
    expected_idf = np.log(3.0 / 2.0) / np.log(3.0)

    np.testing.assert_array_equal(out.vocab.col_bits, np.array([2], np.int64))
    assert out.idf is not None
    assert out.idf[0] == pytest.approx(expected_idf, rel=1e-6)


# ----------------------------
# Case 3: Folding
# ----------------------------

def test_fold_csr_mod_basic():
    # 1 row, 6 cols -> fold to 3 cols via mod
    X = sp.csr_matrix(np.array([[1, 0, 2, 0, 3, 0]], dtype=np.float32))
    Y = fold_csr_mod(X, n_folded_features=3)

    # indices 0->0, 2->2, 4->1 => folded row [1,3,2]
    np.testing.assert_allclose(Y.toarray(), np.array([[1.0, 3.0, 2.0]], dtype=np.float32))

def test_fingerprints_to_csr_folded_matches_manual_fold_after_filtering():
    # Two docs with bits:
    # doc0: {1, 2, 10}
    # doc1: {2, 10}
    # df(1)=1, df(2)=2, df(10)=2
    # min_occ=2 removes bit1 -> vocabulary bits {2,10}
    # IMPORTANT: folding is done on CSR COLUMN INDICES (0..n_cols-1), not original bit ids.
    # After filtering, {2,10} -> columns {0,1}. Folding modulo 4 keeps them at {0,1}.
    fps = [
        np.array([1, 2, 10], np.int64),
        np.array([2, 10], np.int64),
    ]
    out = fingerprints_to_csr_folded(fps, n_folded_features=4, min_occurrence=2, sort_bits=True)
    Xf = out.X.toarray()

    expected = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(Xf, expected)

def test_fingerprints_to_tfidf_folded_dense_shape():
    fps = [
        (np.array([1, 2], np.int64), np.array([2.0, 1.0], np.float32)),
        (np.array([2], np.int64), np.array([3.0], np.float32)),
        (np.array([3], np.int64), np.array([1.0], np.float32)),
    ]
    out = fingerprints_to_tfidf_folded(
        fps,
        n_folded_features=8,
        min_occurrence=None,
        max_occurrence=None,
        sort_bits=True,
    )
    assert out.X.shape == (3, 8)
    assert out.idf is not None
    assert out.vocab.col_bits.ndim == 1
    assert out.vocab.df.ndim == 1


def test_fingerprints_to_tfidf_folded_can_be_densified_fixed_length():
    fps = [
        np.array([1, 2], np.int64),
        np.array([2], np.int64),
        np.array([3], np.int64),
    ]
    out = fingerprints_to_tfidf_folded(fps, n_folded_features=4096, sort_bits=True)
    dense = out.X.toarray()
    assert dense.shape == (3, 4096)
    assert dense.dtype == np.float32


# ----------------------------
# Order / stability
# ----------------------------

def test_sort_bits_true_is_deterministic_vocab_order():
    fps = [
        np.array([10, 2], np.int64),
        np.array([2, 5], np.int64),
    ]
    out = fingerprints_to_csr(fps, sort_bits=True)
    np.testing.assert_array_equal(out.vocab.col_bits, np.array([2, 5, 10], dtype=np.int64))


def test_sort_bits_false_preserves_first_seen_vocab_order():
    fps = [
        np.array([10, 2], np.int64),   # first sees 10 then 2
        np.array([2, 5], np.int64),    # then sees 5
    ]
    out = fingerprints_to_csr(fps, sort_bits=False)
    np.testing.assert_array_equal(out.vocab.col_bits, np.array([10, 2, 5], dtype=np.int64))
