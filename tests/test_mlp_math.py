import numpy as np

import mpi_eval_stream
import mpi_train_save


class SingleRankComm:
    def allgather(self, value):
        return [value]

    def Allreduce(self, src, dest, op=None):
        dest[...] = src

    def allreduce(self, value, op=None):
        return value


def test_init_params_shapes_are_reproducible():
    params_a = mpi_train_save.init_params(m=3, n_hidden=4, seed=123)
    params_b = mpi_train_save.init_params(m=3, n_hidden=4, seed=123)

    assert [p.shape for p in params_a] == [(4, 3), (4, 1), (1, 4), (1, 1)]
    for left, right in zip(params_a, params_b):
        np.testing.assert_allclose(left, right)


def test_forward_and_gradients_have_expected_shapes_and_finite_loss():
    params = mpi_train_save.init_params(m=2, n_hidden=3, seed=42)
    X = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, -1.0]])
    y = np.array([[0.5, -0.2, 0.1]])

    grads, loss = mpi_train_save.compute_grads(
        X, y, params, mpi_train_save.relu, mpi_train_save.d_relu
    )

    assert np.isfinite(loss)
    assert [g.shape for g in grads] == [p.shape for p in params]
    assert all(np.all(np.isfinite(g)) for g in grads)


def test_apply_update_with_clipping_keeps_parameters_finite():
    params = mpi_train_save.init_params(m=2, n_hidden=2, seed=1)
    original = [p.copy() for p in params]
    huge_grads = [np.full_like(p, 1_000.0) for p in params]

    mpi_train_save.apply_update(params, huge_grads, lr=0.1, clip=1.0)

    assert all(np.all(np.isfinite(p)) for p in params)
    assert any(not np.allclose(before, after) for before, after in zip(original, params))


def test_allreduce_same_shape_round_trips_single_rank_array():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])

    reduced = mpi_train_save.allreduce_same_shape(SingleRankComm(), arr)

    np.testing.assert_allclose(reduced, arr)


def test_eval_forward_and_rmse_parallel():
    params = [
        np.array([[1.0, -1.0]]),
        np.array([[0.0]]),
        np.array([[2.0]]),
        np.array([[0.5]]),
    ]
    X = np.array([[2.0, -1.0], [1.0, 1.0]])

    pred = mpi_eval_stream.forward(X, params, mpi_eval_stream.relu)
    rmse = mpi_eval_stream.rmse_parallel(SingleRankComm(), sse_local=9.0, n_local=4)

    np.testing.assert_allclose(pred, np.array([[2.5, 0.5]]))
    assert rmse == 1.5
