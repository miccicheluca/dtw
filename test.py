import multiprocessing as mp
import time as tm
import numpy as np
from statistics import median, mode
from dtaidistance import dtw
from ll_computation import compute_ll_from_arr_tuple, gen_lead_lag_matrix


def compute_diff_arr_helper(arr: np.array):
    return mode([x - y for x, y in arr])


def compute_diff_arr(arr: np.array):
    return [compute_diff_arr_helper(arr_tuple) for arr_tuple in arr]


def gen_data(n_assets: int, n_points: int):
    return np.array(
        [
            [
                (np.random.randint(0, 10), np.random.randint(0, 10))
                for _ in range(n_points)
            ]
            for _ in range(n_assets)
        ],
        dtype=np.double,
    )


if __name__ == "__main__":
    clusters = np.array(np.random.randint(0, 2, 4), dtype=np.double)
    print("Clusters: ", clusters)
    start = tm.time()
    data = gen_data(100_000, 20)
    end = tm.time()
    print("creation time:", end - start)

    start = tm.time()
    paths = compute_diff_arr(data)
    end = tm.time()
    print("simple computations:", end - start)

    # # p = mp.Pool(processes=4)

    # # start = tm.time()
    # # res = p.map(compute_diff_arr_helper, data)
    # # end = tm.time()
    # # print("Parralel computations: ", end - start)

    start = tm.time()
    res = compute_ll_from_arr_tuple(data, "mode")
    end = tm.time()
    print("Cython computations: ", end - start)

    # print(paths)
    # print(res)

    # n_assets = 6
    # med = np.array([3, 2, 1, 5, 4, -1], dtype=int)
    # clusters = np.array([0, 0, 0, 1, 1, 1], dtype=np.double)

    # ll_matrix = gen_lead_lag_matrix(
    #     n_assets=n_assets, median_from_paths=med, clusters=clusters
    # )
    # print(ll_matrix)
