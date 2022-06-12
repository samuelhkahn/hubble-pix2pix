import random
from functools import partial
from itertools import takewhile
from typing import List, Tuple

import numpy as np

def idx_generator(start: int, end: int) -> int:
    while True:
        yield random.randint(start, end)

def validate_idx(
    mask: np.ndarray,
    validate_array: np.ndarray,
    upper_y: int,
    upper_x: int,
    img_size: int,
    yx: Tuple[int, int],
) -> bool:
    y, x = yx

    if y + img_size > upper_y or x + img_size > upper_x:
        return False

    slice_y = slice(y, y + img_size)
    slice_x = slice(x, x + img_size)

    return mask[slice_y, slice_x].copy().all()
        
        
def make_idx_collection(
    mask: np.ndarray,
    validation_arr: np.ndarray,
    img_size: int,
    collection_size: int,
    start_y: int,
    end_y: int,
    start_x: int,
    end_x: int,
) -> List[int]:

    y_gen = idx_generator(start_y, end_y)
    x_gen = idx_generator(start_x, end_x)

    valid_count = lambda iyx: iyx[0] < collection_size
    valid_idx_f = partial(validate_idx, mask, validation_arr, end_y, end_x, img_size)

    # We want a collection of valid (y, x) coordinates that we can use to
    # crop from the larger image. Explanation starting from the right:
    #
    # zip(y_gen, x_gen) -- is an inifite generator that produces random integer
    #                      tuples of (y, x) within the start and end ranges
    #
    # filter(valid_idx_f, ...) -- filters tuple pairs from zip(y_gen, x_gen)
    #                             that can't be used to generate a valid sample
    #                             based on the conditions in valid_idx_f function
    #
    # enumerate(...) -- returns a tuple (i, (y, x)) where i is an integer that
    #                   counts the values as they are generated. In our case,
    #                   its a running count of the valid tuples generated
    #
    # takewhile(valid_count, ...) -- takewhile returns the values in the
    #                                collection while valid_count returns True.
    #                                In our case, valid_count returns true while
    #                                the i returned by enumerate(...) is less
    #                                than collection_size
    #
    # [iyx for iyx ...] -- iyx is a valid tuple (i, (y, x)) from takewhile(...)
    return [
        iyx
        for iyx in takewhile(
            valid_count, enumerate(filter(valid_idx_f, zip(y_gen, x_gen)))
        )
    ]