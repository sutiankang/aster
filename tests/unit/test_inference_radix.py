from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import random

import pytest
import torch

from aster.inference import PagedStatePool, PrefixCache, PrefixIdentity, StateError


def state(n, offset=0):
    value = torch.arange(n, dtype=torch.float32).reshape(1, 1, n, 1) + offset
    return ((value, value.clone()),)


def put(pool, cache, identity, tokens, offset=0):
    sequence = pool.create(identity.fingerprint())
    pool.append(sequence, state(len(tokens), offset))
    cache.publish(identity, tokens, sequence)
    pool.release(sequence)


def test_long_prefix_has_linear_metadata_and_shorter_hits():
    pool = PagedStatePool(block_size=16, max_blocks=256)
    cache = PrefixCache(pool, max_entries=128)
    identity = PrefixIdentity("weights")
    put(pool, cache, identity, list(range(4096)))
    assert cache.stats()["radix_nodes"] == 1
    assert cache.stats()["cached_page_references"] == 256
    assert cache.stats()["stored_token_ids"] == 4096
    for length in (1, 16, 17, 511, 4096):
        hit = cache.lookup(identity, list(range(length)))
        assert hit.length == (length - 1) // 16 * 16
        if hit.length:
            torch.testing.assert_close(pool.materialize(hit)[0][0], state(hit.length)[0][0])
        pool.release(hit)
    cache.clear()
    assert pool.used_blocks == 0


def test_split_transfer_and_eviction_preserve_active_request_and_readers():
    pool = PagedStatePool(block_size=2, max_blocks=20)
    cache, identity = PrefixCache(pool), PrefixIdentity("weights")
    put(pool, cache, identity, [1, 2, 3, 4, 5, 6])
    hit = cache.lookup(identity, [1, 2, 3, 4, 5, 6], leave_last_token=False)
    put(pool, cache, identity, [1, 2, 3, 4, 8, 9], offset=20)
    assert cache.stats()["radix_nodes"] == 3
    assert cache.stats()["cached_page_references"] == 4
    assert cache.stats()["stored_token_ids"] == 8
    with pool.borrow(hit):
        cache.clear()
        pool.release(hit)
        assert pool.used_blocks == 3
    assert pool.used_blocks == 0


def test_capacity_evicts_leaf_and_cow_of_returned_tail():
    pool = PagedStatePool(block_size=2, max_blocks=12)
    cache, identity = PrefixCache(pool, max_entries=1), PrefixIdentity("weights")
    put(pool, cache, identity, [1, 2, 3, 4])
    hit = cache.lookup(identity, [1, 2, 3, 4], leave_last_token=False)
    put(pool, cache, identity, [8, 9, 10, 11])
    assert cache.stats()["radix_nodes"] == 1
    torch.testing.assert_close(pool.materialize(hit)[0][0], state(4)[0][0])
    pool.release(hit)
    cache.clear()
    assert pool.used_blocks == 0


def test_randomized_longest_page_prefix_matches_independent_token_oracle():
    rng = random.Random(71)
    pool, identity = PagedStatePool(block_size=2, max_blocks=2048), PrefixIdentity("weights")
    cache, published = PrefixCache(pool, max_entries=2048), []
    for _ in range(80):
        tokens = [rng.randrange(3) for _ in range(rng.randrange(2, 20))]
        put(pool, cache, identity, tokens)
        published.append(tokens[: len(tokens) // 2 * 2])
        for _ in range(3):
            query = rng.choice(published)[: rng.randrange(1, 21)] + [rng.randrange(3)]
            hit = cache.lookup(identity, query)
            expected = max(
                [0]
                + [
                    end
                    for ids in published
                    for end in range(2, min(len(ids), len(query) - 1) + 1, 2)
                    if ids[:end] == query[:end]
                ]
            )
            assert hit.length == expected
            pool.release(hit)
    cache.clear()
    assert pool.used_blocks == 0


def test_domain_isolation_invalid_input_and_concurrent_read_ownership():
    pool, identity = PagedStatePool(block_size=2, max_blocks=16), PrefixIdentity("weights")
    cache = PrefixCache(pool)
    put(pool, cache, identity, list(range(8)))

    def read(_):
        hit = cache.lookup(identity, list(range(8)))
        actual = pool.materialize(hit)[0][0].clone()
        pool.release(hit)
        return actual

    with ThreadPoolExecutor(max_workers=4) as executor:
        for value in executor.map(read, range(40)):
            torch.testing.assert_close(value, state(6)[0][0])
    for field in identity.__dict__:
        miss = cache.lookup(replace(identity, **{field: "other"}), list(range(8)))
        assert miss.length == 0
        pool.release(miss)
    before = cache.stats()
    with pytest.raises(ValueError):
        cache.lookup(identity, [True, 2])
    released = pool.create(identity.fingerprint())
    pool.release(released)
    with pytest.raises(StateError):
        cache.publish(identity, [1, 2], released)
    assert cache.stats() == before
    cache.clear()
    assert pool.used_blocks == 0
