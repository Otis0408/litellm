import types

from litellm.router_utils.cooldown_handlers import should_cooldown_based_on_allowed_fails_policy


def test_should_cooldown_honors_zero_allowed_fails_policy():
    failed_calls = types.SimpleNamespace(_store={})
    failed_calls.get_cache = lambda key: failed_calls._store.get(key)
    failed_calls.set_cache = lambda key, value, ttl=None: failed_calls._store.__setitem__(key, value)

    router = types.SimpleNamespace(
        allowed_fails=3,
        cooldown_time=60,
        failed_calls=failed_calls,
        get_allowed_fails_from_policy=lambda exception: 0,
    )

    # A policy of 0 allowed fails means the first failure should cool the deployment down.
    assert should_cooldown_based_on_allowed_fails_policy(router, "dep-1", Exception("boom")) is True
