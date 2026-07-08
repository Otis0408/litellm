from litellm.router_utils.pattern_match_deployments import PatternMatchRouter


def test_pattern_router_does_not_over_match_suffix():
    router = PatternMatchRouter()
    router.add_pattern("*-turbo", {"litellm_params": {"model": "openai/*-turbo"}, "model_name": "*-turbo"})
    assert router.route("gpt-4-turbo") is not None
    assert router.route("gpt-4-turbo-preview") is None
