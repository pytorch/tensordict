# interaction_type

tensordict.nn.probabilistic.interaction_type() → [InteractionType](tensordict.nn.InteractionType.html#tensordict.nn.InteractionType) | [None](https://docs.python.org/3/builtins/constants.html#None)

Returns the interaction type set by [`set_interaction_type`](tensordict.nn.set_interaction_type.html#tensordict.nn.set_interaction_type), or `None` outside of it.

When it returns `None`, each [`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule)
uses its `default_interaction_type`.