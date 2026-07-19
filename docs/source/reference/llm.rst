.. currentmodule:: tensordict.llm

tensordict.llm
==============

The :mod:`tensordict.llm` module provides containers and utilities for
conversational data, designed for language-model pipelines.

:class:`~tensordict.llm.History` is a :class:`~tensordict.TensorClass` that
stores a conversation (roles, contents, tool calls and responses) as a stacked
tensorclass. It offers a centralized API to convert conversations to and from
strings via Hugging Face ``transformers`` chat templates
(:meth:`~tensordict.llm.History.apply_chat_template` and
:meth:`~tensordict.llm.History.from_text`), with assistant token masking
support across multiple model families — useful, for example, to identify
which tokens of a sequence were produced by the assistant in reinforcement
learning post-training pipelines.

.. code-block::

  >>> import tensordict
  >>> tensordict.set_list_to_stack(True).set()
  >>> from tensordict.llm import History
  >>>
  >>> history = History.from_chats([[
  ...     {"role": "user", "content": "Hello"},
  ...     {"role": "assistant", "content": "Hi there!"},
  ... ]])
  >>> history.role
  [['user', 'assistant']]

Messages with structured (multi-modal) content can be expressed with
:class:`~tensordict.llm.ContentBase`, and custom chat templates registered
with :func:`~tensordict.llm.add_chat_template`.

.. note:: This module is the canonical home of ``History``, which previously
    lived in torchrl as ``torchrl.data.llm.History`` and is still re-exported
    there. torchrl's LLM environments, wrappers and ``ChatHistory`` containers
    build on this class.

.. autosummary::
    :toctree: generated/
    :template: td_template.rst

    History
    ContentBase
    add_chat_template
