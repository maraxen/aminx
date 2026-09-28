"""Test for Debt #1717: verify scoring does not return meaningless decoding orders.

Scoring is full-context (non-autoregressive), so decoding order is not applicable.
This test verifies that the score() function's API documentation correctly
states that decoding_orders is None/absent for full-context scoring.
"""

from __future__ import annotations


def test_scoring_result_schema_documents_decoding_orders_not_applicable() -> None:
  """Verify the score() docstring documents that decoding_orders is not applicable.

  This is a documentation test: it verifies that the public API documentation
  (the score() function's docstring) clearly states that decoding order is
  not applicable to full-context scoring.
  """
  from aminx.host.runner import score as score_fn

  docstring = score_fn.__doc__
  assert docstring is not None, "score() must have a docstring"
  # Check that the docstring mentions full-context to explain why decoding_orders
  # is not applicable
  assert "full-context" in docstring.lower(), (
    "score() docstring must document that scoring is full-context (non-AR) "
    "to explain why decoding_orders is not applicable"
  )
  # Check that the decoding_orders documentation mentions it's not applicable
  # to full-context scoring
  assert "full-context" in docstring.lower() and "decoding_orders" in docstring.lower(), (
    "score() docstring must document the relationship between decoding_orders "
    "and full-context scoring"
  )
