"""
tests/test_entity_match.py — Unit tests for src/entity_match.py: deterministic
person-name extraction from a query, and author-metadata matching (never
body-text matching, per "a mention is not authorship").
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.entity_match import (
    extract_candidate_person_names, name_matches_author_field,
    chunk_author_matches_query,
)


# ── extract_candidate_person_names ──────────────────────────────────────────

def test_extracts_name_with_honorific_and_initial():
    names = extract_candidate_person_names("What blogs has Dr Y Nithiyanandam written?")
    assert names == ["Y Nithiyanandam"]


def test_extracts_name_from_who_is_question():
    names = extract_candidate_person_names("Who is Dr Y Nithiyanandam?")
    assert names == ["Y Nithiyanandam"]


def test_extracts_name_without_honorific():
    names = extract_candidate_person_names("Which articles has Nithiyanandam authored?")
    assert names == ["Nithiyanandam"]


def test_extracts_full_name_no_title():
    names = extract_candidate_person_names("What publications are by Y Nithiyanandam?")
    assert names == ["Y Nithiyanandam"]


def test_no_false_positive_on_institution_name():
    names = extract_candidate_person_names("What has Takshashila published about AI policy?")
    assert names == []  # "Takshashila" alone, "AI" alone — no 2+ token run


def test_no_false_positive_on_plain_query():
    assert extract_candidate_person_names("what blogs are available") == []


# ── name_matches_author_field ────────────────────────────────────────────────

def test_matches_full_form_to_short_form():
    assert name_matches_author_field("Y Nithiyanandam", "Dr. Y. Nithiyanandam")


def test_matches_case_insensitive():
    assert name_matches_author_field("y nithiyanandam", "Y NITHIYANANDAM")


def test_matches_list_of_authors():
    assert name_matches_author_field("Y Nithiyanandam", ["Guest Author", "Dr. Y. Nithiyanandam"])


def test_does_not_match_unrelated_name():
    assert not name_matches_author_field("Y Nithiyanandam", "Pranay Kotasthane")


def test_does_not_match_on_weak_shared_initial_alone():
    # Sharing only a short/common token must not count as a match.
    assert not name_matches_author_field("A Sharma", "A Verma")


def test_empty_inputs_do_not_match():
    assert not name_matches_author_field("", "Dr. Y. Nithiyanandam")
    assert not name_matches_author_field("Y Nithiyanandam", "")
    assert not name_matches_author_field("Y Nithiyanandam", None)


# ── chunk_author_matches_query: the "mention != authored" guarantee ─────────

def test_chunk_matches_when_author_field_has_the_name():
    chunk = {"title": "Space Debris", "author": "Dr. Y. Nithiyanandam",
             "text": "This piece discusses orbital debris."}
    assert chunk_author_matches_query(chunk, ["Y Nithiyanandam"])


def test_chunk_does_not_match_on_body_text_mention_alone():
    """The core precision guarantee: a document that merely MENTIONS the
    person (interview, quote, citation) must NOT be treated as authored by
    them just because the name appears in the text."""
    chunk = {"title": "Interview with Y Nithiyanandam", "author": "Guest Author",
             "text": "In this interview, Y Nithiyanandam discusses his research."}
    assert not chunk_author_matches_query(chunk, ["Y Nithiyanandam"])


def test_chunk_with_no_author_field_does_not_match():
    chunk = {"title": "Untitled", "text": "Some content."}
    assert not chunk_author_matches_query(chunk, ["Y Nithiyanandam"])


def test_chunk_matches_authors_list_field():
    chunk = {"title": "Co-authored Report", "authors": ["A. Author", "Y. Nithiyanandam"]}
    assert chunk_author_matches_query(chunk, ["Y Nithiyanandam"])


# ── is_authorship_query ──────────────────────────────────────────────────────

def test_is_authorship_query_true_for_written_blogs():
    from src.entity_match import is_authorship_query
    assert is_authorship_query("What blogs has Dr Y Nithiyanandam written?")


def test_is_authorship_query_false_for_identity_question():
    from src.entity_match import is_authorship_query
    assert not is_authorship_query("Who is Dr Y Nithiyanandam?")


def test_is_authorship_query_true_for_authored_variant():
    from src.entity_match import is_authorship_query
    assert is_authorship_query("Which articles has Nithiyanandam authored?")


def test_is_authorship_query_true_for_publications_variant():
    from src.entity_match import is_authorship_query
    assert is_authorship_query("What publications are by Dr Nithiyanandam?")
