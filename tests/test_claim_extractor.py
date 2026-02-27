"""Tests for src/utils/claim_extractor.py"""

import pytest
from utils.claim_extractor import extract_claims, _looks_like_claim


class TestLooksLikeClaim:
    def test_sentence_with_number_is_claim(self):
        assert _looks_like_claim("The company has 500 employees worldwide.") is True

    def test_sentence_with_factual_marker_is_claim(self):
        assert _looks_like_claim("The vaccine is effective against the virus.") is True

    def test_question_is_not_claim(self):
        assert _looks_like_claim("Is the earth round?") is False

    def test_too_short_is_not_claim(self):
        assert _looks_like_claim("Short.") is False

    def test_too_long_is_not_claim(self):
        long_sentence = "A" * 321
        assert _looks_like_claim(long_sentence) is False

    def test_opinion_marker_excluded(self):
        assert _looks_like_claim("I think the policy will work well for everyone.") is False

    def test_we_believe_excluded(self):
        assert _looks_like_claim("We believe this approach is correct and valid.") is False

    def test_sentence_with_was_is_claim(self):
        assert _looks_like_claim("The president was elected in a democratic process.") is True

    def test_sentence_with_said_is_claim(self):
        assert _looks_like_claim("The minister said the budget will increase next year.") is True

    def test_sentence_with_according_is_claim(self):
        assert _looks_like_claim("According to experts, the results are significant.") is True


class TestExtractClaims:
    def test_empty_string_returns_empty(self):
        assert extract_claims("") == []

    def test_whitespace_only_returns_empty(self):
        assert extract_claims("   \n\t  ") == []

    def test_factual_sentences_are_extracted(self):
        text = "The study was published in 2023. Scientists reported significant findings."
        claims = extract_claims(text)
        assert len(claims) > 0

    def test_opinion_sentences_are_filtered(self):
        # When both factual and opinion sentences exist, opinions are skipped in heuristic pass.
        # A pure-opinion text triggers the fallback (returns first long sentence by design).
        # Test that when factual content is available, opinions are NOT selected over it.
        text = "I think the policy is good. The government reported 1 million vaccinations."
        claims = extract_claims(text)
        # The factual sentence (with number) should be preferred over the opinion
        assert any("million" in c for c in claims)

    def test_max_claims_limit_respected(self):
        sentences = [f"The result was {i} units in the study." for i in range(20)]
        text = " ".join(sentences)
        claims = extract_claims(text, max_claims=3)
        assert len(claims) <= 3

    def test_deduplication(self):
        sentence = "The GDP grew by 3 percent last year."
        text = f"{sentence} {sentence} {sentence}"
        claims = extract_claims(text)
        unique = set(c.lower() for c in claims)
        assert len(unique) == len(claims)

    def test_fallback_returns_first_long_sentence(self):
        # No factual markers or numbers — should fall back
        text = "This is a purely subjective piece of writing about beauty and art."
        claims = extract_claims(text)
        # Fallback should return the first sentence up to 320 chars
        assert len(claims) <= 1

    def test_single_factual_sentence(self):
        text = "NASA confirmed water was found on the moon in 2020."
        claims = extract_claims(text)
        assert len(claims) == 1
        assert "NASA" in claims[0] or "nasa" in claims[0].lower()

    def test_default_max_claims_is_five(self):
        sentences = [f"The company reported {i} billion dollars in revenue." for i in range(10)]
        text = " ".join(sentences)
        claims = extract_claims(text)
        assert len(claims) <= 5

    def test_output_strips_whitespace(self):
        text = "  The report confirmed 90 percent accuracy in trials.  "
        claims = extract_claims(text)
        for claim in claims:
            assert claim == claim.strip()
