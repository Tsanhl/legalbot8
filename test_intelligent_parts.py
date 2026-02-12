"""
Test the intelligent part division system
"""

from gemini_service import detect_long_essay
from gemini_service import LONG_RESPONSE_PART_WORD_CAP

print("=" * 80)
print("INTELLIGENT PART DIVISION SYSTEM - TEST")
print("=" * 80)

test_cases = [
    ("Write a 6000 word essay on contract law", 6000),
    ("Write a 8000 word essay on tort law", 8000),
    ("Write a 10000 word essay on criminal law", 10000),
    ("Write a 12000 word essay on EU law", 12000),
    ("Write a 16000 word essay on human rights", 16000),
    ("Write a 20000 word dissertation on medical law", 20000),
    ("Write a 24000 word dissertation on property law", 24000),
    ("Write a 40000 word thesis on international law", 40000),
]

for prompt, expected_words in test_cases:
    result = detect_long_essay(prompt)
    print(f"\n{'=' * 80}")
    print(f"📝 REQUEST: {expected_words:,} words")
    print(f"{'=' * 80}")
    print(f"Suggested parts: {result['suggested_parts']}")
    print(f"Words per part: ~{result['words_per_part']:,}")
    print(f"Total: {result['suggested_parts']} × {result['words_per_part']:,} = {result['suggested_parts'] * result['words_per_part']:,} words")
    print(f"\n📋 Recommendation Message:")
    print(result['suggestion_message'])
    
    # Validate
    words_per_part = result['words_per_part']
    if words_per_part > LONG_RESPONSE_PART_WORD_CAP:
        raise AssertionError(f"Per-part cap violated: {words_per_part} > {LONG_RESPONSE_PART_WORD_CAP}")
    if expected_words > 2000 and not result['is_long_essay']:
        raise AssertionError("Expected long-essay split for >2,000 words, but is_long_essay=False")
    if expected_words <= 2000 and result['is_long_essay']:
        raise AssertionError("Did not expect split for ≤2,000 words, but is_long_essay=True")
    print(f"\n✅ OK: Each part ≤ {LONG_RESPONSE_PART_WORD_CAP:,} words")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print("\nKey Observations:")
print("- Requests >2,000 words are split into parts")
print("- Each part is capped at 2,000 words")
print("- System provides a per-part plan for continuation")
