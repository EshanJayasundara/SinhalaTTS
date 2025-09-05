import pytest
from g2p import (
    apply_consonant_rule_01,
    apply_consonant_rule_02,
    apply_vowel_rules,
    apply_diphthong_rules,
    complete_g2p_conversion
)


# -----------------------
# Consonant Rule Tests
# -----------------------

def test_consonant_rule_01_rsy_duplication(): #Passed
    # චිත්‍ර -> /citrə/ -> /cittrə/
    inp = ["c", "i", "t", "rə"]
    out = apply_consonant_rule_01(inp)
    assert "tt" in "".join(out) or out.count("t") >= 2

def test_consonant_rule_01_ysy_duplication(): #Passed
    # අනය -> /anjə/ -> /annjə/
    inp = ["a", "ɲ", "j", "ə"]
    out = apply_consonant_rule_01(inp)
    assert out.count("ɲ") >= 2 or out.count("j") >= 2

def test_consonant_rule_01_exception_first_syllable():
    # ශ්‍රම -> /ʃrəmə/ -> /ʃrəmə/ (no duplication)
    inp = ["ʃ", "rə", "mə"]
    out = apply_consonant_rule_01(inp)
    assert out == inp

def test_consonant_rule_02_add_k_before_ɲ(): #Passed
    # ප්‍රඥා -> /prəɲa:/ -> /prəkɲa:/
    inp = ["p", "r", "ə", "ɲ", "a:"]
    out = apply_consonant_rule_02(inp)
    assert out[:4] == ["p", "r", "ə", "k"]


# -----------------------
# Vowel Rule Tests
# -----------------------

def test_vowel_rule_01_first_syllable_schwa():
    # මම -> /məmə/ -> /mama/
    inp = ["mə", "mə"]
    out = apply_vowel_rules(inp)
    assert "a" in out[0]

def test_vowel_rule_02_raha(): #Passed
    # consonant + rə h -> consonant + ra h
    inp = ["k", "rə", "h"]
    out = apply_vowel_rules(inp)
    assert "ra" in "".join(out)

def test_vowel_rule_03_h_schwa(): #Passed
    # [a, e, æ, o, ə] h ə -> [a, e, æ, o, ə] h a
    inp = ["a", "h", "ə"]
    out = apply_vowel_rules(inp)
    assert out[-1] == "a"

def test_vowel_rule_04_schwa_cluster(): #Passed
    # /əCC/ -> /aCC/
    inp = ["ə", "t", "r"]
    out = apply_vowel_rules(inp)
    assert out[0] == "a"

def test_vowel_rule_05_word_final_schwa(): #Passed
    # ə + final consonant (not r, b, ɖ, ʈ) -> a
    inp = ["k", "ə", "t"]
    out = apply_vowel_rules(inp)
    assert "a" in out

def test_vowel_rule_06_final_schwa_before_ji_vu(): #Passed
    inp = ["k", "ə", "ji"]
    out = apply_vowel_rules(inp)
    assert out[1] == "a"

def test_vowel_rule_07_keru_case(): #Passed
    inp = ["k", "ə", "r", "u"]
    out = apply_vowel_rules(inp)
    assert out[1] == "a"

def test_vowel_rule_08_kal_special(): #Passed
    inp = ["kal", "a:", "j"]
    out = apply_vowel_rules(inp)
    assert out[0].startswith("kəl")


# -----------------------
# Diphthong Rule Tests
# -----------------------

@pytest.mark.parametrize("inp, expected", [
    (["i", "vu"], "iu"),
    (["i:", "vu"], "i:u"),
    (["e", "vu"], "eu"),
    (["e:", "vu"], "e:u"),
    (["æ", "vu"], "æu"),
    (["æ:", "vu"], "æ:u"),
    (["o", "vu"], "ou"),
    (["a", "vu"], "au"),
    (["a:", "vu"], "a:u"),
    (["u", "yi"], "ui"),
    (["u:", "yi"], "u:i"),
    (["o", "yi"], "oi"),
    (["o:", "yi"], "o:i"),
    (["a", "yi"], "ai"),
    (["a:", "yi"], "a:i"),
    (["e", "yi"], "ei"),
    (["e:", "yi"], "e:i"),
    (["æ", "yi"], "æi"),
    (["æ:", "yi"], "æ:i"),
])
def test_diphthong_rules(inp, expected): #Passed
    out = apply_diphthong_rules(inp)
    assert expected in out


# -----------------------
# Full Pipeline Smoke Test
# -----------------------

def test_complete_pipeline(monkeypatch):
    # Simple stub to check rule chain executes
    monkeypatch.setattr("g2p.rough_g2p_conversion", lambda w: ["k", "ə", "r", "u"])
    result = complete_g2p_conversion("කරු")
    assert "a" in result or "u" in result
