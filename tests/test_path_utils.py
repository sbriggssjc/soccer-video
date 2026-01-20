from tools.path_utils import build_output_stem, normalize_stem


def test_normalize_stem_dedupes_cinematic_tags():
    stem = "clip.__CINEMATIC.__CINEMATIC.__CINEMATIC"
    normalized_once = normalize_stem(stem)
    normalized_twice = normalize_stem(normalized_once + ".__CINEMATIC")
    assert normalized_once == normalized_twice
    assert normalized_once.count(".__CINEMATIC") == 1


def test_build_output_stem_dedupes_preset_and_final():
    stem = "clip__SEGMENT_SMOOTH.__CINEMATIC.__CINEMATIC_portrait_FINAL"
    output = build_output_stem(
        stem,
        preset="SEGMENT_SMOOTH",
        portrait=True,
        is_final=True,
        extra_tags=["CINEMATIC"],
    )
    assert output.count("__SEGMENT_SMOOTH") == 1
    assert output.count("_portrait_FINAL") == 1
    assert output.count(".__CINEMATIC") == 1
