from faireval.output_protocol import analyze_ranking_output


CANDIDATES = ("i001", "i002", "i003", "i004")


def test_exact_json_is_strict_and_semantically_valid():
    result = analyze_ranking_output(
        '{"ranked_item_ids":["i001","i002"]}',
        candidate_ids=CANDIDATES,
        k=2,
    )
    assert result.strict_format_valid
    assert result.semantic_ranking_valid
    assert result.ranking == ("i001", "i002")
    assert result.forensic_class == "A_strict_exact_json_valid"


def test_fenced_json_is_semantically_valid_but_not_strict():
    result = analyze_ranking_output(
        '```json\n{"ranked_item_ids":["i001","i002"]}\n```',
        candidate_ids=CANDIDATES,
        k=2,
    )
    assert not result.strict_format_valid
    assert result.semantic_ranking_valid
    assert result.ranking == ("i001", "i002")
    assert result.format_violations == ("markdown_fence",)
    assert result.forensic_class == "B_deterministic_envelope_recoverable"


def test_prose_plus_one_object_is_mechanically_recoverable():
    result = analyze_ranking_output(
        'Result: {"ranked_item_ids":["i001","i002"]}',
        candidate_ids=CANDIDATES,
        k=2,
    )
    assert result.semantic_ranking_valid
    assert "extra_text" in result.format_violations
    assert "extract_single_unambiguous_json_object" in result.normalization_actions


def test_multiple_embedded_objects_are_ambiguous_and_rejected():
    result = analyze_ranking_output(
        '{"ranked_item_ids":["i001","i002"]} then {"ranked_item_ids":["i003","i004"]}',
        candidate_ids=CANDIDATES,
        k=2,
    )
    assert not result.semantic_ranking_valid
    assert result.parser_ambiguity
    assert "parser_ambiguity" in result.semantic_errors


def test_top_level_array_has_distinct_wrong_schema_reason():
    result = analyze_ranking_output(
        '["i001","i002"]',
        candidate_ids=CANDIDATES,
        k=2,
    )
    assert not result.semantic_ranking_valid
    assert "wrong_top_level_schema" in result.semantic_errors
    assert "invalid_json_syntax" not in result.semantic_errors


def test_wrong_k_is_not_truncated():
    result = analyze_ranking_output(
        '{"ranked_item_ids":["i001","i002","i003"]}',
        candidate_ids=CANDIDATES,
        k=2,
    )
    assert not result.semantic_ranking_valid
    assert result.ranking is None
    assert "wrong_k" in result.semantic_errors


def test_duplicate_is_rejected():
    result = analyze_ranking_output(
        '{"ranked_item_ids":["i001","i001"]}',
        candidate_ids=CANDIDATES,
        k=2,
    )
    assert not result.semantic_ranking_valid
    assert "duplicate_item_ids" in result.semantic_errors


def test_leading_zero_ids_are_preserved_exactly():
    result = analyze_ranking_output(
        '{"ranked_item_ids":["i001","i002"]}',
        candidate_ids=CANDIDATES,
        k=2,
    )
    assert result.ranking == ("i001", "i002")
    assert not result.candidate_id_mutation_detected

    changed = analyze_ranking_output(
        '{"ranked_item_ids":["i1","i002"]}',
        candidate_ids=CANDIDATES,
        k=2,
    )
    assert not changed.semantic_ranking_valid
    assert "out_of_candidate_item" in changed.semantic_errors


def test_invalid_candidate_is_rejected():
    result = analyze_ranking_output(
        '{"ranked_item_ids":["i001","invented"]}',
        candidate_ids=CANDIDATES,
        k=2,
    )
    assert not result.semantic_ranking_valid
    assert "out_of_candidate_item" in result.semantic_errors


def test_extra_keys_are_format_violation_without_semantic_mutation():
    result = analyze_ranking_output(
        '{"ranked_item_ids":["i001","i002"],"note":"x"}',
        candidate_ids=CANDIDATES,
        k=2,
    )
    assert not result.strict_format_valid
    assert result.semantic_ranking_valid
    assert result.ranking == ("i001", "i002")
    assert "extra_keys" in result.format_violations


def test_non_string_ids_are_not_stringified():
    result = analyze_ranking_output(
        '{"ranked_item_ids":[1,"i002"]}',
        candidate_ids=("1", "i002"),
        k=2,
    )
    assert not result.semantic_ranking_valid
    assert "non_string_item_id" in result.semantic_errors
    assert result.ranking is None
