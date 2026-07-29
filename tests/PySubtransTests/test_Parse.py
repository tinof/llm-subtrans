import unittest
from enum import Enum

import regex

from PySubtrans.Helpers import GetValueName, GetValueFromName
from PySubtrans.Helpers.Parse import ParseDelayFromHeader, ParseNames
from PySubtrans.Helpers.TestCases import LoggedTestCase
from PySubtrans.TranslationParser import default_pattern, fallback_patterns


class TestParseDelayFromHeader(LoggedTestCase):
    test_cases = [
        ("5", 5.0),
        ("10s", 10.0),
        ("5m", 300.0),
        ("500ms", 1.0),
        ("1500ms", 1.5),
        ("abc", 32.1),
    ]

    def test_ParseDelayFromHeader(self):
        for value, expected in self.test_cases:
            with self.subTest(value=value):
                result = ParseDelayFromHeader(value)
                self.assertLoggedEqual(
                    f"delay parsed from {value}", expected, result, input_value=value
                )


class TestParseNames(LoggedTestCase):
    test_cases = [
        ("John, Jane, Alice", ["John", "Jane", "Alice"]),
        (["John", "Jane", "Alice"], ["John", "Jane", "Alice"]),
        ("Mike, Murray, Mabel, Marge", ["Mike", "Murray", "Mabel", "Marge"]),
        ("", []),
        ([], []),
        ([""], []),
    ]

    def test_ParseNames(self):
        for value, expected in self.test_cases:
            with self.subTest(value=value):
                result = ParseNames(value)
                self.assertLoggedSequenceEqual(
                    f"names parsed from {value}",
                    expected,
                    result,
                    input_value=value,
                )


class TestParseValues(LoggedTestCase):
    class TestEnum(Enum):
        Test1 = 1
        Test2 = 2
        TestValue = 4
        TestExample = 5

    class TestObject:
        def __init__(self, name):
            self.name = name

    get_value_name_cases = [
        (12345, "12345"),
        (True, "True"),
        ("Test", "Test"),
        ("TEST", "TEST"),
        ("TestName", "TestName"),
        (TestEnum.Test1, "Test1"),
        (TestEnum.Test2, "Test2"),
        (TestEnum.TestValue, "Test Value"),
        (TestEnum.TestExample, "Test Example"),
        (TestObject("Test Object"), "Test Object"),
    ]

    def test_GetValueName(self):
        for value, expected in self.get_value_name_cases:
            with self.subTest(value=value):
                result = GetValueName(value)
                self.assertLoggedEqual(
                    f"name for {value}", expected, result, input_value=value
                )

    get_value_from_name_cases = [
        (
            "Test Name",
            ["Test Name", "Another Name", "Yet Another Name"],
            None,
            "Test Name",
        ),
        (
            "Nonexistent Name",
            ["Test Name", "Another Name", "Yet Another Name"],
            "Default Value",
            "Default Value",
        ),
        (34567, [12345, 34567, 98765], None, 34567),
        ("12345", [12345, 34567, 98765], None, 12345),
        ("Test2", TestEnum, None, TestEnum.Test2),
    ]

    def test_GetValueFromName(self):
        for value, names, default, expected in self.get_value_from_name_cases:
            with self.subTest(value=value):
                result = GetValueFromName(value, names, default)
                self.assertLoggedEqual(
                    "value from name",
                    expected,
                    result,
                    input_value=(value, names, default),
                )


class TestTranslationLinePatterns(LoggedTestCase):
    """
    The line header may carry a trailing annotation (e.g. "#12 [2.4s, max 36 chars]") when
    include_line_timings is enabled, and models frequently echo it back in their response.
    """

    # (name, response, expected [(number, body)])
    match_cases = [
        ("plain", "#12\nOriginal>\nHi\nTranslation>\nHei", [("12", "Hei")]),
        ("no_original", "#12\nTranslation>\nHei", [("12", "Hei")]),
        ("annotated", "#12 [2.4s, max 36 chars]\nTranslation>\nHei", [("12", "Hei")]),
        (
            "annotated_with_original",
            "#12 [2.4s, max 36 chars]\nOriginal>\nHi\nTranslation>\nHei",
            [("12", "Hei")],
        ),
        (
            "annotated_multiple_lines",
            "#12 [2.4s]\nTranslation>\nEka\n#13 [1.8s]\nTranslation>\nToka",
            [("12", "Eka"), ("13", "Toka")],
        ),
        (
            "annotated_merge_tag",
            "#12 [2.4s]\nTranslation>\n[MERGE 12+13] Yhdistetty",
            [("12", "[MERGE 12+13] Yhdistetty")],
        ),
        (
            "annotated_dialog",
            "#12 [3.0s, max 45 chars]\nTranslation>\n- Huomenta.\n- Moi.",
            [("12", "- Huomenta.\n- Moi.")],
        ),
    ]

    def test_default_pattern_matches_annotated_headers(self):
        for name, response, expected in self.match_cases:
            with self.subTest(case=name):
                matches = [
                    (match.group("number"), match.group("body"))
                    for match in regex.finditer(
                        default_pattern, response, regex.MULTILINE
                    )
                ]
                self.assertLoggedSequenceEqual(
                    f"default_pattern matches for {name}",
                    expected,
                    matches,
                    input_value=response,
                )

    def test_fallback_patterns_tolerate_annotated_headers(self):
        """Every fallback except the last (bare number) must also survive an annotation"""
        with_original = "#12 [2.4s, max 36 chars]\nOriginal>\nHi\nTranslation>\nHei\n\n"
        # fallback_patterns[4] has no Original> branch, so it needs a bare response
        without_original = "#12 [2.4s, max 36 chars]\nTranslation>\nHei\n\n"
        for index, pattern in enumerate(fallback_patterns[:-1]):
            response = without_original if index == 4 else with_original
            with self.subTest(fallback=index):
                match = regex.search(pattern, response, regex.MULTILINE)
                self.assertLoggedIsNotNone(
                    f"fallback_patterns[{index}] matched", match, input_value=response
                )
                assert match is not None
                self.assertLoggedEqual(
                    f"fallback_patterns[{index}] number", "12", match.group("number")
                )

    def test_number_excludes_annotation(self):
        match = regex.search(
            default_pattern, "#7 [1.5s, max 22 chars]\nTranslation>\nHei", regex.MULTILINE
        )
        self.assertLoggedIsNotNone("annotated header matched", match)
        assert match is not None
        self.assertLoggedEqual("captured number", "7", match.group("number"))


if __name__ == "__main__":
    unittest.main()
