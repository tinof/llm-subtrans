import tempfile
from pathlib import Path

from PySubtrans.Helpers.TestCases import LoggedTestCase
from PySubtrans.MKV.SubtitleFilter import remove_markup_only_cues


class TestRemoveMarkupOnlyCues(LoggedTestCase):
    """
    Tests for dropping cues that filtering left with nothing a viewer would see
    """

    def _run(self, content: str) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.srt"
            path.write_text(content, encoding="utf-8")
            removed = remove_markup_only_cues(path)
            return removed, path.read_text(encoding="utf-8")

    def test_removes_positioning_code_only_cue_and_renumbers(self):
        removed, result = self._run(
            "1\n00:00:01,000 --> 00:00:02,000\nHola.\n\n"
            "2\n00:01:41,735 --> 00:01:46,439\n{\\an8}\n\n"
            "3\n00:02:02,025 --> 00:02:04,758\n{\\an8}Adiós.\n"
        )

        self.assertLoggedEqual("cues removed", 1, removed)
        self.assertLoggedEqual(
            "remaining cues",
            "1\n00:00:01,000 --> 00:00:02,000\nHola.\n\n"
            "2\n00:02:02,025 --> 00:02:04,758\n{\\an8}Adiós.\n",
            result,
        )

    def test_removes_cue_with_only_empty_html_tags(self):
        removed, _ = self._run(
            "1\n00:00:01,000 --> 00:00:02,000\n<i></i>\n\n2\n00:00:03,000 --> 00:00:04,000\nHola.\n"
        )

        self.assertLoggedEqual("cues removed", 1, removed)

    def test_leaves_file_untouched_when_every_cue_has_text(self):
        content = (
            "1\n00:00:01,000 --> 00:00:02,000\n{\\an8}Hola.\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n<i>Adiós.</i>\n"
        )
        removed, result = self._run(content)

        self.assertLoggedEqual("cues removed", 0, removed)
        self.assertLoggedEqual("file content", content, result)
