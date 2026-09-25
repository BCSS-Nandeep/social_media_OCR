"""Pure tests for src/video/sampler.py -- duration -> frame-count policy,
centered timestamp sampling, and >10-minute chunking. No I/O, no ffmpeg, no
GPU: these are plain functions of a float, tested against the integration
spec's own table."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.video import sampler  # noqa: E402


class FrameCountForTests(unittest.TestCase):
    # (duration_seconds, expected_frame_count_or_None)
    CASES = [
        (5, 4), (10, 4),
        (20, 6), (30, 6),
        (45, 8), (60, 8),
        (90, 10), (120, 10),
        (180, 12), (300, 12),
        (600, 16),
        (601, None),
    ]

    def test_duration_to_frame_count_table(self):
        for duration, expected in self.CASES:
            with self.subTest(duration=duration):
                self.assertEqual(sampler.frame_count_for(duration), expected)

    def test_boundary_values_use_the_lower_bracket(self):
        # "0-10 sec -> 4", ">10-30 sec -> 6": exactly 10.0 is still 4, not 6.
        self.assertEqual(sampler.frame_count_for(10.0), 4)
        self.assertEqual(sampler.frame_count_for(10.0001), 6)
        self.assertEqual(sampler.frame_count_for(600.0), 16)
        self.assertEqual(sampler.frame_count_for(600.0001), None)


class SampleTimestampsTests(unittest.TestCase):
    def test_30_seconds_6_frames_matches_spec_example(self):
        result = sampler.sample_timestamps(30.0, 6)
        expected = [2.5, 7.5, 12.5, 17.5, 22.5, 27.5]
        for got, want in zip(result, expected):
            self.assertAlmostEqual(got, want, places=6)

    def test_60_seconds_8_frames_matches_spec_example(self):
        result = sampler.sample_timestamps(60.0, 8)
        expected = [3.75, 11.25, 18.75, 26.25, 33.75, 41.25, 48.75, 56.25]
        for got, want in zip(result, expected):
            self.assertAlmostEqual(got, want, places=6)

    def test_timestamps_are_strictly_increasing(self):
        result = sampler.sample_timestamps(123.4, 10)
        self.assertEqual(result, sorted(result))
        self.assertEqual(len(set(result)), len(result))

    def test_timestamps_are_evenly_spaced(self):
        result = sampler.sample_timestamps(90.0, 9)
        gaps = [b - a for a, b in zip(result, result[1:])]
        for gap in gaps:
            self.assertAlmostEqual(gap, gaps[0], places=6)

    def test_first_and_last_sample_are_not_on_the_boundary(self):
        # Centered sampling: first sample is duration/(2*count) in, not 0;
        # last sample is duration/(2*count) before the end, not duration.
        result = sampler.sample_timestamps(10.0, 4)
        self.assertGreater(result[0], 0.0)
        self.assertLess(result[-1], 10.0)

    def test_rejects_non_positive_count(self):
        with self.assertRaises(ValueError):
            sampler.sample_timestamps(30.0, 0)


class ChunkWindowsTests(unittest.TestCase):
    def test_eleven_minute_video_makes_two_chunks(self):
        windows = sampler.chunk_windows(11 * 60.0)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0], (0.0, 600.0))
        self.assertEqual(windows[1], (600.0, 660.0))

    def test_exactly_ten_minutes_is_one_window(self):
        # frame_count_for already returns 16 (not None) at exactly 600s, so
        # chunk_windows is never called for this case in practice -- but the
        # function itself should still behave sanely if it is.
        windows = sampler.chunk_windows(600.0)
        self.assertEqual(windows, [(0.0, 600.0)])

    def test_twenty_minute_video_makes_two_equal_chunks(self):
        windows = sampler.chunk_windows(20 * 60.0)
        self.assertEqual(windows, [(0.0, 600.0), (600.0, 1200.0)])

    def test_windows_cover_the_whole_duration_with_no_gaps_or_overlap(self):
        duration = 17 * 60.0 + 23.0
        windows = sampler.chunk_windows(duration)
        self.assertEqual(windows[0][0], 0.0)
        self.assertEqual(windows[-1][1], duration)
        for (_, end_a), (start_b, _) in zip(windows, windows[1:]):
            self.assertEqual(end_a, start_b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
