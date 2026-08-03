"""Restore human reading order over the detected regions.

PaddleOCR emits boxes in detector order, which is roughly top-to-bottom but not
reliable across columns or around large display type. Posters also mix font
sizes freely, so a fixed pixel tolerance does not work -- the line-grouping
tolerance here is derived from each block's own height.
"""

from __future__ import annotations

from .ocr_engine import TextBlock


def group_into_lines(blocks: list[TextBlock], overlap_ratio: float = 0.5) -> list[list[TextBlock]]:
    """Cluster blocks into visual lines by vertical overlap, then sort each line L->R.

    Two blocks share a line when their vertical spans overlap by at least
    ``overlap_ratio`` of the shorter block's height. That keeps a small caption
    from being absorbed into an adjacent headline.
    """
    if not blocks:
        return []

    remaining = sorted(blocks, key=lambda b: (b.y_min, b.x_min))
    lines: list[list[TextBlock]] = []

    for block in remaining:
        placed = False
        for line in lines:
            anchor = line[-1]
            span = min(anchor.y_max, block.y_max) - max(anchor.y_min, block.y_min)
            shorter = max(1.0, min(anchor.height, block.height))
            if span / shorter >= overlap_ratio:
                line.append(block)
                placed = True
                break
        if not placed:
            lines.append([block])

    for line in lines:
        line.sort(key=lambda b: b.x_min)
    lines.sort(key=lambda line: min(b.y_min for b in line))
    return lines


def order_blocks(blocks: list[TextBlock]) -> tuple[list[TextBlock], list[int]]:
    """Return blocks in reading order plus the 1-based line number of each."""
    ordered: list[TextBlock] = []
    line_numbers: list[int] = []
    for line_no, line in enumerate(group_into_lines(blocks), start=1):
        for block in line:
            ordered.append(block)
            line_numbers.append(line_no)
    return ordered, line_numbers


def render_text(blocks: list[TextBlock], line_numbers: list[int], joiner: str = " ") -> str:
    """Flatten ordered blocks to plain text, one output line per visual line."""
    lines: list[str] = []
    current: list[str] = []
    previous = None

    for block, line_no in zip(blocks, line_numbers):
        if previous is not None and line_no != previous:
            lines.append(joiner.join(current))
            current = []
        current.append(block.text)
        previous = line_no

    if current:
        lines.append(joiner.join(current))
    return "\n".join(lines)
