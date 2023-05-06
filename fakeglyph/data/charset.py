# See https://www.unicode.org/charts/


def unicode_letters_in_range(start: str, end: str) -> str:
    ids = range(ord(start), ord(end) + 1)
    letters = "".join(chr(idx) for idx in ids)
    return letters


def cjk_unified_ideographs() -> str:
    return unicode_letters_in_range("\u4E00", "\u9FFF")


def cjk_extension() -> str:
    ranges = [
        ("\U00004E00", "\U00009FFF"),  # Base
        ("\U00003400", "\U00004D8F"),  # Extension A
        ("\U00020000", "\U0002A6DF"),  # Extension B
        ("\U0002A700", "\U0002B739"),  # Extension C
        ("\U0002B740", "\U0002B81D"),  # Extension D
        ("\U0002B820", "\U0002CEA1"),  # Extension E
        ("\U0002CEB0", "\U0002EBE0"),  # Extension F
        ("\U00030000", "\U0003134A"),  # Extension G
        ("\U00031350", "\U000323AF"),  # Extension H
    ]
    letters = "".join(unicode_letters_in_range(*r) for r in ranges)
    return letters
