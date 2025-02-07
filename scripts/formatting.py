def create_table(headers: list, rows: list, alignments: list = None) -> str:
    """Create a text-based table with headers and rows"""
    if alignments is None:
        alignments = ["left"] * len(headers)

    def split_number(value):
        try:
            value = str(value).replace(",", "")
            if "." in value:
                left, right = value.split(".")
                return left, right
            return value, ""
        except:
            return value, ""

    # Find maximum widths for decimal alignment
    max_left_width = 0
    max_right_width = 0
    for row in rows:
        if row[1]:  # Skip empty values
            left, right = split_number(str(row[1]))
            max_left_width = max(max_left_width, len(left))
            max_right_width = max(max_right_width, len(right))

    # Calculate column widths
    column_widths = [
        35,
        max_left_width + (max_right_width > 0 and max_right_width + 1 or 0) + 2,
    ]

    def format_cell(content, width, alignment, col_idx):
        content = str(content)
        if alignment == "right" and col_idx == 1 and content:  # Number column
            return content.rjust(width)
        elif alignment == "center":
            return content.center(width)
        return content.ljust(width)

    separator = "+" + "+".join(["-" * width for width in column_widths]) + "+"
    header_row = (
        "|"
        + "|".join(
            format_cell(h, w, "center", i)
            for i, (h, w) in enumerate(zip(headers, column_widths))
        )
        + "|"
    )

    formatted_rows = []
    for row in rows:
        formatted_row = (
            "|"
            + "|".join(
                format_cell(cell, width, align, i)
                for i, (cell, width, align) in enumerate(
                    zip(row, column_widths, alignments)
                )
            )
            + "|"
        )
        formatted_rows.append(formatted_row)

    return "\n".join([separator, header_row, separator, *formatted_rows, separator])
