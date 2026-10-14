"""Regenerate page-pruning fixtures with pyarrow and thriftpy2.

Excluded pages have invalid compressed values or levels. Filtered scans skip those
payloads, while unfiltered scans must fail. Headers and footer remain valid.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import thriftpy2
from thriftpy2.protocol import TCompactProtocol
from thriftpy2.transport import TMemoryBuffer

ROOT = Path(__file__).resolve().parents[3]
PARQUET = thriftpy2.load(
    str(ROOT / "third_party/parquet/parquet.thrift"),
    module_name="page_statistics_thrift",
)
DIRECTORY = Path(__file__).resolve().parent
ROWS = 4096
PAGE_ROWS = 333


def data_pages(data, column):
    transport = TMemoryBuffer(data[column.data_page_offset :])
    offset = column.data_page_offset
    values = 0
    while values < column.num_values:
        before = len(transport.getvalue())
        header = PARQUET.PageHeader()
        header.read(TCompactProtocol(transport))
        offset += before - len(transport.getvalue())
        if header.type == PARQUET.PageType.DATA_PAGE:
            page = header.data_page_header
        else:
            assert header.type == PARQUET.PageType.DATA_PAGE_V2
            page = header.data_page_header_v2
        yield offset, header, page
        transport.read(header.compressed_page_size)
        offset += header.compressed_page_size
        values += page.num_values


def generate(version, dictionary):
    groups = [i // PAGE_ROWS for i in range(ROWS)]
    nullable = [None if i >= 12 * PAGE_ROWS else x for i, x in enumerate(groups)]
    table = pa.table(
        {
            "x": pa.array(groups, type=pa.int32()),
            "s": [f"value_{x:02d}" for x in groups],
            "n": pa.array(nullable, type=pa.int32()),
            "m": pa.array(nullable, type=pa.int32()),
            "id": pa.array(range(ROWS), type=pa.int64()),
        }
    )
    name = f"{'dictionary' if dictionary else 'plain'}_v{version}"
    path = DIRECTORY / f"{name}.parquet"
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=["x", "s", "n", "m"] if dictionary else False,
        data_page_version=f"{version}.0",
        data_page_size=1,
        write_batch_size=PAGE_ROWS,
        row_group_size=ROWS,
    )
    metadata = pq.read_metadata(path)
    assert metadata.num_row_groups == 1
    data = bytearray(path.read_bytes())
    for column_index, page_index in [(0, 1), (1, 1), (2, 1), (3, 12)]:
        pages = list(
            data_pages(bytes(data), metadata.row_group(0).column(column_index))
        )
        assert len(pages) == 13
        offset, header, page = pages[page_index]
        assert page.num_values == (100 if column_index == 3 else PAGE_ROWS)
        assert page.statistics is not None
        if column_index == 3:
            assert page.statistics.null_count == 100
        else:
            assert page.statistics.null_count == 0
        # V2 leaves levels uncompressed; corrupt values, or levels for an all-NULL page.
        if version == 2 and column_index != 3:
            offset += (
                page.definition_levels_byte_length + page.repetition_levels_byte_length
            )
        end = pages[page_index][0] + header.compressed_page_size
        assert offset < end
        data[offset:end] = b"\xff" * (end - offset)
    path.write_bytes(data)


def main():
    for version in (1, 2):
        for dictionary in (False, True):
            generate(version, dictionary)
    table = pa.table(
        {
            "x": pa.array(
                [None if i % 7 == 0 else i for i in range(ROWS)], type=pa.int32()
            ),
            "id": range(ROWS),
        }
    )
    pq.write_table(
        table,
        DIRECTORY / "no_statistics.parquet",
        write_statistics=False,
        use_dictionary=False,
        data_page_size=1,
        write_batch_size=PAGE_ROWS,
        compression="zstd",
    )

    special_floats = {1: float("nan"), 2: float("inf"), 3: float("-inf"), 4: None}
    floats = [special_floats.get(i // PAGE_ROWS, float(i)) for i in range(ROWS)]
    nested = [
        (
            None
            if i % 11 == 0
            else [] if i % 7 == 0 else [i // PAGE_ROWS, i // PAGE_ROWS + 1]
        )
        for i in range(ROWS)
    ]
    structs = [None if i % 13 == 0 else {"v": i // PAGE_ROWS} for i in range(ROWS)]
    table = pa.table({"f": floats, "l": nested, "s": structs, "id": range(ROWS)})
    pq.write_table(
        table,
        DIRECTORY / "edge_cases.parquet",
        data_page_size=128,
        write_batch_size=PAGE_ROWS,
        compression="zstd",
    )

    table = pa.table(
        {
            "x": [i % 4 for i in range(ROWS)],
            "s": [f"value_{i // 4 % 4:02d}" for i in range(ROWS)],
            "id": range(ROWS),
        }
    )
    pq.write_table(
        table,
        DIRECTORY / "large_dictionary_pages.parquet",
        compression="zstd",
        use_dictionary=["x", "s"],
        data_page_size=16384,
        write_batch_size=ROWS,
    )


if __name__ == "__main__":
    main()
