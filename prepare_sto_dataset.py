#!/usr/bin/env python3

import csv
import shutil
import sys
from pathlib import Path


SOURCE_ROOT = Path("/home/everest/STO_npy")
OUTPUT_ROOT = Path("/home/everest/SV-RCNet_BA_STO/dataset")
MAPPING_CSV = Path("/home/everest/SV-RCNet_BA_STO/split_mapping.csv")

# 원본 번호 -> 새 번호
TRAIN_MAPPING = {
    **{i: i for i in range(1, 14)},
    **{i: i - 1 for i in range(15, 23)},
    40: 22,
    41: 23,
    42: 24,
    43: 25,
    45: 26,
    48: 27,
    49: 28,
}

VAL_MAPPING = {
    44: 29,
    46: 30,
    47: 31,
    23: 32,
}

TEST_MAPPING = {
    24: 33,
    25: 34,
    26: 35,
    28: 36,
    30: 37,
    31: 38,
    32: 39,
    34: 40,
}

# 아래 6개 원본 영상만 metadata 4종과 phase를 함께 복사
FULL_METADATA_SOURCE_IDS = {40, 41, 42, 43, 45, 48}

FULL_FILE_TYPES = [
    "gauze",
    "tool",
    "vessel",
    "organ",
    "phase",
]

PHASE_ONLY_FILE_TYPES = ["phase"]


def case_name(case_id: int) -> str:
    return f"STOa_PS{case_id:03d}_STOs01"


def validate_mapping() -> None:
    all_source_ids = (
        list(TRAIN_MAPPING.keys())
        + list(VAL_MAPPING.keys())
        + list(TEST_MAPPING.keys())
    )

    all_target_ids = (
        list(TRAIN_MAPPING.values())
        + list(VAL_MAPPING.values())
        + list(TEST_MAPPING.values())
    )

    expected_targets = list(range(1, 41))

    if len(TRAIN_MAPPING) != 28:
        raise RuntimeError(
            f"Train 개수 오류: 예상 28개, 실제 {len(TRAIN_MAPPING)}개"
        )

    if len(VAL_MAPPING) != 4:
        raise RuntimeError(
            f"Validation 개수 오류: 예상 4개, 실제 {len(VAL_MAPPING)}개"
        )

    if len(TEST_MAPPING) != 8:
        raise RuntimeError(
            f"Test 개수 오류: 예상 8개, 실제 {len(TEST_MAPPING)}개"
        )

    if len(all_source_ids) != 40:
        raise RuntimeError(
            f"전체 원본 개수 오류: 예상 40개, 실제 {len(all_source_ids)}개"
        )

    if len(set(all_source_ids)) != 40:
        duplicates = sorted(
            source_id
            for source_id in set(all_source_ids)
            if all_source_ids.count(source_id) > 1
        )
        raise RuntimeError(f"중복 원본 번호 발견: {duplicates}")

    if sorted(all_target_ids) != expected_targets:
        raise RuntimeError(
            "새 번호가 1~40으로 연속되지 않습니다.\n"
            f"현재 새 번호: {sorted(all_target_ids)}"
        )

    print("[검증 완료] Train 28 / Val 4 / Test 8 / Total 40")
    print("[검증 완료] 새 번호 1~40 연속")
    print("[검증 완료] 원본 번호 중복 없음")


def expected_files(
    split_name: str,
    mapping: dict[int, int],
) -> list[dict]:
    records = []

    for source_id, target_id in mapping.items():
        source_case = case_name(source_id)
        target_case = case_name(target_id)

        source_dir = SOURCE_ROOT / source_case
        target_dir = OUTPUT_ROOT / split_name / target_case

        file_types = (
            FULL_FILE_TYPES
            if source_id in FULL_METADATA_SOURCE_IDS
            else PHASE_ONLY_FILE_TYPES
        )

        for file_type in file_types:
            source_file = source_dir / f"{source_case}_{file_type}.npy"
            target_file = target_dir / f"{target_case}_{file_type}.npy"

            records.append(
                {
                    "split": split_name,
                    "source_id": source_id,
                    "target_id": target_id,
                    "source_case": source_case,
                    "target_case": target_case,
                    "file_type": file_type,
                    "source_file": source_file,
                    "target_file": target_file,
                }
            )

    return records


def check_source_files(records: list[dict]) -> None:
    missing = []

    for record in records:
        if not record["source_file"].is_file():
            missing.append(record["source_file"])

    if missing:
        print("\n[오류] 다음 원본 파일이 없습니다:")
        for path in missing:
            print(f"  - {path}")

        print(f"\n누락 파일 수: {len(missing)}")
        sys.exit(1)

    print(f"[검증 완료] 필요한 원본 파일 {len(records)}개가 모두 존재합니다.")


def copy_files(records: list[dict]) -> None:
    if OUTPUT_ROOT.exists():
        print(f"[주의] 기존 출력 폴더 삭제: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    copied_count = 0
    copied_size = 0

    for record in records:
        source_file = record["source_file"]
        target_file = record["target_file"]

        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)

        copied_count += 1
        copied_size += target_file.stat().st_size

        print(
            f"[복사] {record['split']:5s} "
            f"{record['source_id']:03d} -> {record['target_id']:03d} "
            f"{record['file_type']}"
        )

    print(f"\n[복사 완료] 파일 수: {copied_count}")
    print(f"[복사 완료] 전체 크기: {copied_size / (1024 ** 3):.3f} GiB")


def write_mapping_csv() -> None:
    rows = []

    split_mappings = [
        ("train", TRAIN_MAPPING),
        ("val", VAL_MAPPING),
        ("test", TEST_MAPPING),
    ]

    for split_name, mapping in split_mappings:
        for source_id, target_id in mapping.items():
            rows.append(
                {
                    "split": split_name,
                    "source_id": source_id,
                    "target_id": target_id,
                    "source_folder": case_name(source_id),
                    "target_folder": case_name(target_id),
                    "included_files": (
                        "gauze,tool,vessel,organ,phase"
                        if source_id in FULL_METADATA_SOURCE_IDS
                        else "phase"
                    ),
                }
            )

    rows.sort(key=lambda row: row["target_id"])

    with MAPPING_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "source_id",
                "target_id",
                "source_folder",
                "target_folder",
                "included_files",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[생성 완료] {MAPPING_CSV}")


def final_validation() -> None:
    split_expectations = {
        "train": 28,
        "val": 4,
        "test": 8,
    }

    print("\n========== 최종 검증 ==========")

    total_dirs = 0
    total_files = 0

    for split_name, expected_count in split_expectations.items():
        split_dir = OUTPUT_ROOT / split_name
        case_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())

        actual_count = len(case_dirs)
        file_count = sum(
            1
            for case_dir in case_dirs
            for path in case_dir.iterdir()
            if path.is_file() and path.suffix == ".npy"
        )

        print(
            f"{split_name:5s}: "
            f"folders={actual_count}/{expected_count}, "
            f"npy_files={file_count}"
        )

        if actual_count != expected_count:
            raise RuntimeError(
                f"{split_name} 폴더 개수 오류: "
                f"예상 {expected_count}, 실제 {actual_count}"
            )

        total_dirs += actual_count
        total_files += file_count

    # phase 40개 + metadata 추가 6개 × 4종 = 총 64개
    expected_file_count = 40 + (6 * 4)

    if total_dirs != 40:
        raise RuntimeError(f"전체 폴더 개수 오류: {total_dirs}")

    if total_files != expected_file_count:
        raise RuntimeError(
            f"전체 npy 파일 개수 오류: "
            f"예상 {expected_file_count}, 실제 {total_files}"
        )

    print(f"total: folders={total_dirs}/40, npy_files={total_files}/64")
    print("[최종 검증 완료] 28:4:8 구성과 파일 개수가 정확합니다.")


def main() -> None:
    validate_mapping()

    records = []

    for split_name, mapping in [
        ("train", TRAIN_MAPPING),
        ("val", VAL_MAPPING),
        ("test", TEST_MAPPING),
    ]:
        records.extend(expected_files(split_name, mapping))

    # 복사 전에 모든 원본 파일을 먼저 검사
    check_source_files(records)

    copy_files(records)
    write_mapping_csv()
    final_validation()


if __name__ == "__main__":
    main()
