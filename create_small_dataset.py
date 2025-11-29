#!/usr/bin/env python3

"""
Create a small synthetic Sudoku dataset for HRM testing.
"""

import os
import json
import numpy as np

from dataset.common import PuzzleDatasetMetadata


def create_simple_sudoku_dataset(output_dir: str = "data/sudoku-small", num_puzzles: int = 10):
    """
    Create a small synthetic dataset with simple Sudoku puzzles.
    Each puzzle is a 9x9 grid, flattened to 81 elements.
    Vocab: 0=PAD, 1-9=numbers
    """

    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    def create_solved_grid():
        # Create a basic solved Sudoku grid (patterned for testing)
        base = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        grid = np.tile(base, (9, 1)).T  # Repeat columns
        # Shuffle rows to maintain validity
        row_perm = np.random.permutation(9)
        grid = grid[row_perm]
        return grid.flatten()

    def remove_cells(grid: np.ndarray, keep_frac: float = 0.3):
        """Create puzzle by randomly removing cells"""
        flat = grid.copy()
        mask = np.random.rand(81) < keep_frac
        flat[~mask] = 0  # Set to 0 (empty)
        return flat

    # Generate datasets
    for split, num_samples in [("train", num_puzzles), ("test", max(num_puzzles // 5, 1))]:
        split_dir = os.path.join(output_dir, split)

        all_inputs = []
        all_labels = []
        puzzle_indices = [0]
        group_indices = [0]

        for i in range(num_samples):
            # Generate solved grid
            solved = create_solved_grid()

            # Create puzzle with some cells removed
            puzzle = remove_cells(solved, keep_frac=np.random.uniform(0.2, 0.5))

            all_inputs.append(puzzle)
            all_labels.append(solved)
            puzzle_indices.append(len(all_inputs))
            group_indices.append(i + 1)

        # Convert to numpy
        inputs = np.array(all_inputs, dtype=np.int32) + 1  # 1-9, adjust for vocab (0=PAD, 1-9=numbers)
        labels = np.array(all_labels, dtype=np.int32) + 1

        # Puzzle identifiers (all same type)
        puzzle_identifiers = np.zeros(num_samples, dtype=np.int32)

        # Save dataset
        np.save(os.path.join(split_dir, "all__inputs.npy"), inputs)
        np.save(os.path.join(split_dir, "all__labels.npy"), labels)
        np.save(os.path.join(split_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)
        np.save(os.path.join(split_dir, "all__puzzle_indices.npy"), np.array(puzzle_indices, dtype=np.int32))
        np.save(os.path.join(split_dir, "all__group_indices.npy"), np.array(group_indices, dtype=np.int32))

        # Metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=81,
            vocab_size=10,  # PAD + 0-9

            pad_id=0,
            ignore_label_id=0,

            blank_identifier_id=0,
            num_puzzle_identifiers=1,

            total_groups=num_samples,
            mean_puzzle_examples=1,
            sets=["all"]
        )

        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)

    # Save identifiers
    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        json.dump(["sudoku"], f)

    print("f Created synthetic dataset:")
    print(f"  Train: {num_puzzles} puzzles")
    print(f"  Test: {max(num_puzzles // 5, 1)} puzzles")
    print(f"  Location: '{output_dir}'")


if __name__ == "__main__":
    create_simple_sudoku_dataset()
