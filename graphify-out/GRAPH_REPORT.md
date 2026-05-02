# Graph Report - Ferrum  (2026-05-02)

## Corpus Check
- 16 files · ~5,962 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 124 nodes · 236 edges · 7 communities detected
- Extraction: 66% EXTRACTED · 34% INFERRED · 0% AMBIGUOUS · INFERRED: 80 edges (avg confidence: 0.79)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]

## God Nodes (most connected - your core abstractions)
1. `encode()` - 19 edges
2. `move_to_index()` - 14 edges
3. `FerrumDataset` - 11 edges
4. `FerrumNet` - 11 edges
5. `make_model()` - 9 edges
6. `_game_passes_filter()` - 8 edges
7. `legal_move_mask()` - 8 edges
8. `build_optimizer()` - 8 edges
9. `train()` - 8 edges
10. `make_model()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `Entry point for Ferrum Phase 1 training.  Usage:     python train_pipeline.py --` --uses--> `FerrumNet`  [INFERRED]
  train_pipeline.py → model.py
- `Overfit to a single batch — loss should decrease over 20 steps.` --uses--> `FerrumNet`  [INFERRED]
  tests\test_trainer.py → model.py
- `_extract_positions()` --calls--> `encode()`  [INFERRED]
  data_loader.py → board_encoder.py
- `probe_elo_vs_stockfish()` --calls--> `encode()`  [INFERRED]
  evaluator.py → board_encoder.py
- `test_output_shape_and_dtype()` --calls--> `encode()`  [INFERRED]
  tests\test_encoder.py → board_encoder.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.15
Nodes (22): encode(), encode_flip(), _flip_move_index(), Return (horizontally-mirrored tensor, mirrored move index)., test_black_king_on_e8(), test_black_pawns_on_rank_7(), test_castling_plane_no_rights(), test_castling_plane_starting() (+14 more)

### Community 1 - "Community 1"
Cohesion: 0.16
Nodes (16): Dataset, FerrumDataset, _game_passes_filter(), _time_control_type(), _games(), _make_h5(), test_augment_preserves_policy_and_mask(), test_augment_train_only() (+8 more)

### Community 2 - "Community 2"
Cohesion: 0.18
Nodes (15): probe_elo_vs_stockfish(), index_to_move(), legal_move_mask(), move_to_index(), _sign(), test_all_legal_moves_set_in_mask(), test_mask_count_matches_legal_moves(), test_mask_count_starting_position() (+7 more)

### Community 3 - "Community 3"
Cohesion: 0.3
Nodes (15): build_optimizer(), build_scheduler(), evaluate_accuracy(), train(), training_step(), make_batch(), make_model(), Overfit to a single batch — loss should decrease over 20 steps. (+7 more)

### Community 4 - "Community 4"
Cohesion: 0.21
Nodes (11): load_checkpoint(), ResBlock, save_checkpoint(), make_model(), test_checkpoint_architecture_mismatch(), test_checkpoint_roundtrip(), test_ferrumnet_policy_shape(), test_ferrumnet_value_shape() (+3 more)

### Community 5 - "Community 5"
Cohesion: 0.21
Nodes (11): _extract_positions(), make_dataloader(), _outcome_value(), parse_pgn_to_hdf5(), _process_pgn_file(), Parse PGN files in parallel and write train/val/test splits to one HDF5 file., Worker function: parse one PGN file and return list of per-game position tuples., Write positions from a list of games into an HDF5 group. (+3 more)

### Community 6 - "Community 6"
Cohesion: 0.31
Nodes (9): compute_perplexity(), FerrumNet, make_loader(), make_model(), Overfit a tiny model on one batch; perplexity should drop., probe_elo_vs_stockfish must be importable and have the right signature., test_perplexity_decreases_after_overfit(), test_perplexity_is_positive() (+1 more)

## Knowledge Gaps
- **4 isolated node(s):** `Return (horizontally-mirrored tensor, mirrored move index).`, `Worker function: parse one PGN file and return list of per-game position tuples.`, `Write positions from a list of games into an HDF5 group.`, `Parse PGN files in parallel and write train/val/test splits to one HDF5 file.`
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Community 5` to `Community 3`, `Community 4`, `Community 6`?**
  _High betweenness centrality (0.421) - this node is a cross-community bridge._
- **Why does `_extract_positions()` connect `Community 5` to `Community 0`, `Community 2`?**
  _High betweenness centrality (0.379) - this node is a cross-community bridge._
- **Why does `FerrumNet` connect `Community 6` to `Community 3`, `Community 4`, `Community 5`?**
  _High betweenness centrality (0.270) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `encode()` (e.g. with `_extract_positions()` and `probe_elo_vs_stockfish()`) actually correct?**
  _`encode()` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `move_to_index()` (e.g. with `_extract_positions()` and `test_encode_flip_mirrors_king()`) actually correct?**
  _`move_to_index()` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `FerrumDataset` (e.g. with `test_ferrum_dataset_getitem()` and `test_augment_preserves_policy_and_mask()`) actually correct?**
  _`FerrumDataset` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `FerrumNet` (e.g. with `Entry point for Ferrum Phase 1 training.  Usage:     python train_pipeline.py --` and `Overfit a tiny model on one batch; perplexity should drop.`) actually correct?**
  _`FerrumNet` has 8 INFERRED edges - model-reasoned connections that need verification._