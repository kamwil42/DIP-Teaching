#!/bin/bash
# COLMAP 3D reconstruction pipeline
# Usage: bash run_colmap.sh

set -e

DATASET_PATH="data"
IMAGE_PATH="$DATASET_PATH/images"
COLMAP_PATH="$DATASET_PATH/colmap"

mkdir -p "$COLMAP_PATH/sparse"
mkdir -p "$COLMAP_PATH/dense"

echo "=== Step 1: Feature Extraction ==="
colmap feature_extractor \
    --database_path "$COLMAP_PATH/database.db" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1 \
    --FeatureExtraction.use_gpu 0

echo "=== Step 2: Feature Matching ==="
colmap exhaustive_matcher \
    --database_path "$COLMAP_PATH/database.db" \
    --FeatureMatching.use_gpu 0

echo "=== Step 3: Sparse Reconstruction (Bundle Adjustment) ==="
colmap mapper \
    --database_path "$COLMAP_PATH/database.db" \
    --image_path "$IMAGE_PATH" \
    --output_path "$COLMAP_PATH/sparse"

echo "=== Step 4: Export to PLY ==="
colmap model_converter \
    --input_path "$COLMAP_PATH/sparse/0" \
    --output_path "$COLMAP_PATH/sparse/0/sparse.ply" \
    --output_type PLY

# Below is code for dense pipeline, which is skipped due to hardware limitations (no CUDA)

# echo "=== Step 5: Image Undistortion ==="
# colmap image_undistorter \
#     --image_path "$IMAGE_PATH" \
#     --input_path "$COLMAP_PATH/sparse/0" \
#     --output_path "$COLMAP_PATH/dense"

# echo "=== Step 6: Dense Reconstruction (Patch Match Stereo) ==="
# colmap patch_match_stereo \
#     --workspace_path "$COLMAP_PATH/dense"

# echo "=== Step 7: Stereo Fusion ==="
# colmap stereo_fusion \
#     --workspace_path "$COLMAP_PATH/dense" \
#     --output_path "$COLMAP_PATH/dense/fused.ply"

echo "=== Done! ==="
echo "Results:"
echo "  Sparse: $COLMAP_PATH/sparse/0/"
# echo "  Dense:  $COLMAP_PATH/dense/fused.ply"
