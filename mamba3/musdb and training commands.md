rm musdb18hq.zip

echo "✅ MUSDB18-HQ successfully downloaded and extracted"
echo "   Location: /workspace/datasets/musdb18hq"
echo "   Training tracks: $(ls musdb18hq/train | wc -l)"
echo "   Test tracks: $(ls musdb18hq/test | wc -l)"
echo "   Total size: $(du -sh musdb18hq | cut -f1)"


python train.py \
    --data /workspace/Datasets/train \
    --save ./checkpoints \
    --batch 128 \
    --grad-accum 2 \
    --num-workers 8 \
    --cache-in-ram \
    --device cuda \
    --steps 100000 \
    --ckpt-every 1000 \
    --val-every 2000 \
    --log-every 50 \
    --resume ./checkpoints/checkpoint_006500.pt \
    2>&1 | tee logs/train_h200_full.log" C-m