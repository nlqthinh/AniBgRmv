from bg_engine import BgRemoveEngine

engine = BgRemoveEngine(
    checkpoint_path="models/F2_A1_mild_aug_from_FP3_768_unet_mit_b1_3k5_best.pth",
    size=768,
)

result = engine.process_one(
    image_path="test_images/Image (2).PNG",
    output_dir="outputs",
)

print(result)