#! /bin/bash
# style transfer
python3 scripts/inference.py --from_file assets/style_prompts.txt --ref_images assets/style_images \
        --subject_text "style" --ckpt pretrained/deadiff_v1.ckpt \
        --outdir outputs/styles --n_samples 1 --n_rows 1 \
        --ddim_steps 50 \
        --config configs/inference_deadiff_512x512.yaml

# semantics transfer
python3 scripts/inference.py --from_file assets/content_prompts.txt --ref_images assets/content_images \
        --subject_text "content" --ckpt pretrained/deadiff_v1.ckpt \
        --outdir outputs/semantics --n_samples 1 --n_rows 1 \
        --ddim_steps 50 \
        --config configs/inference_deadiff_512x512.yaml