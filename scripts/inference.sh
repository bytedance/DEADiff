# Copyright (2024) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 

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