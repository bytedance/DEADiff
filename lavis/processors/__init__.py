"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.processors.base_processor import BaseProcessor

from lavis.processors.blip_processors import (
    BlipImageTrainProcessor,
    BlipImageEvalProcessor,
    BlipCaptionProcessor,
)

from lavis.common.registry import registry

__all__ = [
    "BaseProcessor",
    # BLIP
    "BlipImageTrainProcessor",
    "BlipImageEvalProcessor",
    "BlipCaptionProcessor",
]
