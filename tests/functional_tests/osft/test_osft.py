# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shutil

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "osft"
OSFT_FSDP2_CHECKPOINT_FILENAME = "L2_OSFT_FSDP2_Checkpoint.sh"


class TestOSFT:
    def test_osft_fsdp2_checkpoint(self):
        try:
            run_test_script(TEST_FOLDER, OSFT_FSDP2_CHECKPOINT_FILENAME)
        finally:
            shutil.rmtree("checkpoints_osft/", ignore_errors=True)
