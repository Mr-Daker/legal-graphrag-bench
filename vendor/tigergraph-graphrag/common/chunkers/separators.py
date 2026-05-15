# Copyright (c) 2024-2026 TigerGraph, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

TEXT_SEPARATORS = [
    "\n\n",
    "\n",
    " ",
    "\u3002",  # CJK full stop (。)
    "\uff0c",  # CJK comma (，)
    "\u3001",  # CJK enumeration comma (、)
    "\uff1b",  # CJK semicolon (；)
    "\uff01",  # CJK exclamation mark (！)
    "\uff1f",  # CJK question mark (？)
    "",
]
